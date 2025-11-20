#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/flight/api.h>
#include <arrow/filesystem/api.h>
#include <arrow/ipc/api.h>
#include <arrow/compute/api.h>

#include <util.h>

#include <chrono>
#include <cxxopts.hpp>

class StorageService : public arrow::flight::FlightServerBase
{
public:
	explicit StorageService(std::shared_ptr<arrow::fs::FileSystem> root) 
		: root_(std::move(root)) {}

	arrow::Status ListFlights(const arrow::flight::ServerCallContext&,
														const arrow::flight::Criteria*,
														std::unique_ptr<arrow::flight::FlightListing>* listings) override {
		
		return arrow::Status::OK();
	}

	arrow::Status GetSchema(const arrow::flight::ServerCallContext &context,
													const arrow::flight::FlightDescriptor &request,
													std::unique_ptr<arrow::flight::SchemaResult> *schema) override {
		std::cout << "GetSchema: " << request.path[0] << std::endl;
		auto toget_schema = table_schemas_[request.path[0]];
		ARROW_ASSIGN_OR_RAISE(*schema, 
													arrow::flight::SchemaResult::Make(*toget_schema));

		return arrow::Status::OK();
	}

	arrow::Status DoGet(const arrow::flight::ServerCallContext&,
						const arrow::flight::Ticket& request,
						std::unique_ptr<arrow::flight::FlightDataStream>* stream) override {
		std::string req_name = request.ticket;
		std::cout << "Request: " << req_name << std::endl;
		// The request for record batchs will be in the form of "table_name|col1|col2|...|/0/2"
		// 0: local ID of the shuffle workers for that table
		// 2: number of the shuffle workers for that table
		// The record batches are distributed in round robin
		size_t slash_pos = req_name.find('/');
		std::vector<std::string> used_col_names;
		if (slash_pos != std::string::npos) {
			std::string full_table = req_name.substr(0, slash_pos);
			// fill col_names and return table name
			std::string table = ParseFullTable(full_table, used_col_names);
			if (named_batches_.find(table) == named_batches_.end()) {
				return arrow::Status::Invalid("Request Partition Table Not Exists: ", table);
			}
			int num_batches = named_batches_[table].size();
			auto schema = named_batches_[table][0]->schema();
			auto used_col_indices = GetUsedColIndices(schema, used_col_names);
			
			arrow::RecordBatchVector to_send_batches;
			std::string shfl_id = req_name.substr(slash_pos+1);
			if (shfl_id.empty()) {	// Full table
				for (auto cur_batch : named_batches_[table]) {
					ARROW_ASSIGN_OR_RAISE(auto to_send, DoGetBatch(table, cur_batch, used_col_indices));
					to_send_batches.push_back(to_send);
				}
			}
			else {	// A batch only
				const int batch_idx = std::stoi(shfl_id);
				if (batch_idx < num_batches) {
					auto cur_batch = named_batches_[table][batch_idx];
					ARROW_ASSIGN_OR_RAISE(auto to_send, DoGetBatch(table, cur_batch, used_col_indices));
					to_send_batches.push_back(to_send);
				}
				else {
					return arrow::Status::Invalid("Illegal batch idx: ", batch_idx);
				}
			}
			ARROW_ASSIGN_OR_RAISE(auto batch_reader, arrow::RecordBatchReader::Make(to_send_batches));
			*stream = std::unique_ptr<arrow::flight::FlightDataStream>(
				new arrow::flight::RecordBatchStream(batch_reader)
			);
			return arrow::Status::OK();
		} 
		if (req_name.find("@") != std::string::npos) {
			// TODO: block until preproc_map_rb_.size() is equal to number of all commited workers
			arrow::RecordBatchVector map_rbs;
			for (const auto pair : preproc_map_rb_) {
				map_rbs.push_back(pair.second);
			}
			if (map_rbs.size() == 0) {
				std::shared_ptr<arrow::Schema> dull_schema = arrow::schema({
					arrow::field("no", arrow::int32()),
				});
				map_rbs.push_back(arrow::RecordBatch::MakeEmpty(dull_schema).ValueOrDie());
			}
			ARROW_ASSIGN_OR_RAISE(auto batch_reader, arrow::RecordBatchReader::Make(map_rbs));
			*stream = std::unique_ptr<arrow::flight::FlightDataStream>(
				new arrow::flight::RecordBatchStream(batch_reader)
			);
			return arrow::Status::OK();
		}
		assert(false && "should not happen");
		// [Deprecated] If get a partition from Flight server
		// The request for a partition will be in the form of "table_name:0"
		// size_t sep_pos = req_name.find(':');
		// if (sep_pos != std::string::npos) {
		// 	std::string partition_table = req_name.substr(0, sep_pos);
		// 	int paritition_id = std::stoi(req_name.substr(sep_pos+1));
		// 	// Check
		// 	if (named_batch_partitions_.find(partition_table) == named_batch_partitions_.end()) {
		// 		return arrow::Status::Invalid("Request Partition Table Not Exists: ", req_name);
		// 	}
		// 	else if (paritition_id >= named_batch_partitions_[partition_table][0].size()) {
		// 		return arrow::Status::Invalid("Request Parition ID Not Exists", req_name);
		// 	}
		// 	arrow::RecordBatchVector req_partition;
		// 	for (auto batch_partitions : named_batch_partitions_[partition_table]) {
		// 		req_partition.push_back(batch_partitions[paritition_id]);
		// 	}
		// 	ARROW_ASSIGN_OR_RAISE(auto batch_reader, arrow::RecordBatchReader::Make(req_partition));
		// 	*stream = std::unique_ptr<arrow::flight::FlightDataStream>(
		// 		new arrow::flight::RecordBatchStream(batch_reader)
		// 	);
		// 	return arrow::Status::OK();
		// }
		return arrow::Status::OK();
	}

	const arrow::flight::ActionType kActionReset{"reset", ""};
	const arrow::flight::ActionType kActionOffloadSel{"offload_sel", ""};
	const arrow::flight::ActionType kActionSetOffloadTbl{"set_offload_tbl", ""};
	arrow::Status DoAction(const arrow::flight::ServerCallContext&,
												const arrow::flight::Action& action,
												std::unique_ptr<arrow::flight::ResultStream>* result) override {
	if (action.type == kActionReset.type) {
			table2sel_.clear();
			preproc_map_rb_.clear();
			return arrow::Status::OK();
		}
		else if (action.type == kActionSetOffloadTbl.type) {
			cur_offload_tbl_ = action.body->ToString();
			return arrow::Status::OK();
		}
		else if (action.type == kActionOffloadSel.type) {
			ARROW_ASSIGN_OR_RAISE(auto sel_expr, arrow::compute::Deserialize(action.body));
			table2sel_[cur_offload_tbl_] = sel_expr;
			std::cout << cur_offload_tbl_ << std::endl;
			std::cout << sel_expr.ToString() << std::endl;
			return arrow::Status::OK();
		}
		return arrow::Status::NotImplemented("Unknown action type");
	}
	
	arrow::Result<std::vector<int>>
	GetColIndicesFromName(std::vector<std::string> col_names, std::shared_ptr<arrow::Schema> schema) {
		std::vector<int> indices;
		for (auto col : col_names) {
			int i = schema->GetFieldIndex(col);
			if (i == -1) {
				return arrow::Status::Invalid(col, "does not exist");
			}
			indices.push_back(i);
		}
		return indices;
	}

	// Only used for int2str map now
	arrow::Status DoPut(const arrow::flight::ServerCallContext& context,
											std::unique_ptr<arrow::flight::FlightMessageReader> reader,
											std::unique_ptr<arrow::flight::FlightMetadataWriter>) override {
	auto descriptor = reader->descriptor();
		if (descriptor.type == descriptor.CMD) {	// for preprocess
			// assume that only < 10 sub-queries
			int subq_id = std::stoi(descriptor.cmd.substr(0, 1));
			int worker_id = std::stoi(descriptor.cmd.substr(1));
			ARROW_ASSIGN_OR_RAISE(auto map_rbs, reader->ToRecordBatches());
			assert(map_rbs.size() == 1 && "the map should have only 1 record batch");
			preproc_map_rb_[subq_id*100+worker_id] = map_rbs[0];
		}
		else if (descriptor.type == descriptor.PATH) {	// for result
			const std::string res_name = descriptor.path[0];
			int64_t total_num_rows = 0;
			ARROW_ASSIGN_OR_RAISE(auto res_batches, reader->ToRecordBatches());
			for (const auto batch : res_batches) {
				total_num_rows += batch->num_rows();
			}
			if (res_batches.size() > 1 && !std::getenv("NOT_MERGE_RES") &&
					res_name.find("res_") != std::string::npos) {
				// only merge result of sub-query
				std::cout << "MERGING" << std::endl;
				ARROW_ASSIGN_OR_RAISE(auto res_1batch, ConcatRecordBatches(res_batches));
				named_batches_[res_name] = {res_1batch};
			} else {
				named_batches_[res_name] = res_batches;
			}
			std::cout << "Store " << res_name << " with #batches " << res_batches.size()
								<< " and #rows " << total_num_rows << std::endl;
		}
		
		return arrow::Status::OK();
	}

	arrow::Status LoadData(int max_num_batches, std::vector<std::string> tables, 
												 std::vector<std::vector<std::string>> table_cols,
												 std::string load_schema) {
		assert(tables.size() == table_cols.size());
		for (int t = 0; t < tables.size(); t++) {
			auto table_name = tables[t];
			auto cols = table_cols[t];
			std::cout << "loading " << table_name << std::endl;
			table_names_.push_back(table_name);	
			for (auto col : cols) {
				col2table_[col] = table_name;
			}
			
			ARROW_ASSIGN_OR_RAISE(auto infile, root_->OpenInputFile(table_name +".arrow"));
			ARROW_ASSIGN_OR_RAISE(auto full_batch_reader, arrow::ipc::RecordBatchFileReader::Open(infile));
			ARROW_ASSIGN_OR_RAISE(const std::vector<int> col_indices, 
														GetColIndicesFromName(cols, full_batch_reader->schema()));
			auto read_opt = arrow::ipc::IpcReadOptions::Defaults();
			read_opt.included_fields = col_indices;
			ARROW_ASSIGN_OR_RAISE(auto batch_reader, 
														arrow::ipc::RecordBatchFileReader::Open(infile, read_opt));
			
			arrow::RecordBatchVector batches;
			int num_read_batches = 0;
			if (max_num_batches == 0) {
				num_read_batches = batch_reader->num_record_batches();
			}
			else {
				num_read_batches = (max_num_batches < batch_reader->num_record_batches())?
														max_num_batches : batch_reader->num_record_batches();
			}
			int64_t num_read_rows = 0;
			for (int b = 0; b < num_read_batches; b++) {
				std::cout << b << std::endl;
				ARROW_ASSIGN_OR_RAISE(auto batch, batch_reader->ReadRecordBatch(b))
				batches.push_back(batch);
				num_read_rows += batch->num_rows();
			}
			std::cout << "finish loading" << std::endl;
			
			table_schemas_[table_name] = batches[0]->schema();
			named_batches_[table_name] = batches;
			// concatenate batches (to make sure #build batches is 1)
			bool build_concat = (load_schema == "ssb" && table_name == "customer");
			if (build_concat && batches.size() == 2) {
				ARROW_ASSIGN_OR_RAISE(auto concat_batch, ConcatRecordBatches(batches));
				named_batches_[table_name] = {concat_batch};
			}

			std::cout << table_name << " has " << batch_reader->num_record_batches() 
								<< " batches (read " << num_read_batches << "and " << num_read_rows
								<< " rows), store as " 
								<< named_batches_.at(table_name).size() << " batches" << std::endl;
			std::cout << table_schemas_[table_name]->ToString() << std::endl;
		}
		
		return arrow::Status::OK();
	}

	arrow::Status DoPartition(int num_partition, std::string table, std::vector<std::string> req_cols ) {
		// Get schemas of request cols
		arrow::FieldVector req_fields;
		for (auto col_name : req_cols) {
			auto field = named_batches_[table][0]->schema()->GetFieldByName(col_name);
			assert(field);
			req_fields.push_back(field);
		}
		auto req_schema = arrow::schema(req_fields);
		// Use the first col to partition
		const auto part_col = req_cols[0];
		for (auto batch : named_batches_[table]) {
			assert(batch->GetColumnByName(part_col)->type_id() == arrow::Type::INT32);
			const int64_t num_rows = batch->num_rows();
			// For a input record batch, it may be partitioned to multiple record batches
			arrow::RecordBatchVector partitions(num_partition);
			std::vector<arrow::ArrayVector> partition_arrays(num_partition);
			std::vector<std::vector<std::unique_ptr<arrow::ArrayBuilder>>> partition_builders(num_partition);
			for (int p = 0; p < num_partition; p++) {
				// Prepare the pointer vector
				partition_arrays[p].resize(req_cols.size());
				for (auto col_name : req_cols) {
					auto col = batch->GetColumnByName(col_name);
					ARROW_ASSIGN_OR_RAISE(auto col_builder, arrow::MakeBuilder(col->type()));
					// Resize to sufficient space to be more efficient ?
					ARROW_RETURN_NOT_OK(col_builder->Resize(num_rows/num_partition));
					partition_builders[p].push_back(std::move(col_builder));
				}
			}
			// Scan the batch to distribute
			for (int64_t i = 0; i < batch->num_rows(); i++) {
				auto int_col = std::static_pointer_cast<arrow::Int32Array>(batch->GetColumnByName(part_col));
				int32_t tgt = int_col->Value(i) % num_partition;
				// Distribute columns to the partitions, with a unified scalar
				for (int c = 0; c < req_cols.size(); c++) {
					auto col = batch->GetColumnByName(req_cols[c]);
					ARROW_ASSIGN_OR_RAISE(auto s_p, col->GetScalar(i));
					// auto s = *s_p;
					ARROW_RETURN_NOT_OK(partition_builders[tgt][c]->AppendScalar(*s_p));
				}
				
			}
			// Finish all arrays
			for (int p = 0; p < num_partition; p++) {
				for (int c = 0; c < req_cols.size(); c++) {
					ARROW_ASSIGN_OR_RAISE(partition_arrays[p][c], 
																partition_builders[p][c]->Finish()); 
				}
				// Construct record batches and save
				partitions[p] = arrow::RecordBatch::Make(req_schema, partition_arrays[p][0]->length(),
																								 partition_arrays[p]);
			}

			// Save paritions for current record batch
			named_batch_partitions_[table].push_back(partitions);
			
			// for (int p = 0; p < num_partition; p++) {
			// 	std::cout << "Partition" << p <<" of " << table << std::endl;
			// 	std::cout << "Num rows: " << partitions[p]->num_rows() << std::endl;
			// 	std::cout << partitions[p]->ToString() << std::endl;
			// }
		}

		return arrow::Status::OK();
	}

	arrow::Status DoPartitionByTake(int num_partition, std::string table_name, std::vector<std::string> req_cols) {
		auto schema = named_batches_[table_name][0]->schema();
		std::vector<int> req_cols_idx;
		for (auto req_col : req_cols) {
			for (int i = 0; i < schema->num_fields(); i++) {
				if (schema->field(i)->name() == req_col) {
					req_cols_idx.push_back(i);
					break;
				}
			}
		}
		// Assume that size of a Record Batch will not exceed the range of int32
		// std::vector<std::shared_ptr<arrow::Int32Array>> partiion_indices;
		std::vector<std::unique_ptr<arrow::Int32Builder>> partition_indices_builder;
		for (int p = 0; p < num_partition; p++) {
			partition_indices_builder.push_back(std::make_unique<arrow::Int32Builder>());
		}
		for (auto batch : named_batches_[table_name]) {
			ARROW_ASSIGN_OR_RAISE(auto req_batch, batch->SelectColumns(req_cols_idx));
			// Use the first col to partition
			assert(req_batch->column(0)->type_id() == arrow::Type::INT32);
			auto part_col = std::static_pointer_cast<arrow::Int32Array>(req_batch->column(0));
			for (int32_t i = 0; i < part_col->length(); i++) {
				int32_t tgt = part_col->Value(i) % num_partition;
				ARROW_RETURN_NOT_OK(partition_indices_builder[tgt]->Append(i));
			}
			// Get row indices for each partitions and select them
			arrow::RecordBatchVector partitions;
			for (int p = 0; p < num_partition; p++) {
				ARROW_ASSIGN_OR_RAISE(auto partiion_indices, partition_indices_builder[p]->Finish());
				ARROW_ASSIGN_OR_RAISE(auto part, arrow::compute::Take(req_batch, partiion_indices));
				partitions.push_back(part.record_batch());
				std::cout << "Got a partition " << p <<" with rows: " << part.record_batch()->num_rows() << std::endl;
			}
			named_batch_partitions_[table_name].push_back(partitions);
		}

		return arrow::Status::OK();
	}

	private:
	std::shared_ptr<arrow::fs::FileSystem> root_;
	std::vector<std::string> table_names_;
	std::string cur_offload_tbl_;
	std::unordered_map<std::string, arrow::compute::Expression> table2sel_;
	std::unordered_map<std::string, std::string> col2table_;
	std::unordered_map<std::string, std::shared_ptr<arrow::Schema>> table_schemas_;
	std::unordered_map<std::string, arrow::RecordBatchVector> named_batches_;
	// Inner vector: paritions, outer vector: from diffrent record batches
	std::unordered_map<std::string, std::vector<arrow::RecordBatchVector>> named_batch_partitions_;

	// The number of initial shuffle workers for a table
	std::unordered_map<std::string, int> num_shfl_workers_;
	std::string ParseFullTable(std::string full_table, std::vector<std::string>& col_names) const {
		std::string table_name;
		std::string remain = full_table;
		bool first = true;
		while (remain.length() > 0) {
			// std::cout << remain << "\n";
			size_t sep_pos = remain.find('|');
			std::string cur = remain.substr(0, sep_pos);
			if (first) {
				table_name = cur;
				first = false;
			}
			else {
				col_names.push_back(cur);
			}
			remain = remain.substr(sep_pos+1);
		}
		return table_name;
	}

	std::vector<int> GetUsedColIndices(
		std::shared_ptr<arrow::Schema> schema, const std::vector<std::string>& used_col_names
	) const {
		std::vector<int> used_col_indices;
		auto all_col_names = schema->field_names();
		for (auto used_col : used_col_names) {
			auto c = std::find(all_col_names.begin(), all_col_names.end(), used_col);
			used_col_indices.push_back(std::distance(all_col_names.begin(), c));
		}
		return used_col_indices;
	}
	arrow::Result<std::shared_ptr<arrow::RecordBatch>> DoGetBatch(std::string tbl_name,
		std::shared_ptr<arrow::RecordBatch> batch, const std::vector<int>& used_col_indices
	) const {
		// Extract needed columns
		ARROW_ASSIGN_OR_RAISE(auto used_col_batch, batch->SelectColumns(used_col_indices));
		if (table2sel_.contains(tbl_name)) {
			// Get select filter
			ARROW_ASSIGN_OR_RAISE(auto filter, GetSelFilter(batch, table2sel_.at(tbl_name)));
			// Applied on needed columns
			ARROW_ASSIGN_OR_RAISE(auto sel_batch, arrow::compute::Filter(used_col_batch, filter));
			std::cout << tbl_name << " Before: " << batch->num_rows() 
				<< "  After: " << sel_batch.record_batch()->num_rows() << std::endl;
			return sel_batch.record_batch();
		}
		return used_col_batch;
	}
	arrow::Result<arrow::Datum> GetSelFilter(
		std::shared_ptr<arrow::RecordBatch> batch, arrow::compute::Expression sel_expr
	) const {
		std::shared_ptr<arrow::Schema> schema = batch->schema();
		ARROW_ASSIGN_OR_RAISE(auto sel_cond, sel_expr.Bind(*schema));
		ARROW_ASSIGN_OR_RAISE(auto exec_batch, arrow::compute::MakeExecBatch(*schema, batch));
		ARROW_ASSIGN_OR_RAISE(auto filter, arrow::compute::ExecuteScalarExpression(sel_cond, exec_batch));
		return filter;
	}

	arrow::Result<std::shared_ptr<arrow::RecordBatch>>
	ConcatRecordBatches(arrow::RecordBatchVector batches) const {
		arrow::ArrayVector concat_cols(batches[0]->num_columns());
		for (int c = 0; c < batches[0]->num_columns(); c++) {
			arrow::ArrayVector cols_to_concat;
			for (const auto& batch : batches) {
				cols_to_concat.push_back(batch->column(c));
			}
			ARROW_ASSIGN_OR_RAISE(concat_cols[c], arrow::Concatenate(cols_to_concat));
		}

		auto merged_batch = arrow::RecordBatch::Make(
			batches[0]->schema(), concat_cols[0]->length(), concat_cols
		);
		return merged_batch;
	}

	std::unordered_map<int, std::shared_ptr<arrow::RecordBatch>> preproc_map_rb_;
};

arrow::Status RunServer(std::string load_schema, int max_num_batches, 
												std::string data_dir, int num_partition)
{
	auto fs = std::make_shared<arrow::fs::LocalFileSystem>();
	auto root = std::make_shared<arrow::fs::SubTreeFileSystem>(data_dir, fs);

	arrow::flight::Location server_location;
	ARROW_ASSIGN_OR_RAISE(server_location, arrow::flight::Location::ForGrpcTcp("0.0.0.0", 36433));
	arrow::flight::FlightServerOptions options(server_location);
	auto server = std::unique_ptr<StorageService>(new StorageService(std::move(root)));
	
	std::ifstream metadata_file(data_dir + "/metadata.json");
	nlohmann::json metadata;
	try {
		metadata_file >> metadata;
	} catch (const std::exception& e) {
		std::cerr << "Error parsing JSON: " << e.what() << std::endl;
	}
	
	std::vector<std::string> tables;
	std::vector<std::vector<std::string>> table_cols;
	for (auto it = metadata["tables"].begin(); it != metadata["tables"].end(); it++) {
		std::string tbl_name = it.key();
		// Filter tables
		if (load_schema == "tpcds") {
			if (tbl_name!="date_dim" && tbl_name!="store_sales" && tbl_name!="item") {
				continue;
			}
		}
		else if (load_schema == "tpch") {
			if (tbl_name == "partsupp" || tbl_name == "part") {
				continue;
			}
		}
		tables.push_back(tbl_name);
		std::vector<std::string> this_tbl_cols;
		// Filter columns
		if (load_schema == "ssb") {
			if (tbl_name == "lineorder") {
				this_tbl_cols = {"lo_revenue", "lo_orderdate", "lo_partkey", "lo_suppkey",
												"lo_custkey", "lo_supplycost"};
				table_cols.push_back(this_tbl_cols);
				continue;
			}
		}
		else if (load_schema == "tpcds") {
			if (tbl_name == "store_sales") {
				this_tbl_cols = {"ss_sold_date_sk", "ss_item_sk", "ss_sales_price", "ss_ext_sales_price"};
				table_cols.push_back(this_tbl_cols);
				continue;
			}
		}
		else if (load_schema == "tpch") {
			if (tbl_name == "lineitem") {
				this_tbl_cols = {"l_orderkey", "l_suppkey", "l_extendedprice", "l_discount",
												"l_shipdate"};
				table_cols.push_back(this_tbl_cols);
				continue;
			}
			else if (tbl_name == "order") {
				this_tbl_cols = {"o_custkey", "o_orderkey", "o_orderdate", "o_shippriority"};
				table_cols.push_back(this_tbl_cols);
				continue;
			}
		}
		for (auto it = metadata["tables"][tbl_name]["columns"].begin(); it != metadata["tables"][tbl_name]["columns"].end(); it++) {
			nlohmann::json c = it.value();
			this_tbl_cols.push_back(c["name"]);
		}
		table_cols.push_back(this_tbl_cols);
	}
	// std::vector<std::string> tables = {"part", "supplier", "date", "lineorder"};
	// std::vector<std::vector<std::string>> table_cols = {
	// 	{"p_partkey", "p_brand1", "p_category"},
	// 	{"s_suppkey", "s_region"},
	// 	{"d_datekey", "d_year"},
	// 	{"lo_partkey", "lo_suppkey", "lo_orderdate", "lo_revenue"}
	// };

	ARROW_RETURN_NOT_OK(server->LoadData(max_num_batches, tables, table_cols, load_schema));

	if (num_partition > 0) {
		auto parti_start = std::chrono::high_resolution_clock::now();
		// ARROW_RETURN_NOT_OK(server->DoPartition(num_partition, "lineorder", 
		// 										{"lo_partkey", "lo_suppkey", "lo_orderdate", "lo_revenue"}));
		ARROW_RETURN_NOT_OK(server->DoPartitionByTake(num_partition, "lineorder", 
												{"lo_partkey", "lo_suppkey", "lo_orderdate", "lo_revenue"}));
		auto parti_end = std::chrono::high_resolution_clock::now();
		auto parti_time = std::chrono::duration_cast<std::chrono::milliseconds>(parti_end - parti_start);
		std::cout << "Partition took " << parti_time.count() << " ms" << std::endl;
	}

	ARROW_RETURN_NOT_OK(server->Init(options));
	std::cout << "Listening on addr " << server->location().ToString() << std::endl;
	ARROW_RETURN_NOT_OK(server->SetShutdownOnSignals({SIGTERM}));
	ARROW_RETURN_NOT_OK(server->Serve());
	
	return arrow::Status::OK();
}



int main(int argc, char** argv)
{
	cxxopts::Options options("mem_server", "Remote memory server");
	options.add_options()
		("load", "To pre-filter some columns", cxxopts::value<std::string>()->default_value("default"))
		("max_numb", "Max number of loading batches", cxxopts::value<int>()->default_value("0"))
		("data_dir", "Data to be used", cxxopts::value<std::string>())
		("num_p", "Number of partitions, do not set to disable partitioning in storage server",
				cxxopts::value<int>()->default_value("0"))
		("help", "Print usage")
	;
	auto result = options.parse(argc, argv);
	if (result.count("help")) {
		std::cout << options.help() << std::endl;
		exit(0);
	}
	int max_num_batches = result["max_numb"].as<int>();
	std::string load_schema = result["load"].as<std::string>();
	std::string data_dir = result["data_dir"].as<std::string>();
	if (!std::filesystem::is_directory(data_dir)) {
		std::cerr << "Data directory " << data_dir << " does not exist" << std::endl;
		return EXIT_FAILURE;
	}
	std::cout << data_dir << std::endl;

	int num_p = result["num_p"].as<int>();

	// Modify output formats
	std::streambuf* original_cout_buf = std::cout.rdbuf();
	std::string prefix = "[Storage] ";
	PrefixBuf prefixBuf(original_cout_buf, prefix);
	std::cout.rdbuf(&prefixBuf);

	auto status = RunServer(load_schema, max_num_batches, data_dir, num_p);
	if (!status.ok()) {
	std::cerr << status.ToString() << std::endl;
		return EXIT_FAILURE;
	}
	
	std::cout.rdbuf(original_cout_buf);
	return EXIT_SUCCESS;	
}