#include <unordered_map>
#include <iostream>
#include <fstream>

#include <base_worker.h>
#include <knl_executor.h>
#include <shfl_server.h>

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cxxopts.hpp>

#define SHFL_SERVER_START_PORT 11451

#define GPU_SHFL_PROFILE

class GPU_Worker : public Base_Worker
{
	using RawBatch = std::vector<uint8_t*>;
public:
	GPU_Worker(int id);
	~GPU_Worker();
	arrow::Status GetFlightTables(std::vector<std::string> toget_table_names) override;
	arrow::Status RequestCols(std::string tbl_name, std::vector<std::string> col_names);
	
	// Shuffle via Arrow Flight
	// arrow::Status FlightShuffle();
	arrow::Status FlightGetFromShfl(int num_in_workers, std::vector<std::string> toget_table_names);
	
private:
	std::unique_ptr<knl_executor> executor_;
	nlohmann::json knl_info_;

	std::unordered_map<std::string, arrow::RecordBatchVector> named_batches_;
	std::unordered_map<std::string, arrow::ArrayVector> preprocessd_cols_batches_;
	
	arrow::Status SelCol(std::string tbl_name, std::string col_name);

	std::unique_ptr<GPU_ShuffleServer> shfl_server_;

	arrow::Status ShflRecv(std::string tbl_name, std::shared_ptr<arrow::Schema> schema,
												 int src_start_id, int num_src_worker) override;
	arrow::Status ShflSend(int in_batch, int in_partition,
												 std::shared_ptr<arrow::Schema> schema, int dest_start_id) override;

	// --------------------- For asynchronous execution --------------------- 
	void ThreadSync() override;
	void UpdateShflBatch(std::string tbl_name, std::shared_ptr<arrow::Schema> schema) override;
	void UpdateShflBatchPartition(std::string tbl_name, int b, int p);
	void InitShflRecv(std::string tbl_name, std::shared_ptr<arrow::Schema> schema) override;
	void InitExecution() override;
	void UpdateExecBatch() override;
	void ExecutePartition(int b, int p) override;
	row_size_t GetResNumRows(int in_batch, int in_partition) override;
	uint64_t GetWorkerUsedMem() override;
	void FreeIntermediates(int in_batch, int in_partition) override;

	arrow::Result<std::shared_ptr<arrow::Table>> AggregateFinalResults(int num_batches) override {
		assert(false && "should not be called by GPU worker");
	}
	arrow::Status CommitResult(std::shared_ptr<arrow::Table> res) override {
		assert(false && "should not be called by GPU worker");
	}

	std::string probe_knl_;
	std::vector<std::string> build_knl_, build_key_, build_filter_;
	std::vector<std::vector<std::string>> build_payload_;
	std::vector<std::string> probe_key_, probe_payload_;
	std::unordered_map<std::string, std::vector<std::string>> computed_cols_;
	std::vector<std::string> aggr_col_, build_gb_key_, probe_gb_key_, all_gb_key_, aggr_res_col_;
	std::vector<std::string> all_payloads_, res_cols_;
};

GPU_Worker::GPU_Worker(int id)
	: Base_Worker(id)
{
	worker_type_ = "GPU";
	// uint64_t executor_init_buf_size = 0;
	// if (std::getenv("GPU_IBUF_SIZE")) {
	// 	uint64_t size_in_mb = std::stoi(std::getenv("GPU_IBUF_SIZE"));
	// 	executor_init_buf_size = size_in_mb * 1024 * 1024;
	// }
	executor_ = std::make_unique<knl_executor>(id, num_inp_partitions_);
}

GPU_Worker::~GPU_Worker()
{}

arrow::Status GPU_Worker::GetFlightTables(std::vector<std::string> toget_table_names)
{
	std::unique_ptr<arrow::flight::FlightStreamReader> stream;

	for (int i = 0; i < toget_table_names.size(); i++) {
		std::string table_name = toget_table_names[i];
		ARROW_ASSIGN_OR_RAISE(auto schema, GetSchemaFromPlan(table_name));
		std::string request = MakeFlightRequest(table_name, schema, true);
		ARROW_ASSIGN_OR_RAISE(stream, client_->DoGet(arrow::flight::Ticket{request}));
		ARROW_ASSIGN_OR_RAISE(auto batches, stream->ToRecordBatches());
		
		// [Deprecated] If get a partition from Flight server
		// size_t sep_pos = table_name.find(':');
		// if (sep_pos != std::string::npos) {
		// 	table_name = table_name.substr(0, sep_pos);
		// }
		named_batches_[table_name] = batches;

		auto column_names = batches[0]->schema()->field_names();
		for (auto col_name : column_names) {
			executor_->InitFlightColBatch(col_name, batches.size(), false);
		}
		for (int b = 0; b < batches.size(); b++) {
			for (auto col_name : column_names) {
				executor_->UpdateFlightColBatchNumRows(col_name, b, batches[b]->num_rows());
				// cols_batches_size_[col_name].push_back(batches[b]->num_rows());
			}
			#ifdef DBG_PRINT
			std::cout << batches[b]->schema()->ToString() << std::endl;
			std::cout << table_name << " batch#" << b 
								<< " has " << batches[b]->num_rows() << " rows" << std::endl;
			#endif
		}
		ARROW_RETURN_NOT_OK(RequestCols(table_name, column_names));
	}
	
	ARROW_RETURN_NOT_OK(CommitPreproc());
	return arrow::Status::OK();
}

arrow::Status GPU_Worker::SelCol(std::string tbl_name, std::string col_name)
{
	// Table check
	if (named_batches_.find(tbl_name) == named_batches_.end()) {
		return arrow::Status::Invalid("Table ", tbl_name, " not existed");
	}
	auto batches = named_batches_[tbl_name];
	// Column check
	bool has_col = false;
	for (int c = 0; c < batches[0]->num_columns(); c++) {
		if (batches[0]->column_name(c) == col_name) {
			has_col = true;
			break;
		}
	}
	if (!has_col) {
		return arrow::Status::Invalid("Column ", col_name, " not existed");
	}

	// Collect raw pointer of record batchs
	for (int b = 0; b < batches.size(); b++) {
		auto col = named_batches_[tbl_name][b]->GetColumnByName(col_name);
		auto col_data = col->data();
		auto t = col->type_id();
		// Preprocess string and decimal arrays
		if (t == arrow::Type::STRING) {
			std::shared_ptr<arrow::Array> toint;
			ARROW_RETURN_NOT_OK(
				StringArray2Int(std::static_pointer_cast<arrow::StringArray>(col), &toint)
			);
			preprocessd_cols_batches_[col_name].push_back(toint);
			col_data = toint->data();
			#ifdef DBG_PRINT
			std::cout << col_name << " batch " << b << " length " 
								<< preprocessd_cols_batches_[col_name][b]->length() << std::endl;
			std::cout << preprocessd_cols_batches_[col_name][b]->ToString() << std::endl;
			std::cout << col_name << " fuck string " << col_data->buffers.size() << std::endl;
			std::cout << "map with " << map.size() << std::endl;
			#endif
		}
		else if (t == arrow::Type::DECIMAL128) {
			std::shared_ptr<arrow::Array> toint;
			ARROW_RETURN_NOT_OK(
				DecimalArray2Int(std::static_pointer_cast<arrow::Decimal128Array>(col), &toint)
			);
			preprocessd_cols_batches_[col_name].push_back(toint);
			col_data = toint->data();
			#ifdef DBG_PRINT
			std::cout << col_name << " fuck decimal128" << col_data->buffers.size() << std::endl;
			#endif
		}
		else if (t == arrow::Type::FIXED_SIZE_BINARY) {
			std::shared_ptr<arrow::Array> toint;
			ARROW_RETURN_NOT_OK(
				FixedBinArray2Int(std::static_pointer_cast<arrow::FixedSizeBinaryArray>(col), 
													&toint)
			);
			preprocessd_cols_batches_[col_name].push_back(toint);
			col_data = toint->data();
			#ifdef DBG_PRINT
			std::cout << col_name << " batch " << b << " length " 
								<< preprocessd_cols_batches_[col_name][b]->length() << std::endl;
			std::cout << preprocessd_cols_batches_[col_name][b]->ToString() << std::endl;
			std::cout << col_name << " fuck fixed binary" << std::endl;
			std::cout << "map with " << map.size() << std::endl;
			#endif
		}
		else {
			assert(t == arrow::Type::INT32);		// make sure only 1 validility buffer and 1 data buffer
			// assert(col_data->buffers.size() == 2);		// make sure only 1 validility buffer and 1 data buffer
		}
		const int32_t* col_ptr = col_data->GetValues<int32_t>(1);
		// cols_batches_ptr_[col_name].push_back(col_ptr);
		executor_->UpdateFlightColBatch(col_name, b, col_ptr);

		#ifdef DBG_PRINT
		std::cout << col_name << std::endl;
		for (int i = 0; i < 10; i++) {
			std::cout << col_ptr[i] << std::endl;
		}
		#endif
	}

	return arrow::Status::OK();
}

arrow::Status GPU_Worker::RequestCols(std::string tbl_name, std::vector<std::string> req_col_names)
{
	for (auto col_name : req_col_names) {
		ARROW_RETURN_NOT_OK(SelCol(tbl_name, col_name));
	}
	return arrow::Status::OK();
}


// #define AUTO_LAUNCH_WORKER

arrow::Status GPU_Worker::FlightGetFromShfl(int num_in_workers, std::vector<std::string> toget_partition_names)
{
	for (int w = 0; w < num_in_workers; w++) {
		arrow::flight::Location shfl_server_location;
		ARROW_ASSIGN_OR_RAISE(shfl_server_location, 
													arrow::flight::Location::ForGrpcTcp("localhost", SHFL_SERVER_START_PORT+w));
		ARROW_ASSIGN_OR_RAISE(auto shfl_client, arrow::flight::FlightClient::Connect(shfl_server_location));
		
		for (auto partition_name : toget_partition_names) {
			// Must try to get a partition
			size_t sep_pos = partition_name.find(':');
			if (sep_pos == std::string::npos) {
				return arrow::Status::Invalid(partition_name, " is not a partition");
			}
			auto table_name = partition_name.substr(0, sep_pos);
			
			#ifdef AUTO_LAUNCH_WORKER
			auto get_stream = shfl_client->DoGet(arrow::flight::Ticket{partition_name});
			while (!get_stream.ok()) {
				get_stream = shfl_client->DoGet(arrow::flight::Ticket{partition_name});
			}
			auto stream = std::move(get_stream.ValueOrDie());
			#else
			ARROW_ASSIGN_OR_RAISE(auto stream, shfl_client->DoGet(arrow::flight::Ticket{partition_name}));
			#endif

			ARROW_ASSIGN_OR_RAISE(auto batches, stream->ToRecordBatches());
			// Check 
			if (named_batches_.find(table_name) == named_batches_.end()) {
				named_batches_[table_name] = batches;
			}
			else {
				named_batches_[table_name].insert(named_batches_[table_name].end(),
																					batches.begin(), batches.end());
			}
		}
	}
	
	return arrow::Status::OK();
}

void GPU_Worker::InitShflRecv(std::string tbl_name, std::shared_ptr<arrow::Schema> schema)
{
	int num_inp_batches = (tbl_name == probe_tbl_name_)? num_inp_probe_batches_ : 1;
	std::unique_lock named_partition_lock_(named_partition_mutex_);
	named_partition_ready_[tbl_name].resize(num_inp_batches);
	for (int b = 0; b < num_inp_batches; b++) {
		named_partition_ready_[tbl_name][b].resize(num_inp_partitions_, 0);
	}
	executor_->InitShflInput(schema->field_names(), num_inp_batches);
}

void GPU_Worker::ThreadSync()
{
	executor_->SyncContext();
}

void GPU_Worker::InitExecution()
{
	executor_->LoadModule();
 
	knl_info_ = exec_plan_["stage_kernel_info"][stage_no_];
	probe_knl_ = knl_info_["probe_knl"];
	build_knl_ = knl_info_["build_knl"];
	build_key_ = knl_info_["build_key"];
	build_filter_ = knl_info_["build_filter"];
	build_payload_ = knl_info_["build_payload"];
	probe_key_ = knl_info_["probe_key"];
	probe_payload_ = knl_info_["probe_payload"];
	if (knl_info_.contains("computed_cols")) {
		computed_cols_ = knl_info_["computed_cols"];
	}
	if (knl_info_.find("aggr_col") != knl_info_.end()) {
		aggr_col_ = knl_info_["aggr_col"];
		build_gb_key_ = knl_info_["groupby_build"];
		probe_gb_key_ = knl_info_["groupby_probe"];
		aggr_res_col_ = knl_info_["aggr_res_col"];
		all_gb_key_ = knl_info_["groupby"];
	}
	res_cols_ = table_schema_[knl_info_["result_name"]]["name"];

	all_payloads_ = probe_payload_;
	for (auto build_payload_t : build_payload_) {
		for (auto build_pld : build_payload_t) {
			all_payloads_.push_back(build_pld);
		}
	}

	std::unique_lock res_lock(res_mutex_);
	partition_res_status_.resize(num_inp_probe_batches_);
	for (int b = 0; b < num_inp_probe_batches_; b++) {
		partition_res_status_[b].resize(num_inp_partitions_, 0);
	}	
	executor_->InitKnlExecutionParam(probe_knl_, 
																	build_knl_, build_key_, build_filter_, build_payload_,
																	probe_key_, probe_payload_,
																	partition_col_, num_out_partitions_,
																	computed_cols_, aggr_col_, build_gb_key_, probe_gb_key_,
																	all_gb_key_, aggr_res_col_, res_cols_);
	executor_->InitKnlExecution(all_payloads_, num_inp_probe_batches_);
}

void GPU_Worker::ExecutePartition(int b, int p)
{
	std::string build_tbl_name = (input_shlf_tbls_[0] == probe_tbl_name_)? input_shlf_tbls_[1] : input_shlf_tbls_[0];
	
	// std::cout << "bp" << b << " " << p << std::endl;
	executor_->UpdateExecution(b, p);
	uint32_t exec_size =  executor_->Execute();
#ifdef BW_PROFILE
	exec_rows_.push_back(exec_size);
#endif

	partition_res_status_[b][p] = 1;
}

void GPU_Worker::UpdateShflBatch(std::string tbl_name, std::shared_ptr<arrow::Schema> schema)
{
	std::unique_lock named_partition_lock_(named_partition_mutex_);
	named_partition_ready_[tbl_name].emplace_back(std::vector<uint8_t>(num_inp_partitions_, 0));

	executor_->UpdateComingBatchInput(schema->field_names());
}

void GPU_Worker::UpdateExecBatch()
{
	std::unique_lock lock(res_mutex_);
	executor_->UpdateComingBatchOutput(all_payloads_);
	partition_res_status_.emplace_back(std::vector<uint8_t>(num_inp_partitions_, 0));
}

void GPU_Worker::UpdateShflBatchPartition(std::string tbl_name, int b, int p)
{
	std::unique_lock named_partition_lock_(named_partition_mutex_);
	named_partition_ready_[tbl_name][b][p] = 1;
}

// #define SHFL_DBG_PRINT
arrow::Status GPU_Worker::ShflRecv(std::string tbl_name, std::shared_ptr<arrow::Schema> schema,
																	 int src_start_id, int num_src_worker)
{
	#ifdef SHFL_DBG_PRINT
	std::cout << "Shfl recv " << schema->ToString() << " from "
						<< src_start_id << "+" << num_src_worker << std::endl;
	#endif
	// Note that when receiving from initial shuffle workers
	// the schema in the exec_plan should be same with the one in storage server
	int num_cols = schema->num_fields();
	auto column_names = schema->field_names();

	int num_all_batches = num_src_worker;
	
	// Initialize buffer and number of rows
	std::vector<row_size_t> num_rows_srcw(num_src_worker, 0);	

	int curr_recv_batch = curr_recv_batches_[tbl_name]-1;
	int req_size = num_src_worker * num_cols;
	for (int p = 0; p < num_inp_partitions_; p++) {
		executor_->InitColPartitionBatch(column_names, curr_recv_batch, p, num_src_worker);
		int req_idx = 0;
		MPI_Request recv_shfl_reqs[req_size];
		for (int w = 0; w < num_src_worker; w++) {
			int src_worker_id = src_start_id + w;
#ifdef GPU_SHFL_PROFILE
			timer_start("ShflRecv-Sync size");
#endif
			MPI_CHECK(MPI_Recv(&num_rows_srcw[w], 1, MPI_ROW_SIZE_T, src_worker_id,
													0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
#ifdef GPU_SHFL_PROFILE
			timer_end("ShflRecv-Sync size");
#endif
			total_proc_size_ += num_rows_srcw[w];
			executor_->UpdateColPartitionBatchNumRows(column_names, curr_recv_batch, p, w, num_rows_srcw[w]);
			#ifdef SHFL_DBG_PRINT
			std::cout << tbl_name << p << " " << w << std::endl;
			#endif
			for (int c = 0; c < num_cols; c++) {
				std::string col_name = column_names[c];
				buf_size_t recv_size = 0;
				int mpi_tag = MPI_TAGC + c + (id_in_stage_+p*num_workers_same_stage_)*num_cols + src_worker_id*MPI_TAGC;
				MPI_CHECK(MPI_Recv(&recv_size, 1, MPI_BUF_SIZE_T, src_worker_id,
														0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
				executor_->UpdateColPartitionBatchSize(col_name, curr_recv_batch, p, w, recv_size);
#ifdef GPU_SHFL_PROFILE
				timer_start("ShflRecv-Wait input buf");
#endif
				uint8_t* col_batch = executor_->AllocateColPartitionBuf(col_name, curr_recv_batch, p, w, recv_size);
#ifdef GPU_SHFL_PROFILE
				timer_end("ShflRecv-Wait input buf");
#endif
				// MPI_CHECK(MPI_Recv(col_batch, recv_size, MPI_UINT8_T, src_worker_id,
				// 										mpi_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
				MPI_CHECK(MPI_Irecv(col_batch, recv_size, MPI_UINT8_T, src_worker_id,
														mpi_tag, MPI_COMM_WORLD, &recv_shfl_reqs[req_idx]));
				req_idx += 1;
			}
			
			// for (int b = 0; b < num_batches[w]; b++) {
			// 	// Receive column by column
			// 	for (int c = 0; c < num_cols; c++) {
			// 		std::string col_name = column_names[c];
			// 		buf_size_t recv_size = 0;
			// 		int mpi_tag = 100000 + c + id_in_stage_*num_cols 
			// 									+ b*num_workers_same_stage_*num_cols + src_worker_id*100000;
			// 		MPI_CHECK(MPI_Recv(&recv_size, 1, MPI_BUF_SIZE_T, src_worker_id,
			// 												mpi_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
			// 		executor_->UpdateColBatchSize(col_name, batch_idx, recv_size);
			// 		#ifdef SHFL_DBG_PRINT	
			// 		std::cout << w << " recv mpi tag: " << mpi_tag << ", buf size: " 
			// 							<< recv_size << std::endl;
			// 		#endif
					
			// 		uint8_t* col_batch = executor_->AllocateColBuf(col_name, batch_idx, recv_size);
			// 		MPI_CHECK(MPI_Irecv(col_batch, recv_size, MPI_UINT8_T, src_worker_id,
			// 												mpi_tag, MPI_COMM_WORLD, &recv_shfl_reqs[req_idx]));
			// 		req_idx++;
			// 	}
			// 	batch_idx++;
			// }
		}
		assert(req_idx == req_size);
#ifdef GPU_SHFL_PROFILE
		timer_start("ShflRecv-WaitAll");
#endif
		MPI_CHECK(MPI_Waitall(req_size, recv_shfl_reqs, MPI_STATUS_IGNORE));
#ifdef GPU_SHFL_PROFILE
		timer_end("ShflRecv-WaitAll");
#endif
		const int batch_pos = curr_recv_batches_[tbl_name] - 1;
		UpdateShflBatchPartition(tbl_name, batch_pos, p);
	}

	#ifdef SHFL_DBG_PRINT
	// for (int b = 0; b < num_all_batches; b++) {
	// 	row_size_t num_rows = executor_->GetColSize("lo_revenue", b);
	// 	uint8_t* buf = executor_->GetColBuf("lo_revenue", b);
	// 	int* last_col_as_int_dev = reinterpret_cast<int*>(buf);
	// 	int test_size = 10;
	// 	int* last_col_as_int;
	// 	std::cout << "last 10 elems as int32 of last col of batch" << b << ": ";
	// 	for (int i = 0; i < test_size; i++) {
	// 		std::cout << last_col_as_int[num_rows-1-i] << " ";
	// 	}
	// 	std::cout << std::endl;
	// }
	#endif

	return arrow::Status::OK();
}

// #define SHFL_SEND_DBG
arrow::Status GPU_Worker::ShflSend(int in_batch, int in_partition,
																	 std::shared_ptr<arrow::Schema> schema, int dest_start_id)
{
	int num_cols = schema->num_fields();
	auto column_names = schema->field_names();
	for (auto res_col : column_names) {
		// std::cout << res_col << std::endl;
		assert(executor_->CheckResCol(res_col) && "res not in");
	}

	int num_res_batches = executor_->GetNumResBatch();
	assert(num_res_batches == 1);

	int req_size = num_out_partitions_ * num_cols;
	MPI_Request send_shfl_reqs[req_size];

	int req_idx = 0;
	row_size_t total_res_rows = 0;
	for (int b = 0; b < num_res_batches; b++) {
		for (int p = 0; p < num_out_partitions_; p++) {
			int tgt_worker_id = dest_start_id + (p % num_shfl_tgt_workers_); 
			row_size_t res_partition_size = executor_->GetPartitionResSize(in_batch, in_partition, p);
			total_res_rows += res_partition_size;
			MPI_CHECK(MPI_Send(&res_partition_size, 1, MPI_ROW_SIZE_T, tgt_worker_id, 
													0, MPI_COMM_WORLD));
			buf_size_t buf_size = res_partition_size * sizeof(int32_t);		// Assume results of gpu are all INT
			for (int c = 0; c < num_cols; c++) {
				int mpi_tag = MPI_TAGC + c + p*num_cols + b*num_out_partitions_*num_cols + id_*MPI_TAGC;
				MPI_CHECK(MPI_Send(&buf_size, 1, MPI_BUF_SIZE_T, tgt_worker_id,
														0, MPI_COMM_WORLD));
				std::string res_col = column_names[c];
				uint8_t* buf_ptr = (uint8_t*) executor_->GetResColPartitionPtr(res_col, in_batch, in_partition, p);
				// MPI_CHECK(MPI_Send(buf_ptr, buf_size, MPI_UINT8_T, tgt_worker_id,
				// 										mpi_tag, MPI_COMM_WORLD));
				MPI_CHECK(MPI_Isend(buf_ptr, buf_size, MPI_UINT8_T, tgt_worker_id,
														mpi_tag, MPI_COMM_WORLD, &send_shfl_reqs[req_idx++]));
			}
		}
		assert(req_idx == req_size);
		MPI_CHECK(MPI_Waitall(req_size, send_shfl_reqs, MPI_STATUS_IGNORE));
	}
#ifdef BW_PROFILE
	shfl_rows_.push_back(total_res_rows);
	shfl_bytes_.push_back(total_res_rows * num_cols * sizeof(int32_t));
#endif

	return arrow::Status::OK();
}

row_size_t GPU_Worker::GetResNumRows(int in_batch, int in_partition)
{
	row_size_t res_num_rows = 0;
	for (int p = 0; p < num_out_partitions_; p++) {
		res_num_rows += executor_->GetPartitionResSize(in_batch, in_partition, p);
	}
	return res_num_rows;
}

uint64_t GPU_Worker::GetWorkerUsedMem()
{
	return executor_->GetUsedMemSize();
}

void GPU_Worker::FreeIntermediates(int in_batch, int in_partition)
{
	executor_->FreeRes(in_batch, in_partition);
}

arrow::Status RunClient(int id)
{
	GPU_Worker worker(id);
	arrow::flight::Location location;
	std::string server_ip(std::getenv("SR_IP"));
	ARROW_ASSIGN_OR_RAISE(location, arrow::flight::Location::ForGrpcTcp(server_ip, 36433));
	ARROW_RETURN_NOT_OK(worker.ConnectServer(location));

	ARROW_RETURN_NOT_OK(worker.GetData());

	worker.StartExecute();

	ARROW_RETURN_NOT_OK(worker.Finish());

	worker.PrintStats();

	return arrow::Status::OK();
}


int main(int argc, char** argv)
{
	// Parse command line args
	cxxopts::Options options("gpu_worker", "GPU worker");
	options.add_options()
		("ID", "ID to process parition", cxxopts::value<std::string>()->default_value("0"))
		("to_shfl", "Results will be shuffled to following workers", 
				cxxopts::value<bool>()->default_value("false"))
		("from_shfl", "Input data are from shuffle of workers before", 
				cxxopts::value<bool>()->default_value("false"))
		("n_out_p", "Number of result partitions (when sending to shuffle)",
				cxxopts::value<int>()->default_value("0"))
		("n_in_w", "Number of input workers (when receiving from shuffle)",
				cxxopts::value<int>()->default_value("0"))
		("h,help", "Print usage")
	;
	auto result = options.parse(argc, argv);
	if (result.count("help")) {
		std::cout << options.help() << std::endl;
		exit(0);
	}

	int mpi_id, num_procs, provided;
	// MPI_CHECK(MPI_Init(&argc, &argv));
	MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));
	MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
	
	// Modify output formats
	std::streambuf* original_cout_buf = std::cout.rdbuf();
	std::string prefix = "[GPU Worker " + std::to_string(mpi_id) + "] ";
	PrefixBuf prefixBuf(original_cout_buf, prefix);
	std::cout.rdbuf(&prefixBuf);

	// auto status = TestFlight(mpi_id, to_shfl, from_shfl, num_out_partitions, num_in_workers);
	auto status = RunClient(mpi_id);
	if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
		std::cout.rdbuf(original_cout_buf);
		MPI_CHECK(MPI_Finalize());
    return EXIT_FAILURE;
  }

	std::cout.rdbuf(original_cout_buf);
	MPI_CHECK(MPI_Finalize());
  return EXIT_SUCCESS;	
}