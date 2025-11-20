#ifndef DISAGG_BASE_WORKER_HEADER
#define DISAGG_BASE_WORKER_HEADER

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/flight/api.h>
#include <arrow/filesystem/api.h>
#include <arrow/ipc/api.h>
#include <arrow/compute/api.h>

#include <util.h>

#include <thread>
#include <future>
#include <shared_mutex>
#include <csignal>

#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <unordered_map>

#define MPI_TAGC 10000

// #define EXEC_PLAN_DBG
// #define WORKER_PROFILE
// #define BW_PROFILE

#define DHAP_PRINT_ON(target_hostname, var)                           \
    do {                                                            \
        char hostname[HOST_NAME_MAX + 1];                          \
        if (gethostname(hostname, sizeof(hostname)) == 0) {       \
            std::string current_hostname(hostname);               \
            if (current_hostname == (target_hostname)) {          \
                std::cout << #var << ": " << (var) << std::endl;  \
            }                                                       \
        } else {                                                    \
            perror("gethostname failed");                           \
        }                                                           \
    } while (0)

class Base_Worker
{
public:
	Base_Worker(int id);
	~Base_Worker();
	arrow::Status ConnectServer(arrow::flight::Location location);
	arrow::Status GetData();
	arrow::Status GetShflTables(std::vector<std::string> toget_table_names);
	virtual arrow::Status ShflRecv(std::string tbl_name, std::shared_ptr<arrow::Schema> schema,
												 				 int src_start_id, int num_src_worker) = 0;
	void StartExecute();
	arrow::Status Finish();

	std::map<std::string, std::chrono::milliseconds> time_stats_;
	void PrintStats();
	void DumpStats();

protected:
	int id_;		// ID of MPI
	// Info about execution plan
	void GetExecInfo();
	arrow::Result<std::shared_ptr<arrow::Schema>> GetSchemaFromPlan(std::string tbl);
	std::string plan_dir_, subq_id_str_;
	nlohmann::json exec_plan_;
	nlohmann::json table_schema_;
	std::vector<int> num_stage_workers_;
	int num_stages_, stage_no_, id_in_stage_, num_workers_same_stage_;
	int num_inp_probe_batches_ = 1;
	int num_inp_partitions_ = 0;
	int num_out_partitions_ = 0;
	int num_shfl_tgt_workers_ = 0;
	uint64_t all_res_num_rows = 0;
	std::string probe_tbl_name_;
	std::string res_tbl_name_;
	std::string partition_col_ = "";
	std::string worker_type_;
	// For shuffle workers
	std::string shfl_table_name_;
	std::vector<std::string> shfl_col_names_;
	int num_same_shfl_workers_;				// # workers that shuffle the same table
	int local_shfl_id_;								// Local ID for the shfl table
	int shfl_tgt_stage_;
	int shfl_tgt_start_id_;

	// For Flight communication
	std::unique_ptr<arrow::flight::FlightClient> client_;
	std::vector<std::unique_ptr<arrow::flight::FlightClient>> shfl_clients_;

	// For data preprocessing
	arrow::Status StringArray2Int(std::shared_ptr<arrow::StringArray> string_array, 
																std::shared_ptr<arrow::Array>* int_array);
	
	arrow::Status DecimalArray2Int(std::shared_ptr<arrow::DecimalArray> decimal_array, 
																 std::shared_ptr<arrow::Array>* int_array);
	
	arrow::Status FixedBinArray2Int(std::shared_ptr<arrow::FixedSizeBinaryArray> bin_array, 
																	std::shared_ptr<arrow::Array>* int_array);

	// --------------------- For asynchronous execution --------------------- 
	std::vector<std::string> input_shlf_tbls_;
	// To hold shuffle data
	// named_partition_ready_[tbl_name][batch][partition_id]
	std::shared_mutex named_partition_mutex_;
	std::unordered_map<std::string, std::vector<std::vector<uint8_t>>> named_partition_ready_;
	std::shared_mutex curr_recv_mutex_;
	std::unordered_map<std::string, int> curr_recv_batches_;
	std::future<int> probe_future_recv_batches_;
	int build_recv_finished_ = 0;
	// To hold results	
	int curr_exec_batches_ = 0;
	std::future<int> future_exec_batches_;
	std::shared_mutex res_mutex_;
	// 0: not processed;  1: processed, not sent;  2: sent
	std::vector<std::vector<uint8_t>> partition_res_status_;

	std::string MakeFlightRequest(std::string tbl_name, std::shared_ptr<arrow::Schema> schema, 
																bool full_tbl, int batch_idx);
	virtual arrow::Status GetFlightTables(std::vector<std::string> toget_table_names) = 0;
	arrow::Status GetShflPartitions(std::vector<std::string> toget_table_names);
	virtual void ThreadSync() {};
	virtual void InitShflRecv(std::string tbl_name, std::shared_ptr<arrow::Schema> schema) = 0;
	int ShflRecvLoop(std::string tbl_name, std::shared_ptr<arrow::Schema> schema, 
									 int src_start_id, int num_src_worker);
	virtual void UpdateShflBatch(std::string tbl_name, std::shared_ptr<arrow::Schema> schema) = 0;
	int Execute();
	virtual void InitExecution() = 0;
	virtual void UpdateExecBatch() = 0;
	virtual void ExecutePartition(int b, int p) = 0;
	virtual arrow::Status ShflSend(int in_batch, int in_partition,
												 				 std::shared_ptr<arrow::Schema> schema, int dest_start_id) = 0;
	virtual row_size_t GetResNumRows(int in_batch, int in_partition) = 0;

	virtual uint64_t GetWorkerUsedMem() = 0;		// memory used size in bytes
	virtual void FreeIntermediates(int in_batch, int in_partition) = 0;

	virtual arrow::Result<std::shared_ptr<arrow::Table>> AggregateFinalResults(int num_batches) = 0;
	virtual arrow::Status CommitResult(std::shared_ptr<arrow::Table> res) = 0;

	// For profile
	#ifdef WORKER_PROFILE
	std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> start_times_;
	std::unordered_map<std::string, long long> total_elapsed_times_;
	std::vector<std::string> event_order_;
	std::shared_mutex timer_mutex_;
	long long get_elapsed_time(const std::string& event) {
		if (total_elapsed_times_.find(event) != total_elapsed_times_.end()) {
			return total_elapsed_times_[event];
		} else {
			return 0;
		}
	}
	#endif
	void timer_start(const std::string& event) {
		#ifdef WORKER_PROFILE
		std::unique_lock timer_lock(timer_mutex_);
		auto now = std::chrono::high_resolution_clock::now();
		if (total_elapsed_times_.find(event) == total_elapsed_times_.end()) {
			// Only record the order if it's the first time the event is started
			event_order_.push_back(event);
		}
		start_times_[event] = now; // Record the start time
		#endif
	}
	void timer_end(const std::string& event) {
		#ifdef WORKER_PROFILE
		std::unique_lock timer_lock(timer_mutex_);
		auto now = std::chrono::high_resolution_clock::now();
		if (start_times_.find(event) != start_times_.end()) {
			// Calculate elapsed time since start
			auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_times_[event]).count();
			total_elapsed_times_[event] += elapsed;
			start_times_.erase(event);
		} else {
			std::cerr << "Error: Timer for event '" << event << "' was not started." << std::endl;
			exit(1);
		}
		#endif
	}
	#ifdef BW_PROFILE
	std::vector<uint32_t> exec_rows_;
	std::vector<uint32_t> shfl_rows_;
	std::vector<uint64_t> shfl_bytes_;
	#endif
	void ProfileDump() {
		#ifdef WORKER_PROFILE
		std::string out = "============== Worker Profile =================\n";
		for (const auto& event : event_order_) {
			out += event + ": " + std::to_string(total_elapsed_times_[event]) + " ms\n";
		}
		#ifdef BW_PROFILE
		if (stage_no_ < num_stages_-1) {
			uint64_t total_shfl_rows = std::accumulate(shfl_rows_.begin(), shfl_rows_.end(), uint64_t(0));
			uint64_t total_shfl_bytes = std::accumulate(shfl_bytes_.begin(), shfl_bytes_.end(), uint64_t(0));
			long long shfl_send_time = get_elapsed_time("ShflSend");
			long long mpi_send_time = get_elapsed_time("ShflSend") 
				- get_elapsed_time("ShflSend-Partition Indexing") - get_elapsed_time("ShflSend-Partition Take");
			std::string shfl_send_bw = shfl_send_time > 0 ?
				std::to_string(total_shfl_rows/shfl_send_time) : "inf";
			std::string mpi_send_bw = mpi_send_time > 0 ? 
				std::to_string(total_shfl_bytes * 1000.0 / (1024*1024) / mpi_send_time) : "inf";
			out += "ShflRows: " + std::to_string(total_shfl_rows) + "\n";
			out += "ShflBytes: " + std::to_string(total_shfl_bytes) + "\n";
			out += "MPISend time: " + std::to_string(mpi_send_time) + " ms\n";

			out += "ShflSend BW: " + shfl_send_bw + " tuples/ms\n";
			out += "MPISend BW: " + mpi_send_bw + " MB/s\n";

			if (stage_no_ > 0) {
				uint64_t total_exec_rows = std::accumulate(exec_rows_.begin(), exec_rows_.end(), uint64_t(0));
				long long total_exec_time = get_elapsed_time("Execution");
				std::string exec_bw = total_exec_time > 0 ? 
					std::to_string(total_exec_rows/total_exec_time) : "inf";
				out += "Exec BW: " + exec_bw + " tuples/ms\n";
			}
		}
		#endif
		out += "===============================================";
		if (stage_no_ == 0) {
			// For shfl workers (which locate on master node), print to a seperated file
			std::string filename = std::string(std::getenv("DHAP_LOG_DIR")) + "/stage0.log";
			std::ofstream file(filename, std::ios_base::app);
			if (file.is_open()) {
				file << out << std::endl;
			} else {
				std::cerr << "Unable to open file " << filename << std::endl;
			}
		}
		else {
			std::cout << out << std::endl;
		}
		#endif
	}

	uint64_t total_proc_size_ = 0;

	std::hash<std::string> str_hash_ = std::hash<std::string>{};
	std::unordered_map<std::string, int> preproc_str2int_;
	int PreprocStr2Int(std::string s);
	arrow::Status CommitPreproc();
};

Base_Worker::Base_Worker(int id) 
	: id_(id)
{
	plan_dir_ = std::getenv("PLAN_DIR");
	subq_id_str_ = std::getenv("SUB_QUERY");
	std::ifstream plan_file(plan_dir_ + "/" + "plan" + subq_id_str_ + ".json");
	try {
		plan_file >> exec_plan_;
	} catch (const std::exception& e) {
		std::cerr << "Error parsing JSON: " << e.what() << std::endl;
	}
	table_schema_ = exec_plan_["table_schema"];
	plan_file.close();
	GetExecInfo();
}

// #define PROFILE_WORKER
Base_Worker::~Base_Worker()
{
	if (stage_no_ < num_stages_ - 1) {	
		ProfileDump();
	}
}

arrow::Status Base_Worker::ConnectServer(arrow::flight::Location location)
{
	ARROW_ASSIGN_OR_RAISE(client_, arrow::flight::FlightClient::Connect(location));
	ARROW_ASSIGN_OR_RAISE(auto r, client_->DoAction(arrow::flight::Action{"test"}));
	// Wait until connected
	auto get_list = client_->ListFlights();
	while (!get_list.ok()) {
		get_list = client_->ListFlights();
	}
	// std::cout << "Connected to " << location.ToString() << std::endl;
	return arrow::Status::OK();
}

arrow::Status Base_Worker::CommitPreproc()
{
	if (preproc_str2int_.empty()) {
		return arrow::Status::OK();
	}
	auto map_schema = arrow::schema({
		arrow::field("int", arrow::int32()),
		arrow::field("str", arrow::utf8())
	});
	arrow::Int32Builder int_builder;
	arrow::StringBuilder str_builder;
	for (const auto& pair : preproc_str2int_){
		ARROW_RETURN_NOT_OK(int_builder.Append(pair.second));
		ARROW_RETURN_NOT_OK(str_builder.Append(pair.first));
	}
	std::shared_ptr<arrow::Int32Array> int_array;
	std::shared_ptr<arrow::StringArray> str_array;
	ARROW_RETURN_NOT_OK(int_builder.Finish(&int_array));
	ARROW_RETURN_NOT_OK(str_builder.Finish(&str_array));
	// auto map_table = arrow::Table::Make(map_schema, {int_array, str_array});
	auto map_rb = arrow::RecordBatch::Make(map_schema, preproc_str2int_.size(), 
																				 {int_array, str_array});
	
	auto descriptor = arrow::flight::FlightDescriptor::Command(subq_id_str_+std::to_string(id_));
	std::unique_ptr<arrow::flight::FlightStreamWriter> writer;
	std::unique_ptr<arrow::flight::FlightMetadataReader> metadata_reader;
	ARROW_ASSIGN_OR_RAISE(auto put_stream, client_->DoPut(descriptor, map_schema));
	writer = std::move(put_stream.writer);
	metadata_reader = std::move(put_stream.reader);

	ARROW_RETURN_NOT_OK(writer->WriteRecordBatch(*map_rb));
	ARROW_RETURN_NOT_OK(writer->Close());

	return arrow::Status::OK();
}

int Base_Worker::PreprocStr2Int(std::string s) {
	int hv = 0;
	if (preproc_str2int_.find(s) != preproc_str2int_.end()) {
		hv = preproc_str2int_.at(s);
	}
	else {
		hv = str_hash_(s);
		preproc_str2int_[s] = hv;
	}
	return hv;
}

arrow::Status Base_Worker::StringArray2Int(std::shared_ptr<arrow::StringArray> string_array, 
																					 std::shared_ptr<arrow::Array>* int_array)
{
	// Use a vector of unique strings as map, the index is mapped int
	arrow::Int32Builder int_builder;
	for (int64_t i = 0; i < string_array->length(); i++) {
		const std::string s = string_array->GetString(i);
		const int hv = PreprocStr2Int(s);
		ARROW_RETURN_NOT_OK(int_builder.Append(hv));
	}
	ARROW_RETURN_NOT_OK(int_builder.Finish(int_array));
	
	return arrow::Status::OK();
}

arrow::Status Base_Worker::DecimalArray2Int(std::shared_ptr<arrow::DecimalArray> decimal_array, 
															 						 	std::shared_ptr<arrow::Array>* int_array)
{
	auto i32_type = arrow::int32();
	auto i32_holder = arrow::TypeHolder(i32_type);
	*int_array = arrow::compute::Cast(*decimal_array, i32_holder).ValueOrDie();

	return arrow::Status::OK();
}

arrow::Status Base_Worker::FixedBinArray2Int(std::shared_ptr<arrow::FixedSizeBinaryArray> bin_array,
																						 std::shared_ptr<arrow::Array>* int_array)
{
	arrow::Int32Builder int_builder;
	for (int64_t i = 0; i < bin_array->length(); i++) {
		std::string s = bin_array->GetString(i);
		const int hv = PreprocStr2Int(s);
		ARROW_RETURN_NOT_OK(int_builder.Append(hv));
	}
	ARROW_RETURN_NOT_OK(int_builder.Finish(int_array));

	return arrow::Status::OK();
}

void Base_Worker::GetExecInfo()
{
	const std::vector<int> num_stage_workers = exec_plan_["num_stage_workers"];
	num_stage_workers_ = num_stage_workers;
	num_stages_ = num_stage_workers.size();

	int num_workers_before = 0;
	for (int s = 0; s < num_stages_; s++) {
		if (id_ < num_workers_before + num_stage_workers[s]) {
			stage_no_ = s;
			id_in_stage_ = id_ - num_workers_before;
			num_workers_same_stage_ = num_stage_workers[s];
			#ifdef EXEC_PLAN_DBG
			std::cout << num_stages_ << " " << stage_no_ << " " << id_in_stage_ 
								<< " " << num_workers_same_stage_ << std::endl;
			#endif
			break;
		}
		else {
			num_workers_before += num_stage_workers[s];
		}
	}
	// Only shuffle workers need to know
	if (stage_no_ == 0) {
		nlohmann::json num_shfl_workers = exec_plan_["num_shfl_workers"];
		int num_before_shfl_workers = 0;
		for (auto it = num_shfl_workers.begin(); it != num_shfl_workers.end(); it++) {
			#ifdef EXEC_PLAN_DBG
			std::cout << it.key() << " " << it.value() << std::endl;
			#endif
			int v = it.value();
			if (id_ < num_before_shfl_workers + v) {
				shfl_table_name_ = it.key();
				num_same_shfl_workers_ = v;
				local_shfl_id_ = id_ - num_before_shfl_workers;
				break;
			}
			else {
				num_before_shfl_workers += v;	
			}
		}
		shfl_col_names_ = exec_plan_["table_schema"][shfl_table_name_]["name"];
		shfl_tgt_stage_ = exec_plan_["shfl_target_stage"][shfl_table_name_];
		num_shfl_tgt_workers_ = num_stage_workers_[shfl_tgt_stage_];
		partition_col_ = exec_plan_["shfl_worker_partition_col"][shfl_table_name_];
		shfl_tgt_start_id_ = 0;
		for (int s = 0; s < shfl_tgt_stage_; s++) {
			shfl_tgt_start_id_ += num_stage_workers_[s];
		}
		#ifdef EXEC_PLAN_DBG
		std::cout << "Shuffle " << shfl_table_name_ << " " << local_shfl_id_ << " / "
							<< num_same_shfl_workers_ << " partition on " << partition_col_ << std::endl;
		#endif
	}
	// Indicate the parition column for non-terminal workers
	else if (stage_no_ < num_stages_ - 1) {
		num_shfl_tgt_workers_ = num_stage_workers_[stage_no_+1];
		res_tbl_name_ = exec_plan_["stage_kernel_info"][stage_no_]["result_name"];
		if (stage_no_ < num_stages_ - 2) {
			auto next_knl = exec_plan_["stage_kernel_info"][stage_no_+1];
			partition_col_ = next_knl["probe_key"][0];
			#ifdef EXEC_PLAN_DBG
			std::cout << "Result will partition on " << partition_col_ << std::endl;
			#endif
		}
	}
	if (stage_no_ > 0) {
		probe_tbl_name_ = exec_plan_["stage_request_table"][stage_no_][0];
	}
	if (stage_no_ == 0) {
		num_out_partitions_ = exec_plan_["stage_out_partitions"][shfl_tgt_stage_ - 1];
	}
	else if (stage_no_ > 0) {
		num_out_partitions_ = exec_plan_["stage_out_partitions"][stage_no_];
		int num_total_partitions = exec_plan_["stage_out_partitions"][stage_no_-1];
		assert(num_total_partitions % num_workers_same_stage_ == 0);
		num_inp_partitions_ = num_total_partitions / num_workers_same_stage_;
		// if (id_in_stage_ < num_total_partitions % num_workers_same_stage_) {
		// 	num_inp_partitions_ += 1;
		// }
	}
	// get the number of input batches of this stage
	for (int s = 1; s <= stage_no_; s++) {
		if (s == 1) {
			const std::string probe_tbl_s1 = exec_plan_["stage_request_table"][1][0];
			const int probe_num_rbs = exec_plan_["num_batches"][probe_tbl_s1];
			const int probe_num_shflw = exec_plan_["num_shfl_workers"][probe_tbl_s1];
			num_inp_probe_batches_ = (probe_num_rbs-1) / probe_num_shflw + 1;
		}
		else {
			int last2_stage_out_p = exec_plan_["stage_out_partitions"][s-2];
			int num_workers_last_stage = num_stage_workers_[s-1];
			num_inp_probe_batches_ *= last2_stage_out_p / num_workers_last_stage;
			// std::cout << stage_no_ << " " << num_inp_probe_batches_ << " " << last2_stage_out_p << " " << num_workers_last_stage << std::endl;
		}
	}
}

arrow::Result<std::shared_ptr<arrow::Schema>> Base_Worker::GetSchemaFromPlan(std::string tbl)
{
	assert(table_schema_.find(tbl) != table_schema_.end());
	std::vector<std::string> col_names = table_schema_[tbl]["name"];
	std::vector<std::string> col_types = table_schema_[tbl]["type"];
	int num_cols = col_names.size();
	
	arrow::FieldVector field_vec;
	for (int c = 0; c < num_cols; c++) {
		std::shared_ptr<arrow::DataType> type;
		if (col_types[c] == "i32") {
			type = arrow::int32();
		}
		else if (col_types[c] == "string") {
			type = arrow::utf8();
		}
		// Size of decimal and fixed bin are hard coded
		else if (col_types[c] == "decimal128") {
			type = arrow::decimal128(18, 2);
		}
		else if (col_types[c] == "fixed_bin") {
			type = arrow::fixed_size_binary(7);
		}
		else {
			return arrow::Status::Invalid(col_names[c], ": ", col_types[c], " not supported");
		}
		field_vec.push_back(arrow::field(col_names[c], type));
	}
	return arrow::schema(field_vec);
}

arrow::Status Base_Worker::GetShflTables(std::vector<std::string> toget_table_names)
{
	for (auto tbl : toget_table_names) {
		ARROW_ASSIGN_OR_RAISE(auto schema, GetSchemaFromPlan(tbl));
		int start_worker = 0, num_worker = 0;
		nlohmann::json num_shfl_workers = exec_plan_["num_shfl_workers"];
		// Delete the possible "_s"
		std::string shfl_src_tbl = tbl;
		if (num_shfl_workers.find(shfl_src_tbl) != num_shfl_workers.end()) {
			// From shuffle workers
			num_worker = num_shfl_workers[shfl_src_tbl];
			for (auto it = num_shfl_workers.begin(); it != num_shfl_workers.end(); it++) {
				if (shfl_src_tbl != it.key()) {
					int n = it.value();
					start_worker += n;
				}
				else {
					break;
				}
			}
		}
		else {
			// From upstream workers
			num_worker = num_stage_workers_[stage_no_-1];
			start_worker = id_ - id_in_stage_ - num_worker;
		}
		ARROW_RETURN_NOT_OK(ShflRecv(tbl, schema, start_worker, num_worker));
	}

	return arrow::Status::OK();
}

void Base_Worker::PrintStats()
{
	if (!std::getenv("PRINT_PROF")) {
		return;
	}
	std::string out(
		"Worker "+ std::to_string(id_) 
		+ " [" + worker_type_ + ": " + std::to_string(stage_no_) + "]\n"
	);
	for (const auto& entry : time_stats_) {
		out += entry.first + ":  " + std::to_string(entry.second.count()) + " ms\n";
	}
	std::cout << out << std::endl;
}

void Base_Worker::DumpStats()
{
	std::ofstream time_log("time.log", std::ios::app);
	if (!time_log.is_open()) {
		std::cout << "Failed to open time stats file" << "time.log" << std::endl;
		exit(1);
	}
	std::string out(
		"Worker "+ std::to_string(id_) 
		+ " [" + worker_type_ + ": " + std::to_string(stage_no_) + "]\n"
	);
	for (const auto& entry : time_stats_) {
		out += entry.first + ":  " + std::to_string(entry.second.count()) + " ms\n";
	}
	time_log << out << std::endl;
	time_log.close();
}

// Get from Flight storage server and shuffle
arrow::Status Base_Worker::GetData()
{
	std::vector<std::string> to_get_tbls = exec_plan_["stage_request_table"][stage_no_];
	std::vector<bool> tbl_from_shfl = exec_plan_["request_table_from_shfl"][stage_no_];
	std::vector<std::string> flight_tbls;
	for (int t = 0; t < to_get_tbls.size(); t++) {
		if (!tbl_from_shfl[t]) {
			flight_tbls.push_back(to_get_tbls[t]);
		}
		else {
			input_shlf_tbls_.push_back(to_get_tbls[t]);
		}
	}
	
	ARROW_RETURN_NOT_OK(GetShflPartitions(input_shlf_tbls_));

	ARROW_RETURN_NOT_OK(GetFlightTables(flight_tbls));
	
	return arrow::Status::OK();
}

std::string Base_Worker::MakeFlightRequest(std::string tbl_name, 
	std::shared_ptr<arrow::Schema> schema, bool full_tbl, int batch_idx = 0)
{
	std::string sep_col_names = "|";
	for (auto col_name : schema->field_names()) {
		sep_col_names += col_name + "|";
	}
	std::string request = tbl_name + sep_col_names + "/";
	if (!full_tbl) {
		request += std::to_string(batch_idx);
		// request += std::to_string(local_shfl_id_) + "/" + std::to_string(num_same_shfl_workers_);
	}
	return request;
}

arrow::Status Base_Worker::GetShflPartitions(std::vector<std::string> toget_table_names)
{
	for (auto tbl : toget_table_names) {
		ARROW_ASSIGN_OR_RAISE(auto schema, GetSchemaFromPlan(tbl));
		int start_worker = 0, num_worker = 0;
		nlohmann::json num_shfl_workers = exec_plan_["num_shfl_workers"];
		std::string shfl_src_tbl = tbl;
		if (num_shfl_workers.find(shfl_src_tbl) != num_shfl_workers.end()) {
			// From shuffle workers
			num_worker = num_shfl_workers[shfl_src_tbl];
			for (auto it = num_shfl_workers.begin(); it != num_shfl_workers.end(); it++) {
				if (shfl_src_tbl != it.key()) {
					int n = it.value();
					start_worker += n;
				}
				else {
					break;
				}
			}
		}
		else {
			// From upstream workers
			num_worker = num_stage_workers_[stage_no_-1];
			start_worker = id_ - id_in_stage_ - num_worker;
		}
		std::unique_lock curr_recv_lock(curr_recv_mutex_);
		curr_recv_batches_[tbl] = 0;
		curr_recv_lock.unlock();
		InitShflRecv(tbl, schema);
		std::future<int> future_recv_batches = std::async(std::launch::async, 
			&Base_Worker::ShflRecvLoop, this, tbl, schema, start_worker, num_worker);
		if (tbl == probe_tbl_name_) {
			probe_future_recv_batches_ = std::move(future_recv_batches);
		}
	}

	return arrow::Status::OK();
}

int Base_Worker::ShflRecvLoop(std::string tbl_name, std::shared_ptr<arrow::Schema> schema,
														 int src_start_id, int num_src_worker)
{
	ThreadSync();
	int end_signal = 0, num_recv_batches = 0, num_inp_batches;
	if (stage_no_ == num_stages_ - 1) {
		num_inp_batches = num_inp_probe_batches_;
	}
	else if (tbl_name == probe_tbl_name_) {
		num_inp_batches = num_inp_probe_batches_;
		while (build_recv_finished_ == 0) {
			// for probe table, wait for all build recv finish
		}
	}
	else {		// For build table, only 1 input batch
		num_inp_batches = 1;
	}
	do {
		// UpdateShflBatch(tbl_name, schema);
		num_recv_batches++;
		std::unique_lock curr_recv_lock(curr_recv_mutex_);
		curr_recv_batches_[tbl_name] = num_recv_batches;
		curr_recv_lock.unlock();
		timer_start("ShflRecv");
		auto st = ShflRecv(tbl_name, schema, src_start_id, num_src_worker);
		timer_end("ShflRecv");
		if (st != arrow::Status::OK()) {
			assert(false && "ShflRecv Failed");
		}
		// for (int w = 0; w < num_src_worker; w++) {
		// 	MPI_Recv(&end_signal, 1, MPI_INT, src_start_id+w, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		// }
	} while(num_recv_batches < num_inp_batches);
	
	if (tbl_name != probe_tbl_name_) {
		build_recv_finished_ = 1;
	}
	
	return num_recv_batches; 
}

int Base_Worker::Execute()
{
	ThreadSync();
	assert(input_shlf_tbls_.size() == 2);
	std::string build_tbl_name = (input_shlf_tbls_[0] == probe_tbl_name_)? input_shlf_tbls_[1] : input_shlf_tbls_[0];

	// bool probe_shfl_finished = false;
	// while (!probe_shfl_finished) {
		while (curr_exec_batches_ < num_inp_probe_batches_) {
			if (curr_recv_batches_[build_tbl_name] == 0) {
				// wait until the build batch is starting received to avoid segfault
				continue;
			}
			// UpdateExecBatch();
			curr_exec_batches_++;
			for (int p = 0; p < num_inp_partitions_; p++) {
				bool all_shlf_tbl_ready = false;
				// Wait until all shuffle tables are ready
				while (!all_shlf_tbl_ready) {
					if (named_partition_ready_[build_tbl_name][0][p] == 0 || 
							named_partition_ready_[probe_tbl_name_][curr_exec_batches_-1][p] == 0) {
						all_shlf_tbl_ready = false;
					}
					else {
						all_shlf_tbl_ready = true;
					}
				}
				auto exec_st = std::chrono::high_resolution_clock::now();
				timer_start("Execution");
				ExecutePartition(curr_exec_batches_-1, p);
				timer_end("Execution");
				auto exec_ed = std::chrono::high_resolution_clock::now();
				if (std::getenv("DHAP_NAIVE")) {
					while (partition_res_status_[curr_exec_batches_-1][p] != 2) {
						// wait until sending finishes
					}
				}
			}
		}
	// 	auto probe_recv_st = probe_future_recv_batches_.wait_for(std::chrono::milliseconds(0));
	// 	probe_shfl_finished = (probe_recv_st == std::future_status::ready);
	// }
	assert(curr_exec_batches_ == probe_future_recv_batches_.get());

	return curr_exec_batches_;
}

void Base_Worker::StartExecute()
{
	if (stage_no_ < num_stages_ - 1) {
		InitExecution();
		future_exec_batches_ = std::async(std::launch::async, &Base_Worker::Execute, this);
	}
}

arrow::Status Base_Worker::Finish()
{
	assert(stage_no_ > 0);
	if (stage_no_ == num_stages_ - 1) {
		int num_recv_batches = probe_future_recv_batches_.get();		// block until recv finished
		ARROW_ASSIGN_OR_RAISE(auto table, AggregateFinalResults(num_recv_batches));
		ARROW_RETURN_NOT_OK(CommitResult(table));
		return arrow::Status::OK();
	}
	int dest_start_id;
	std::shared_ptr<arrow::Schema> res_schema;
	if (stage_no_ < num_stages_ - 1) {
		dest_start_id = id_ - id_in_stage_ + num_workers_same_stage_;
		ARROW_ASSIGN_OR_RAISE(res_schema, GetSchemaFromPlan(res_tbl_name_));
	}

	int curr_finish_batch = 0;
	bool exec_finished = false;
	// while (!exec_finished) {
		while(curr_finish_batch < num_inp_probe_batches_) {
			// std::cout << "Finish batch " << curr_finish_batch << std::endl;
			int ending = 0;
			for (int p = 0; p < num_inp_partitions_; p++) {
				while (partition_res_status_[curr_finish_batch][p] == 0) {
					// block until res are ready
				}
				if (stage_no_ < num_stages_ - 1) {
					// auto table = runtime_ctxt_.db->getTable(res_tbl_name_);
					timer_start("ShflSend");
					ARROW_RETURN_NOT_OK(ShflSend(curr_finish_batch, p, res_schema, dest_start_id));
					timer_end("ShflSend");
					partition_res_status_[curr_finish_batch][p] = 2;
				}
				row_size_t res_num_rows = GetResNumRows(curr_finish_batch, p);
				all_res_num_rows += res_num_rows;
				// The final results are also freed now !
				FreeIntermediates(curr_finish_batch, p);
			}
			curr_finish_batch++;
		}
	// 	auto exec_st = future_exec_batches_.wait_for(std::chrono::milliseconds(0));
	// 	exec_finished = (exec_st == std::future_status::ready);
	// }
	assert(curr_finish_batch == future_exec_batches_.get());
	std::cout << "Total #batch: " << curr_finish_batch
						<< "  Result size: " << all_res_num_rows << std::endl;

	return arrow::Status::OK();
}


#endif