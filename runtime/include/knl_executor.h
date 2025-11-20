#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include <cuda.h>
#include <util.h>

#include <thread>
#include <future>
#include <shared_mutex>

class knl_executor
{
public:
	knl_executor(int id, int num_inp_partitions);
	~knl_executor();

	void SyncContext();
	void LoadModule();
	uint32_t Execute();

	inline bool CheckResCol(std::string res_col) {
		if (device_partitioned_res_col_batches_.find(res_col) == 
				device_partitioned_res_col_batches_.end()) {
			return false;
		}
		return true;
	}
	inline int GetNumResBatch() const noexcept {
		return 1;
	}
	// return[p]: result size of output partition p in comming batch b and input partition `in_p_id`
	inline row_size_t GetPartitionResSize(int b, int in_p_id, int out_p_id) {
		std::shared_lock lock(res_mutex_);
		return batch_partitioned_res_size_[b][in_p_id][out_p_id];
	}
	// return: ptr of output partition `out_p_id` in comming batch b and input partition `in_p_id`
	int32_t* GetResColPartitionPtr(std::string col, int b, int in_p_id, int out_p_id) {
		std::shared_lock lock(res_mutex_);
		row_size_t partition_offset = out_p_id * batch_partition_limit_[b][in_p_id];
		return device_partitioned_res_col_batches_[col][b][in_p_id] + partition_offset;
	}

	// Initialize and update metadata (num of batches) for column
	// `shfl` set to `true` when the data is from shuffling
	void InitFlightColBatch(std::string col_name, int num_batches, bool shfl);
	void UpdateFlightColBatchNumRows(std::string col_name, int batch_idx, row_size_t n);
	void UpdateFlightColBatch(std::string col_name, int batch_idx, const int32_t* p);
	
	uint32_t GetColSize(std::string col_name, int loc, bool is_build) {
		uint32_t size = 0;
		if (loc == 0) {
			for (int b = 0; b < flight_col_num_batches_[col_name]; b++) {
				size += flight_col_batches_size_[col_name][b];
			}
		}
		else if (loc == 1) {
			int curr_exec_batch_ = is_build ? curr_exec_build_batch_ : curr_exec_probe_batch_;
			std::shared_lock lock(mpibuf_batch_mutex_);
			// std::cout << col_name << " " << curr_exec_batch_ << " " << mpibuf_col_size_[col_name].size() << col_partition_num_batches_[col_name].size() << std::endl;
			for (int b = 0; b < col_partition_num_batches_[col_name][curr_exec_batch_][curr_exec_parittion_]; b++) {
				size += mpibuf_col_size_[col_name][curr_exec_batch_][curr_exec_parittion_][b];
			}
		}
		else if (loc == 2) {
			for (int b = 0; b < map_intermediates_num_batches_[col_name]; b++) {
				size += map_intermediates_batches_size_[col_name][b];
			}
		}
		return size;
	}
	inline int GetNumBatches(std::string col_name, int loc, bool is_build) {
		if (loc == 0) {
			return flight_col_num_batches_[col_name];
		}
		else if (loc == 1) {
			int curr_exec_batch_ = is_build ? curr_exec_build_batch_ : curr_exec_probe_batch_;
			std::shared_lock lock(mpibuf_batch_mutex_);
			return col_partition_num_batches_[col_name][curr_exec_batch_][curr_exec_parittion_];
		}
		else if (loc == 2) {
			return map_intermediates_num_batches_[col_name];
		}
	}
	inline row_size_t*& GetBatchesSize(std::string col_name, int loc, bool is_build) {
		if (loc == 0) {
			return flight_col_batches_size_[col_name];
		}
		else if (loc == 1) {
			int curr_exec_batch_ = is_build ? curr_exec_build_batch_ : curr_exec_probe_batch_;
			std::shared_lock lock(mpibuf_batch_mutex_);
			return mpibuf_col_size_[col_name][curr_exec_batch_][curr_exec_parittion_];
		}
		else if (loc == 2) {
			return map_intermediates_batches_size_[col_name];
		}
	}
	inline row_size_t GetBatchSize(std::string col_name, int batch_idx, int loc, bool is_build) {
		if (loc == 0) {
			return flight_col_batches_size_[col_name][batch_idx];
		}
		else if (loc == 1) {
			int curr_exec_batch_ = is_build ? curr_exec_build_batch_ : curr_exec_probe_batch_;
			std::shared_lock lock(mpibuf_batch_mutex_);
			return mpibuf_col_size_[col_name][curr_exec_batch_][curr_exec_parittion_][batch_idx];
		}
		else if (loc == 2) {
			return map_intermediates_batches_size_[col_name][batch_idx];
		}
	}

	inline uint8_t* GetMPIBufPtr(std::string col_name, int batch_idx, bool is_build) {
		int curr_exec_batch_ = is_build ? curr_exec_build_batch_ : curr_exec_probe_batch_;
		std::shared_lock lock(mpibuf_batch_mutex_);
		return mpibuf_col_batches_[col_name][curr_exec_batch_][curr_exec_parittion_][batch_idx];
	}

	void InitShflInput(std::vector<std::string> col_names, int num_in_batches);
	void UpdateComingBatchInput(std::vector<std::string> col_names);
	void UpdateComingBatchOutput(std::vector<std::string> payload_cols);
	void InitKnlExecutionParam(std::string kernel_name0,
															std::vector<std::string> build_knl_names0,
															std::vector<std::string> build_key_cols0,
															std::vector<std::string> build_filter_cols0,
															std::vector<std::vector<std::string>> build_payload_cols0,
															std::vector<std::string> probe_key_cols0,
															std::vector<std::string> probe_payload_cols0,
															std::string partition_col0,
															int num_partitions0,
															std::unordered_map<std::string, std::vector<std::string>> computed_cols0,
															std::vector<std::string> aggr_cols0,
															std::vector<std::string> build_gb_key_cols0,
															std::vector<std::string> probe_gb_key_cols0,
															std::vector<std::string> all_gb_keys0,
															std::vector<std::string> aggr_res_cols0,
															std::vector<std::string> res_cols0);
	void InitKnlExecution(std::vector<std::string> payload_cols, int num_in_batches);
	void InitColPartitionBatch(std::vector<std::string> col_name, int b, int p, int num_batches);
	void UpdateColPartitionBatchNumRows(std::vector<std::string> col_names, int b, int p, int p_batch, row_size_t n);
	void UpdateColPartitionBatchSize(std::string col_name, int b, int p, int p_batch, buf_size_t n);
	uint8_t* AllocateColPartitionBuf(std::string col_name, int b, int p, int p_batch, buf_size_t size);
	void UpdateExecution(int b, int p) {
		curr_exec_probe_batch_ = b;
		curr_exec_parittion_ = p;
	}
	void FreeCurrentInput(const std::unordered_set<std::string>& cols);		// only free cols from probe table

	uint64_t GetUsedMemSize() {
		std::shared_lock ms_lock(used_mem_size_mutex_);
		return used_mem_size_;
	}
	// uint64_t GetMPIInputUsedMemSize() {
	// 	std::shared_lock ms_lock(used_mem_size_mutex_);
	// 	return mpi_inp_used_mem_;
	// }
	// uint64_t GetResUsedMemSize() {
	// 	std::shared_lock ms_lock(used_mem_size_mutex_);
	// 	return res_used_mem_;
	// }

	void FreeRes(int in_batch, int in_partition);

private:
	// Parameters for Execute()
	std::string kernel_name;
	std::vector<std::string> build_knl_names;
	std::vector<std::string> build_key_cols;
	std::vector<std::string> build_filter_cols;
	std::vector<std::vector<std::string>> build_payload_cols;
	std::vector<std::string> probe_key_cols;
	std::vector<std::string> probe_payload_cols;
	std::string partition_col;
	int num_partitions;
	std::unordered_map<std::string, std::vector<std::string>> computed_cols;
	std::vector<std::string> aggr_cols;
	std::vector<std::string> build_gb_key_cols;
	std::vector<std::string> probe_gb_key_cols;
	std::vector<std::string> all_gb_keys;
	std::vector<std::string> aggr_res_cols;
	std::vector<std::string> res_cols;

	inline bool ColInBuild(std::string col_name) {
		for (const auto& build_plds : build_payload_cols) {
			for (const auto& build_pld : build_plds) {
				if (col_name == build_pld)	return true;
			}
		}
		for (const auto& build_key : build_key_cols) {
			if (col_name == build_key)	return true;
		}
		for (const auto& build_filter : build_filter_cols) {
			if (col_name == build_filter)	return true;
		}
		return false;
	}
	// Check the existence and location of a column
	// 0: Data from host memory that need be copid
	// 1: Data from shuffle that has already located in device memoyry
	inline int InputColCheck(std::string col) {
		if (map_intermediates_.find(col) != map_intermediates_.end()) {
			return 2;
		}
		if (mpibuf_col_batches_.find(col) != mpibuf_col_batches_.end()) {
			return 1;
		}
		if (flight_col_batchs_dev_.find(col) != flight_col_batchs_dev_.end()) {
			return 0;
		}
		else {
			std::cout << "knl_executor input check: "
								<< "Column " << col << " does not exist" << std::endl;
			exit(1);
		}
	}

	void AllocMapIntermediate(int num_batches, row_size_t* batches_size);
	void FreeMapIntermediate();
	void AllocResBuf(std::string col, row_size_t size);

	// std::string id_;
	CUdevice cu_dev_;
	CUcontext cu_ctxt_;
	CUmodule module_;
	std::vector<std::string> kernel_names_;

	// For data copied from flight data (fused build tables)
	// Number of batches, unified for data from both Flight and shuffle
	std::unordered_map<std::string, int> flight_col_num_batches_;
	// Size in number of rows, stored as array in managed memory for GPU access
	std::unordered_map<std::string, row_size_t*> flight_col_batches_size_;
	// For arrow data that has be got by flight, preprocessed and read-only	
	std::unordered_map<std::string, std::vector<int32_t*>> flight_col_batchs_dev_;
	
	// Input column name to batch buffers (malloced to accept data) 
	// When accept from OpenMPI with CUDA support, they are just device pointers
	int num_inp_partitions_;
	std::shared_mutex mpibuf_batch_mutex_;
	// col_partition_num_batches_[col_name][coming_batch_no.][paritition_id]
	std::unordered_map<std::string, std::vector<std::vector<int>>> col_partition_num_batches_;
	// mpibuf_col_batches_[col_name][coming_batch_no.][paritition_id][partition_batch_idx]
	std::unordered_map<std::string, std::vector<std::vector<std::vector<uint8_t*>>>> mpibuf_col_batches_;
	std::unordered_map<std::string, std::vector<std::vector<row_size_t*>>> mpibuf_col_size_;
	std::unordered_map<std::string, std::vector<std::vector<std::vector<buf_size_t>>>> mpibuf_col_batches_byte_size_;
	int curr_exec_probe_batch_, curr_exec_parittion_;
	const int curr_exec_build_batch_ = 0;

	// For map intermediates
	std::unordered_map<std::string, int**> map_intermediates_;
	std::unordered_map<std::string, row_size_t*> map_intermediates_batches_size_;
	std::unordered_map<std::string, int> map_intermediates_num_batches_;

	// For intermediates output
	std::shared_mutex res_mutex_;
	int num_out_partitions_;
	// batch_partition_limit_[comming_batch][input_partition_id]
	std::vector<std::vector<row_size_t>> batch_partition_limit_;
	// batch_partitioned_res_size_[comming_batch][input_partition_id][output_partition_id]
	std::vector<std::vector<row_size_t*>> batch_partitioned_res_size_;
	// device_partitioned_res_col_batches_[col_name][comming_batch][input_partition_id]
	std::unordered_map<std::string, std::vector<std::vector<int32_t*>>> device_partitioned_res_col_batches_;

	row_size_t num_rows_1time_env_ = 0;

	std::shared_mutex used_mem_size_mutex_;
	uint64_t used_mem_size_ = 0;		// in bytes
	uint64_t flight_used_mem_ = 0;
	uint64_t mpi_inp_used_mem_ = 0;
	uint64_t res_used_mem_ = 0;

	uint64_t mpi_inp_mem_limit_;
	uint64_t res_mem_limit_;
	
	std::chrono::milliseconds probe_knl_time_ = std::chrono::milliseconds(0);
};
