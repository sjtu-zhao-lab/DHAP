#include <iostream>
#include <map>
#include <cuda_runtime.h>

// #include <thrust/iterator/counting_iterator.h>
// #include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/gather.h>

#include <knl_executor.h>
#include <knl_util.cuh>

// #define DBG_BUILD_PRINT
// #define DBG_PROBE_PRINT
// #define DBG_IRES_PRINT
// #define DBG_GB_PRINT
// #define NDEBUG

#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      std::cerr << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                   \
      exit(1);                                                    \
    }                                                             \
  } while(0)

#define CUDART_SAFE_CALL(x)                                       \
  do {                                                            \
    cudaError_t result = x;                                      	\
    if (result != cudaSuccess) {                                 	\
      std::cerr << "\nerror: " #x " failed with error "           \
                << cudaGetErrorString(result) << '\n';						\
      exit(1);                                                    \
    }                                                             \
  } while(0)


knl_executor::knl_executor(int id, int num_inp_partitions)
	: num_inp_partitions_(num_inp_partitions)
{
	int num_dev;
	CUDART_SAFE_CALL(cudaGetDeviceCount(&num_dev));
	CUDA_SAFE_CALL(cuInit(0));
	CUDA_SAFE_CALL(cuDeviceGet(&cu_dev_, id % num_dev));
	CUDA_SAFE_CALL(cuCtxCreate(&cu_ctxt_, 0, cu_dev_));
	
	if (std::getenv("NROWS_1TIME")) {
		num_rows_1time_env_ = std::stoi(std::getenv("NROWS_1TIME"));
	}
	// Set mem limit
	// mpi_inp_mem_limit_ = 1024 * 1024 * 1024;
	// mpi_inp_mem_limit_ *= 2;					// default 2GB
	// res_mem_limit_ = 1024 * 1024 * 1024
	// res_mem_limit_ *= 2;							// default 2GB
	if (std::getenv("L_GPU")) {
		double l_gpu = std::stof(std::getenv("L_GPU")) * 1024*1024*1024;
		mpi_inp_mem_limit_ = l_gpu * INP_MEM_RATIO;
		res_mem_limit_ = l_gpu * (1 - INP_MEM_RATIO);
	}
	else {
		assert(false && "memory limit should be set");
	}
}

// #define PROFILE_KNL_EXE
knl_executor::~knl_executor()
{
	#ifdef PROFILE_KNL_EXE
	std::cout << "Probe Kernel Time (ms): " << probe_knl_time_.count() << std::endl;
	std::cout << "flight used mem " << flight_used_mem_ / (1024*1024) << "MB" << std::endl;
	std::cout << "mpi input used mem " << mpi_inp_used_mem_ / (1024*1024) << "MB" << std::endl;
	std::cout << "res used mem " << res_used_mem_ / (1024*1024) << "MB" << std::endl;
	std::cout << "Probe Kernel Time (ms): " << probe_knl_time_.count() << std::endl;
	#endif
	// CUDA_SAFE_CALL(cuModuleUnload(module_));
	// CUDA_SAFE_CALL(cuCtxDestroy(cu_ctxt_));
	// CUDART_SAFE_CALL(cudaDeviceReset());
}

void knl_executor::SyncContext()
{
	// CUDA_SAFE_CALL(cuCtxCreate(&cu_ctxt_, 0, cu_dev_));
	CUDA_SAFE_CALL(cuCtxSetCurrent(cu_ctxt_));
}

void knl_executor::LoadModule()
{
	std::string knl_path = std::getenv("PLAN_DIR");
	knl_path += "/kernel.cubin";
	CUDA_SAFE_CALL(cuModuleLoad(&module_, knl_path.c_str()));
}

// `shfl==true` when called for shuffled data that located on GPU buffer
// `shfl==false` when called for Flight data that need be copied
void knl_executor::InitFlightColBatch(std::string col_name, int num_batches, bool shfl)
{
	flight_col_num_batches_[col_name] = num_batches;
	CUDART_SAFE_CALL(cudaMallocManaged(&flight_col_batches_size_[col_name], sizeof(row_size_t)*num_batches));
	if (shfl) {
		assert(false);
	}
	else {
		flight_col_batchs_dev_[col_name].resize(num_batches, NULL);
	}
}

void knl_executor::InitShflInput(std::vector<std::string> col_names, int num_in_batches)
{
	std::unique_lock lock(mpibuf_batch_mutex_);
	for (auto col_name : col_names) {
		col_partition_num_batches_[col_name].resize(num_in_batches);
		mpibuf_col_batches_[col_name].resize(num_in_batches);
		mpibuf_col_size_[col_name].resize(num_in_batches);
		mpibuf_col_batches_byte_size_[col_name].resize(num_in_batches);
		for (int b = 0; b < num_in_batches; b++) {
			col_partition_num_batches_[col_name][b].resize(num_inp_partitions_);
			mpibuf_col_batches_[col_name][b].resize(num_inp_partitions_);
			mpibuf_col_size_[col_name][b].resize(num_inp_partitions_);
			mpibuf_col_batches_byte_size_[col_name][b].resize(num_inp_partitions_);
		}
	}
}

void knl_executor::UpdateComingBatchInput(std::vector<std::string> col_names)
{
	std::unique_lock lock(mpibuf_batch_mutex_);
	for (auto col_name : col_names) {
		col_partition_num_batches_[col_name].emplace_back(std::vector<int>(num_inp_partitions_));
		mpibuf_col_batches_[col_name].emplace_back(std::vector<std::vector<uint8_t*>>(num_inp_partitions_));
		mpibuf_col_size_[col_name].emplace_back(std::vector<row_size_t*>(num_inp_partitions_));
		mpibuf_col_batches_byte_size_[col_name].emplace_back(std::vector<std::vector<buf_size_t>>(num_inp_partitions_));
	}
}

void knl_executor::UpdateComingBatchOutput(std::vector<std::string> payload_cols)
{
	std::unique_lock lock(res_mutex_);
	batch_partition_limit_.emplace_back(std::vector<row_size_t>(num_inp_partitions_, 0));
	batch_partitioned_res_size_.emplace_back(std::vector<row_size_t*>(num_inp_partitions_, NULL));
	for (auto pld_col : payload_cols) {
		device_partitioned_res_col_batches_[pld_col].emplace_back(std::vector<int32_t*>(num_inp_partitions_, NULL));
	}
}

void knl_executor::InitKnlExecution(std::vector<std::string> payload_cols, int num_in_batches)
{
	std::unique_lock lock(res_mutex_);
	batch_partition_limit_.resize(num_in_batches);
	batch_partitioned_res_size_.resize(num_in_batches);
	for (auto pld_col : payload_cols) {
		device_partitioned_res_col_batches_[pld_col].resize(num_in_batches);
	}
	for (int b = 0; b < num_in_batches; b++) {
		batch_partition_limit_[b].resize(num_inp_partitions_, 0);
		batch_partitioned_res_size_[b].resize(num_inp_partitions_, NULL);
		for (auto pld_col : payload_cols) {
			device_partitioned_res_col_batches_[pld_col][b].resize(num_inp_partitions_, NULL);
		}
	}
}

void knl_executor::InitColPartitionBatch(std::vector<std::string> col_names, int b, int p, int num_batches)
{
	std::unique_lock lock(mpibuf_batch_mutex_);
	for (auto col_name : col_names) {
		col_partition_num_batches_[col_name][b][p] = num_batches;
		mpibuf_col_batches_[col_name][b][p].resize(num_batches, NULL);
		CUDART_SAFE_CALL(cudaMallocManaged(&(mpibuf_col_size_[col_name][b][p]), sizeof(row_size_t)*num_batches));
		mpibuf_col_batches_byte_size_[col_name][b][p].resize(num_batches, 0);
	}
}

void knl_executor::UpdateColPartitionBatchNumRows(std::vector<std::string> col_names, int b, int p, int p_batch, row_size_t n)
{
	for (auto col_name : col_names) {
		mpibuf_col_size_[col_name][b][p][p_batch] = n;
	}
}

void knl_executor::UpdateColPartitionBatchSize(std::string col_name, int b, int p, int p_batch, buf_size_t n)
{
	mpibuf_col_batches_byte_size_[col_name][b][p][p_batch] = n;
}

void knl_executor::UpdateFlightColBatchNumRows(std::string col_name, int batch_idx, row_size_t n)
{
	flight_col_batches_size_[col_name][batch_idx] = n;
}

void knl_executor::UpdateFlightColBatch(std::string col_name, int batch_idx, const int32_t* p)
{
	const row_size_t batch_size = flight_col_batches_size_[col_name][batch_idx];
	CUDART_SAFE_CALL(cudaMalloc(&flight_col_batchs_dev_[col_name][batch_idx], batch_size*sizeof(int32_t)));
	CUDART_SAFE_CALL(cudaMemcpy(flight_col_batchs_dev_[col_name][batch_idx], p,
						 									batch_size*sizeof(int32_t), cudaMemcpyHostToDevice));
	std::unique_lock ms_lock(used_mem_size_mutex_);
	used_mem_size_ += batch_size*sizeof(int32_t);
	flight_used_mem_ += batch_size*sizeof(int32_t);
}

uint8_t* knl_executor::AllocateColPartitionBuf(std::string col_name, int b, int p, int p_batch, buf_size_t size)
{
	while (mpi_inp_used_mem_ + size > mpi_inp_mem_limit_) {
		// Block until has enough mem
	}
	std::unique_lock lock(mpibuf_batch_mutex_);
	uint8_t*& addr = mpibuf_col_batches_[col_name][b][p][p_batch];
	CUDART_SAFE_CALL(cudaMalloc(&addr, size));
	lock.unlock();
	std::unique_lock ms_lock(used_mem_size_mutex_);
	used_mem_size_ += size;
	mpi_inp_used_mem_ += size;

	return addr;
}

// #define FREE_INP_DBG_PRINT
void knl_executor::FreeCurrentInput(const std::unordered_set<std::string>& cols)
{
	#ifdef FREE_INP_DBG_PRINT
	std::cout << "Before Free Inp" << curr_exec_probe_batch_ << curr_exec_parittion_ << " " << mpi_inp_used_mem_/(1024*1024) << std::endl;
	for (auto col : cols) {
		std::cout << col << " ";
	}
	std::cout << std::endl;
	#endif
	for (auto col : cols) {
		const int num_batches = GetNumBatches(col, 1, false);
		for (int b = 0; b < num_batches; b++) {
			row_size_t batch_num_rows = GetBatchSize(col, b, 1, false);
			uint8_t* ptr = GetMPIBufPtr(col, b, false);
			CUDART_SAFE_CALL(cudaFree(ptr));
			std::unique_lock ms_lock(used_mem_size_mutex_);
			used_mem_size_ -= batch_num_rows * sizeof(int32_t);
			mpi_inp_used_mem_ -= batch_num_rows * sizeof(int32_t);
		}
	}
	#ifdef FREE_INP_DBG_PRINT
	std::cout << "After Free Inp" << curr_exec_probe_batch_ << curr_exec_parittion_ << " " << mpi_inp_used_mem_/(1024*1024) << std::endl;
	#endif
}

// #define FREE_RES_DBG_PRINT
void knl_executor::FreeRes(int in_batch, int in_partition)
{
	#ifdef FREE_RES_DBG_PRINT
	std::cout << "Before Free Res" << in_batch << in_partition << " " << res_used_mem_/(1024*1024) << std::endl;
	#endif
	row_size_t res_size = batch_partition_limit_[in_batch][in_partition] * num_out_partitions_;
	std::shared_lock lock(res_mutex_);
	for (auto kv : device_partitioned_res_col_batches_) {
		auto res_buf = kv.second;
		CUDART_SAFE_CALL(cudaFree(res_buf[in_batch][in_partition]));
		std::unique_lock ms_lock(used_mem_size_mutex_);
		used_mem_size_ -= res_size * sizeof(int32_t);
		res_used_mem_ -= res_size * sizeof(int32_t);
	}
	#ifdef FREE_RES_DBG_PRINT
	std::cout << "After Free Res" << in_batch << in_partition << " " << res_used_mem_/(1024*1024) << std::endl;
	#endif
}

void knl_executor::AllocMapIntermediate(int num_batches, row_size_t* batches_size)
{
	for (const auto& p : computed_cols) {
		for (auto col_name : p.second) {
			CUDART_SAFE_CALL(cudaMallocManaged(&map_intermediates_[col_name], 
																					sizeof(int*)*num_batches));
			CUDART_SAFE_CALL(cudaMallocManaged(&map_intermediates_batches_size_[col_name], 
																					num_batches * sizeof(row_size_t)));
			map_intermediates_num_batches_[col_name] = num_batches;
			for (int b = 0; b < num_batches; b++) {
				const row_size_t batch_size = batches_size[b];
				map_intermediates_batches_size_[col_name][b] = batch_size;
				CUDART_SAFE_CALL(cudaMalloc(&map_intermediates_[col_name][b], batch_size * sizeof(int)));
				CUDART_SAFE_CALL(cudaMemset(map_intermediates_[col_name][b], 0, batch_size * sizeof(int)));
				used_mem_size_ += batch_size * sizeof(int);
				res_used_mem_ += batch_size * sizeof(int);
			}
		}
	}
}

void knl_executor::FreeMapIntermediate()
{
	for (const auto& p : computed_cols) {
		for (auto col_name : p.second) {
			for (int b = 0; b < GetNumBatches(col_name, 2, false); b++) {
				CUDART_SAFE_CALL(cudaFree(map_intermediates_[col_name][b]));
				const row_size_t batch_size = map_intermediates_batches_size_[col_name][b];
				used_mem_size_ -= batch_size * sizeof(int);
				res_used_mem_ -= batch_size * sizeof(int);
			}
		}
	}
}

void knl_executor::AllocResBuf(std::string col, row_size_t size)
{
	std::unique_lock res_lock(res_mutex_);
	device_partitioned_res_col_batches_[col].resize(curr_exec_probe_batch_+1);
	device_partitioned_res_col_batches_[col][curr_exec_probe_batch_].resize(curr_exec_parittion_+1);
	CUDART_SAFE_CALL(cudaMalloc(&device_partitioned_res_col_batches_[col][curr_exec_probe_batch_][curr_exec_parittion_],
										sizeof(int) * size));
}

void knl_executor::InitKnlExecutionParam(std::string kernel_name0,
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
																				std::vector<std::string> res_cols0)
{
	kernel_name = kernel_name0;
	build_knl_names = build_knl_names0;
	build_key_cols = build_key_cols0;
	build_filter_cols = build_filter_cols0;
	build_payload_cols = build_payload_cols0;
	probe_key_cols = probe_key_cols0;
	probe_payload_cols = probe_payload_cols0;
	partition_col = partition_col0;
	num_partitions = num_partitions0;
	computed_cols = computed_cols0;
	aggr_cols = aggr_cols0;
	build_gb_key_cols = build_gb_key_cols0;
	probe_gb_key_cols = probe_gb_key_cols0;
	all_gb_keys = all_gb_keys0;
	aggr_res_cols = aggr_res_cols0;
	res_cols = res_cols0;
}

uint32_t knl_executor::Execute()
{
	const int num_fused_join = build_knl_names.size();
	assert(build_key_cols.size()	== num_fused_join);
	assert(build_filter_cols.size() == num_fused_join);
	assert(build_payload_cols.size() == num_fused_join);
	assert(probe_key_cols.size() == num_fused_join);
	assert(partition_col == "" || 
		std::find(probe_payload_cols.begin(), probe_payload_cols.end(), partition_col) 
							!= probe_payload_cols.end());
	
	// Compile the cuda source into cubin, externally with a new process
	// const char* compile_cmd = " nvcc --cubin -o kernel_q21.cubin kernel_q21.cu -arch=sm_80 -I/cuCollections/include/ -I../include --expt-relaxed-constexpr --extended-lambda";
	// int compile_res = system(compile_cmd);
	// if (compile_res) {
	// 	exit(-1);
	// }

	// Load the module and get kernel function
	CUfunction kernel;
	CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module_, kernel_name.c_str()));

	std::vector<void*> args0;
	// ----------------------------------- Build -----------------------------------
	CUfunction build_knls[build_knl_names.size()];
	for (int i = 0; i < build_knl_names.size(); i++) {
		CUDA_SAFE_CALL(cuModuleGetFunction(&build_knls[i], module_, build_knl_names[i].c_str()));
	}

	std::unordered_map<std::string, int32_t**> device_build_col_batches;
	std::unordered_map<std::string, std::unique_ptr<join_ht_type>> named_join_ht;
	std::unordered_map<std::string, std::unique_ptr<join_ht_mview>> named_join_ht_mview;
	std::unordered_map<std::string, std::unique_ptr<join_ht_view>> named_join_ht_view;
	
	std::chrono::milliseconds bt_cpy_time(0);
	std::chrono::milliseconds build_time(0);
	for (int bi = 0; bi < build_key_cols.size(); bi++) {
		std::string build_col = build_key_cols[bi];
		std::string filter_col = build_filter_cols[bi];
		std::vector<std::string> pld_cols = build_payload_cols[bi];
		
		// Check for the location of the build table
		int build_tbl_loc = -1;
		build_tbl_loc = InputColCheck(build_col);
		
		uint32_t total_build_size = GetColSize(build_col, build_tbl_loc, true);
		named_join_ht[build_col] = std::make_unique<join_ht_type>(total_build_size*2, 
																															cuco::empty_key{hash_sentinel},
																															cuco::empty_value{row_idx_sentinel});
		// As the `get_device_view` returns a right-value
		join_ht_mview tm = named_join_ht[build_col]->get_device_mutable_view();
		named_join_ht_mview[build_col] = std::make_unique<join_ht_mview>(tm);
		join_ht_view t = named_join_ht[build_col]->get_device_view();
		named_join_ht_view[build_col] = std::make_unique<join_ht_view>(t);
		args0.push_back( named_join_ht_view[build_col].get() );
		
		// Parameters for kernel launch
		constexpr int block_size = 128;
		constexpr int cg_size = 8;
		
		// For build, use all the batches
		int num_build_batches = GetNumBatches(build_col, build_tbl_loc, true);
		CUDART_SAFE_CALL(cudaMallocManaged(&device_build_col_batches[build_col], sizeof(int32_t*)*num_build_batches));
		if (filter_col != "") {
			CUDART_SAFE_CALL(cudaMallocManaged(&device_build_col_batches[filter_col], sizeof(int32_t*)*num_build_batches));
		}
		for (auto pld_col : pld_cols) {
			CUDART_SAFE_CALL(cudaMallocManaged(&device_build_col_batches[pld_col], sizeof(int32_t*)*num_build_batches));
		}

		// Build batch-by-batch and prepare the build data batches
		row_size_t batch_row_idx_offset = 0;
		for (int b = 0; b < num_build_batches; b++) {
			std::vector<void*> build_args;
			row_size_t batch_size = GetBatchSize(build_col, b, build_tbl_loc, true);
			build_args.push_back( named_join_ht_mview[build_col].get() );
			build_args.push_back(&batch_row_idx_offset);
			build_args.push_back(&batch_size);

			// For the build column
			int32_t* build_col_batch_ptr = NULL;
			if (build_tbl_loc == 1) {		// On device
				build_col_batch_ptr = reinterpret_cast<int32_t*>(GetMPIBufPtr(build_col, b, true));
			}
			else {		// On host
				build_col_batch_ptr = flight_col_batchs_dev_[build_col][b];
			}
			build_args.push_back(&build_col_batch_ptr);
			device_build_col_batches[build_col][b] = build_col_batch_ptr;
			// For the filter column
			int32_t* filter_col_batch_ptr = NULL;
			if (filter_col != "") {
				if (build_tbl_loc == 1) {
					filter_col_batch_ptr = reinterpret_cast<int32_t*>(GetMPIBufPtr(filter_col, b, true));
				}
				else {
					filter_col_batch_ptr = flight_col_batchs_dev_[filter_col][b];
				}
				build_args.push_back(&filter_col_batch_ptr);
				device_build_col_batches[filter_col][b] = filter_col_batch_ptr;
			}
			// For payload columns
			for (auto pld_col : pld_cols) {
				int32_t* pld_col_batch_ptr = NULL;
				if (build_tbl_loc == 1) {
					pld_col_batch_ptr = reinterpret_cast<int32_t*>(GetMPIBufPtr(pld_col, b, true));
				}
				else {
					pld_col_batch_ptr = flight_col_batchs_dev_[pld_col][b];
				}
				device_build_col_batches[pld_col][b] = pld_col_batch_ptr;
			}
		}

		std::vector<void*> build_args1;
		build_args1.push_back( named_join_ht_mview[build_col].get() );
		build_args1.push_back(&device_build_col_batches[build_col]);
		if (filter_col != "") {
			build_args1.push_back(&device_build_col_batches[filter_col]);
		}
		row_size_t*& col_batches_size = GetBatchesSize(build_col, build_tbl_loc, true);
		build_args1.push_back(&col_batches_size);
		build_args1.push_back(&total_build_size);
		build_args1.push_back(&num_build_batches);
		
		row_size_t num_rows_1time = (num_rows_1time_env_ > 0) ? num_rows_1time_env_ : total_build_size;
		auto const grid_size = (cg_size * num_rows_1time + block_size - 1) / block_size;
		if (total_build_size > 0) {
			CUDA_SAFE_CALL(cuLaunchKernel(
											build_knls[bi],
											grid_size, 1, 1,   // grid dim
											block_size, 1, 1,    // block dim
											0, NULL,             // shared mem and stream
											build_args1.data(),                // arguments
											0) );
		}
		CUDA_SAFE_CALL(cuCtxSynchronize());
		#ifdef DBG_BUILD_PRINT
		std::cout << build_col << ": build size " << total_build_size
							<< ", ht size " << named_join_ht[build_col]->get_size() << std::endl;
		#endif

		// Prepare payloads for current build table
		if (pld_cols.size() > 0) {
			// int num_build_batches = col_num_batches_[build_col];
			args0.push_back(&num_build_batches);
			args0.push_back(&col_batches_size);
			for (auto col : pld_cols) {
				args0.push_back(&device_build_col_batches[col]);
			}
		}
		#ifdef DBG_BUILD_PRINT
		std::cout << build_col << " ht size " << named_join_ht[build_col]->get_size() << std::endl;
		#endif
	}

	// ----------------------------------- Probe & Aggr -----------------------------------
	std::unordered_set<std::string> cols_freeable;
	constexpr int NUM_GROUPS = 1000;

	// Prepare the device pointer for batches of each probe and aggr column
	std::unordered_map<std::string, std::unique_ptr<batch2_pair_equality>> probe_pair_eq;
	std::unordered_map<std::string, int32_t**> device_probe_col_batches;
	std::unordered_map<std::string, int32_t**> device_gb_key_col_batches;
	
	uint32_t total_probe_size = 0;
	int probe_cols_loc = -1;
	for (int p = 0; p < probe_key_cols.size(); p++) {
		auto probe_col = probe_key_cols[p];
		probe_cols_loc = InputColCheck(probe_col);
		const int num_batches = GetNumBatches(probe_col, probe_cols_loc, false);
		CUDART_SAFE_CALL(cudaMallocManaged(&device_probe_col_batches[probe_col], sizeof(int32_t*)*num_batches));
	}
	int num_probe_batches = GetNumBatches(probe_key_cols[0], probe_cols_loc, false);
	for (int b = 0; b < num_probe_batches; b++) {
		total_probe_size += GetBatchSize(probe_key_cols[0], b, probe_cols_loc, false);
	}
	AllocMapIntermediate(num_probe_batches, GetBatchesSize(probe_key_cols[0], probe_cols_loc, false));
	for (auto probe_pld_col : probe_payload_cols) {
		const int num_batches = GetNumBatches(probe_pld_col, probe_cols_loc, false);
		CUDART_SAFE_CALL(cudaMallocManaged(&device_probe_col_batches[probe_pld_col], sizeof(int32_t*)*num_batches));
	}
	for (auto probe_gb_col : probe_gb_key_cols) {
		// probe_cols_loc = InputColCheck(probe_gb_col);
		const int num_batches = GetNumBatches(probe_gb_col, probe_cols_loc, false);
		CUDART_SAFE_CALL(cudaMallocManaged(&device_gb_key_col_batches[probe_gb_col], sizeof(int32_t*)*num_batches));
	}
	assert(aggr_cols.size() <= 1);
	int aggr_cols_loc = -1;
	bool aggr_cols_in_build = false;
	if (aggr_cols.size() > 0) {
		aggr_cols_in_build = ColInBuild(aggr_cols[0]);
	}
	std::unordered_map<std::string, int32_t**> device_aggr_col_batches;
	for (int a = 0; a < aggr_cols.size(); a++) {
		auto aggr_col = aggr_cols[a];
		aggr_cols_loc = InputColCheck(aggr_col);
		const int num_batches = GetNumBatches(aggr_col, aggr_cols_loc, aggr_cols_in_build);
		CUDART_SAFE_CALL(cudaMallocManaged(&device_aggr_col_batches[aggr_col], sizeof(int32_t*)*num_batches));
	}
	// assert(aggr_cols_loc == -1 || aggr_cols_loc == probe_cols_loc);

	std::chrono::milliseconds gt_cpy_time(0);
	for (auto gb_key_col : build_gb_key_cols) {
		int gb_key_col_loc = InputColCheck(gb_key_col);
		const int num_batches = GetNumBatches(gb_key_col, gb_key_col_loc, true);
		CUDART_SAFE_CALL(cudaMallocManaged(&device_gb_key_col_batches[gb_key_col], sizeof(int32_t*)*num_batches));
		// Get data batch-by-batch
		for (int b = 0; b < num_batches; b++) {
			int32_t* gb_key_col_batch_ptr = NULL;
			if (gb_key_col_loc == 1) {
				gb_key_col_batch_ptr = reinterpret_cast<int32_t*>(GetMPIBufPtr(gb_key_col, b, true));
			}
			else {
				gb_key_col_batch_ptr = flight_col_batchs_dev_[gb_key_col][b];
			}
			device_gb_key_col_batches[gb_key_col][b] = gb_key_col_batch_ptr;
		}
		#ifdef DBG_GB_PRINT
		std::cout << "Group key: " << gb_key_col << " with " << num_batches
							<< " batches and locate at " << gb_key_col_loc << std::endl;
		#endif
	}
	// gb_ht_type* gb_ht = new gb_ht_type{NUM_GROUPS*2, cuco::empty_key{groupby_ht_empty_key}, cuco::empty_value{row_idx_sentinel}};;
	std::unique_ptr<gb_ht_type> gb_ht;
	std::unique_ptr<batch_groupby_key_hasher> gb_key_hasher;
	std::unique_ptr<batch_groupby_key_equality> gb_key_eq;

	// Probe batch-by-batch (note that all columns from same probe table should have same number of batches)
	// TODO: as here the groupby keys are join keys and can be store to GPU memory, 
	//			 it can be processed batch-by-batch
	//			 if groupby keys are from probe table which can be very large, a shuffle is needed
	bool is_gb = (build_gb_key_cols.size() > 0 || probe_gb_key_cols.size() > 0);
	row_size_t probe_batch_size = 0;
	num_out_partitions_ = num_partitions;

	// Initialize to hold partitioned intermediates
	int num_res_batches = 1;

	#ifdef DBG_PROBE_PRINT
	std::cout << "probe with " << num_probe_batches << " batches" << std::endl;
	#endif
	std::chrono::milliseconds probe_cpy_time(0);
	std::chrono::microseconds probe_knl_time(0);
	for (int b = 0; b < num_probe_batches; b++) {
		// Launch one kernel for each batch, so the build params are persistent and remains are variable
		std::vector<void*> args = args0;
		probe_batch_size = GetBatchSize(probe_key_cols[0], b, probe_cols_loc, false);
		args.push_back(&probe_batch_size);

		// ================ Prepare input data ================
		auto cpy_start = std::chrono::high_resolution_clock::now();
		// Prepare the probe keys of this batch
		for (int p = 0; p < probe_key_cols.size(); p++) {
			std::string probe_col = probe_key_cols[p], build_col = build_key_cols[p];
			if (probe_cols_loc == 1) {
				device_probe_col_batches[probe_col][b] 
					= reinterpret_cast<int32_t*>(GetMPIBufPtr(probe_col, b, false));
				if (b == 0) {
					cols_freeable.insert(probe_col);
				}
			}
			else {		// Move data to GPU
			 	device_probe_col_batches[probe_col][b] = flight_col_batchs_dev_[probe_col][b];
			}
			// Prepare equality for kernel parameter
			args.push_back(&device_probe_col_batches[probe_col][b]);
			args.push_back(probe_pair_eq[probe_col].get());
		}
		// Prepare probe payloads of this batch
		for (auto pld_col : probe_payload_cols) {
			// int32_t* pld_col_batch_ptr = NULL;
			if (probe_cols_loc == 1) {
				device_probe_col_batches[pld_col][b] 
					= reinterpret_cast<int32_t*>(GetMPIBufPtr(pld_col, b, false));
				if (b == 0) {
					cols_freeable.insert(pld_col);
				}
			}
			else {
				device_probe_col_batches[pld_col][b] = flight_col_batchs_dev_[pld_col][b];
			}
			args.push_back(&device_probe_col_batches[pld_col][b]);
		}
		// Prepare the probe groupby keys of this batch
		for (auto probe_gb_key : probe_gb_key_cols) {
			if (probe_cols_loc == 1) {
				device_gb_key_col_batches[probe_gb_key][b] 
					= reinterpret_cast<int32_t*>(GetMPIBufPtr(probe_gb_key, b, false));
				if (b == 0) {
					cols_freeable.insert(probe_gb_key);
				}
			}
			else {
				device_gb_key_col_batches[probe_gb_key][b] = flight_col_batchs_dev_[probe_gb_key][b];
			}
		}
		// Prepare the aggregations cols
		for (auto aggr_col : aggr_cols) {
			// The size of batch of aggr_col is same as probe_col
			if (aggr_cols_loc == 1) {
				device_aggr_col_batches[aggr_col][b] = 
					reinterpret_cast<int32_t*>(GetMPIBufPtr(aggr_col, b, aggr_cols_in_build));
				if (b == 0) {
					cols_freeable.insert(aggr_col);
				}
			}
			else if (aggr_cols_loc == 0) {
				device_aggr_col_batches[aggr_col][b] = flight_col_batchs_dev_[aggr_col][b];
			}
			else if (aggr_cols_loc == 2) {
				device_aggr_col_batches[aggr_col][b] = map_intermediates_[aggr_col][b];
			}
			args.push_back(&device_aggr_col_batches[aggr_col][b]);
		}
		auto cpy_end = std::chrono::high_resolution_clock::now();
		probe_cpy_time	+= std::chrono::duration_cast<std::chrono::milliseconds>(cpy_end - cpy_start);
	}

	std::vector<void*> argsb = args0;
	argsb.push_back(&total_probe_size);
	argsb.push_back(&num_probe_batches);
	argsb.push_back(&GetBatchesSize(probe_key_cols[0], probe_cols_loc, false));
	for (int p = 0; p < probe_key_cols.size(); p++) {
		std::string probe_col = probe_key_cols[p], build_col = build_key_cols[p];
		int build_col_loc = InputColCheck(build_col);
		probe_pair_eq[probe_col] = std::make_unique<batch2_pair_equality>(
			device_probe_col_batches[probe_col], GetBatchesSize(probe_col, probe_cols_loc, false), GetNumBatches(probe_col, probe_cols_loc, false),
			device_build_col_batches[build_col], GetBatchesSize(build_col, build_col_loc, true), GetNumBatches(build_col, build_col_loc, true)
		);
		argsb.push_back(&device_probe_col_batches[probe_col]);
		argsb.push_back(probe_pair_eq[probe_col].get());
	}
	for (auto pld_col : probe_payload_cols) {
		argsb.push_back(&device_probe_col_batches[pld_col]);
	}

	// Prepare groupby objects
	const int num_gb_keys = all_gb_keys.size();
	std::unordered_map<std::string, bool> is_build_gb_key;
	for (auto k : build_gb_key_cols) {
		is_build_gb_key[k] = true;
	}
	for (auto k : probe_gb_key_cols) {
		is_build_gb_key[k] = false;
	}
	std::vector<row_size_t*> gb_key_bs;
	std::vector<int> gb_key_nb;
	for (const auto& key : all_gb_keys) {
		gb_key_bs.push_back(GetBatchesSize(key, InputColCheck(key), is_build_gb_key.at(key)));
		gb_key_nb.push_back(GetNumBatches(key, InputColCheck(key), is_build_gb_key.at(key)));
	}
	if (num_gb_keys == 2) {
		const auto& key0 = all_gb_keys[0], key1 = all_gb_keys[1];
		gb_key_hasher = std::make_unique<batch_groupby_key_hasher>(
			device_gb_key_col_batches[key0],
			device_gb_key_col_batches[key1],
			gb_key_bs[0], gb_key_bs[1], gb_key_nb[0], gb_key_nb[1]
		);
		gb_key_eq = std::make_unique<batch_groupby_key_equality>(
			device_gb_key_col_batches[key0],
			device_gb_key_col_batches[key1],
			gb_key_bs[0], gb_key_bs[1], gb_key_nb[0], gb_key_nb[1]
		);
	}
	else if (num_gb_keys == 3) {
		const auto& key0 = all_gb_keys[0], key1 = all_gb_keys[1], key2 = all_gb_keys[2];
		gb_key_hasher = std::make_unique<batch_groupby_key_hasher>(
			device_gb_key_col_batches[key0],
			device_gb_key_col_batches[key1],
			device_gb_key_col_batches[key2],
			gb_key_bs[0], gb_key_bs[1], gb_key_bs[2], 
			gb_key_nb[0], gb_key_nb[1], gb_key_nb[2]
		);
		gb_key_eq = std::make_unique<batch_groupby_key_equality>(
			device_gb_key_col_batches[key0],
			device_gb_key_col_batches[key1],
			device_gb_key_col_batches[key2],
			gb_key_bs[0], gb_key_bs[1], gb_key_bs[2], 
			gb_key_nb[0], gb_key_nb[1], gb_key_nb[2]
		);
	}

	uint32_t* d_num_groups;
	row_size_t* partitioned_res_size = NULL;
	row_size_t estimated_p_size = num_partitions? (total_probe_size/num_partitions * 1.5) : 0;
	batch_partition_limit_[curr_exec_probe_batch_][curr_exec_parittion_] = estimated_p_size;
	
	argsb.push_back(&num_partitions);
	argsb.push_back(&estimated_p_size);
	CUDART_SAFE_CALL(cudaMallocManaged(&partitioned_res_size, num_partitions * sizeof(row_size_t)));
	CUDART_SAFE_CALL(cudaMemset(partitioned_res_size, 0, num_partitions * sizeof(row_size_t)));
	argsb.push_back(&partitioned_res_size);
	// Map intermediates
	int map_intermediates_num_batches = num_probe_batches;
	for (const auto& p : computed_cols) {
		auto cols = p.second;
		argsb.push_back(&map_intermediates_num_batches);
		argsb.push_back(&GetBatchesSize(cols[0], 2, false));
		for (auto col_name : cols) {
			argsb.push_back(&map_intermediates_[col_name]);
		}
	}
	if (is_gb) {
		for (auto aggr_col : aggr_cols) {
			argsb.push_back(&device_aggr_col_batches[aggr_col]);
		}
		gb_ht = std::make_unique<gb_ht_type>(NUM_GROUPS, cuco::empty_key{groupby_ht_empty_key}, 
																										 cuco::empty_value{row_idx_sentinel} );
		auto gb_ht_v = gb_ht->get_device_view();
		auto gb_ht_mv = gb_ht->get_device_mutable_view();
		argsb.push_back( &gb_ht_v );
		argsb.push_back( &gb_ht_mv );
		CUDART_SAFE_CALL(cudaMallocManaged(&d_num_groups, sizeof(uint32_t)));
		*d_num_groups = 0;
		argsb.push_back(&d_num_groups);
		// Push left args
		argsb.push_back(gb_key_hasher.get());
		argsb.push_back(gb_key_eq.get());
		batch_partitioned_res_size_[curr_exec_probe_batch_][curr_exec_parittion_] = d_num_groups;
	}
	else {	// A independent join that outputs (partitioned) indices
		batch_partitioned_res_size_[curr_exec_probe_batch_][curr_exec_parittion_] = partitioned_res_size;

		row_size_t all_partitions_size = num_partitions * estimated_p_size;
		for (auto probe_pld_col : probe_payload_cols) {
			while (res_used_mem_ + sizeof(int) * all_partitions_size > res_mem_limit_) {
				// Block until has enough mem
			}
			std::unique_lock res_lock(res_mutex_);
			CUDART_SAFE_CALL(cudaMalloc(&device_partitioned_res_col_batches_[probe_pld_col][curr_exec_probe_batch_][curr_exec_parittion_],
												sizeof(int) * all_partitions_size));
			res_lock.unlock();
			std::unique_lock ms_lock(used_mem_size_mutex_);
			used_mem_size_ += sizeof(int) * all_partitions_size;
			res_used_mem_ += sizeof(int) * all_partitions_size;
			argsb.push_back(&device_partitioned_res_col_batches_[probe_pld_col][curr_exec_probe_batch_][curr_exec_parittion_]);
		}
		for (int f = 0; f < num_fused_join; f++) {
			for (auto build_pld_col : build_payload_cols[f]) {
				while (res_used_mem_ + sizeof(int) * all_partitions_size > res_mem_limit_) {
					// Block until has enough mem
				}
				std::unique_lock res_lock(res_mutex_);
				CUDART_SAFE_CALL(cudaMalloc(&device_partitioned_res_col_batches_[build_pld_col][curr_exec_probe_batch_][curr_exec_parittion_],
													sizeof(int) * all_partitions_size));
				res_lock.unlock();
				std::unique_lock ms_lock(used_mem_size_mutex_);
				used_mem_size_ += sizeof(int) * all_partitions_size;
				res_used_mem_ += sizeof(int) * all_partitions_size;
				argsb.push_back(&device_partitioned_res_col_batches_[build_pld_col][curr_exec_probe_batch_][curr_exec_parittion_]);
			}
		}
	}

	constexpr int block_size = 128;
	constexpr int cg_size = 8;
	row_size_t num_rows_1time = (num_rows_1time_env_ > 0) ? num_rows_1time_env_ : total_probe_size;
	auto const grid_size = (cg_size * num_rows_1time + block_size - 1) / block_size;
	std::chrono::milliseconds knl_time0 = std::chrono::milliseconds(0);
	if (num_rows_1time != 0) {
		TIME_COST(knl_time0,
		CUDA_SAFE_CALL(cuLaunchKernel(kernel,
										grid_size, 1, 1,   // grid dim
										block_size, 1, 1,    // block dim
										1024, NULL,             // shared mem and stream
										argsb.data(),                // arguments
										0) );
		CUDA_SAFE_CALL(cuCtxSynchronize());
		);
		probe_knl_time_ += knl_time0;
	}
	if (is_gb) {
		groupby_ht_keyT* key_idx;
		row_size_t* aggr_idx;
		const row_size_t num_groups = *d_num_groups;
		CUDART_SAFE_CALL(cudaMalloc(&key_idx, sizeof(groupby_ht_keyT)*num_groups));
		CUDART_SAFE_CALL(cudaMalloc(&aggr_idx, sizeof(row_size_t)*num_groups));
		gb_ht->retrieve_all(key_idx, thrust::make_discard_iterator());
		gb_ht->find(key_idx, key_idx+num_groups, aggr_idx, *gb_key_hasher, *gb_key_eq);
		for (auto col : res_cols) {
			AllocResBuf(col, num_groups);
		}
		assert(aggr_cols.size() == aggr_res_cols.size());
		for (int a = 0; a < aggr_cols.size(); a++) {
			const auto aggr_col = aggr_cols[a];
			const auto aggr_res = aggr_res_cols[a];
			batch_iterator<int*> aggr_batch_iterator(
				device_aggr_col_batches[aggr_col], 
				GetBatchesSize(aggr_col, aggr_cols_loc, aggr_cols_in_build),
				GetNumBatches(aggr_col, aggr_cols_loc, aggr_cols_in_build)
			);
			std::shared_lock res_lock(res_mutex_);
			thrust::gather(thrust::device, aggr_idx, aggr_idx+num_groups, aggr_batch_iterator,
				device_partitioned_res_col_batches_[aggr_res][curr_exec_probe_batch_][curr_exec_parittion_]);
		}
		for (int g = 0; g < all_gb_keys.size(); g++) {
			const auto gb_key = all_gb_keys[g];
			batch_iterator<int*> gb_key_batch_iterator(
				device_gb_key_col_batches[gb_key], gb_key_bs[g], gb_key_nb[g]
			);
			auto gb_key_idx = thrust::make_transform_iterator(key_idx, get_first(g));
			std::shared_lock res_lock(res_mutex_);
			thrust::gather(thrust::device, gb_key_idx, gb_key_idx+num_groups, gb_key_batch_iterator,
										 device_partitioned_res_col_batches_[gb_key][curr_exec_probe_batch_][curr_exec_parittion_]);
		}
	}
	FreeMapIntermediate();
	FreeCurrentInput(cols_freeable);

	#ifdef DBG_PROBE_PRINT
	if (is_gb) {
		std::cout << "(All batch) group size: " << *d_num_groups << std::endl;
	}
	else {
		assert(num_partitions > 0);
		for (int p = 0; p < num_partitions; p++) {
			std::cout << "Kernel: Partition " << p << " res size: " <<  partitioned_res_size[p] << std::endl;
		}
	}
	#endif
	return total_probe_size;
}

#undef DBG_BUILD_PRINT
#undef DBG_PROBE_PRINT
#undef DBG_IRES_PRINT
#undef DBG_GB_PRINT
