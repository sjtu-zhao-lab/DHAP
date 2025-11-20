#include <string>
#include <iostream>

#include <base_worker.h>

#include <WrappedLLVMEngine.h>
#include "runtime/ExecutionContext.h"
#include "runtime/ArrowDirDatabase.h"

#include <cxxopts.hpp>

#include <mpi.h>
#include <arrow/acero/groupby.h>

#define CPU_SHFL_PROFILE

class CPU_Worker : public Base_Worker
{
	typedef uint8_t* (*mainFunc)();
	using RawBatch = std::vector<uint8_t*>;
	using RawBatchVector = std::vector<RawBatch>;
public:
	CPU_Worker(int id, bool print_res);
	~CPU_Worker();
	arrow::Status GetFlightTables(std::vector<std::string> toget_table_names) override;
	
	// For the initial shuffle workers
	arrow::Status GetAndShfl();
	
private:
	bool loadLLVM(std::string file);
	
	std::unique_ptr<WrappedExecutionEngine> engine_;
	mainFunc main_fn_;
	runtime::ExecutionContext runtime_ctxt_;
	mlir::MLIRContext mlir_ctxt_;
	mlir::OwningOpRef<mlir::ModuleOp> mlir_module_;
	int numArgs_;
	int numResults_;
	std::string res_name_;

	std::shared_ptr<arrow::RecordBatch> RawBatchToArrow(row_size_t num_rows, RawBatch batch,
																											std::vector<buf_size_t> col_size, 
																											std::shared_ptr<arrow::Schema> schema);
	arrow::Result<std::shared_ptr<arrow::Array>> Preprocess(std::shared_ptr<arrow::Array> col);

	arrow::Status ShflRecv(std::string tbl_name, std::shared_ptr<arrow::Schema> schema,
												 int src_start_id, int num_src_worker) override;
	arrow::Status ShflSend(std::shared_ptr<arrow::Table> table, 
												 std::shared_ptr<arrow::Schema> send_schema, 
												 int dest_start_id);
	// --------------------- For asynchronous execution --------------------- 
	arrow::Status ShflSend(int in_batch, int in_partition,
												 std::shared_ptr<arrow::Schema> send_schema, int dest_start_id) override;
	// named_partition_table_[tbl_name][batch][partition_id]
	std::unordered_map<std::string, std::vector<std::vector<std::shared_ptr<arrow::Table>>>> named_partition_table_;
	std::unordered_map<std::string, std::vector<std::vector<RawBatchVector>>> named_partition_raw_batches_;
	std::vector<std::vector<std::shared_ptr<arrow::Table>>> partition_res_tables_;
	
	void InitShflRecv(std::string tbl_name, std::shared_ptr<arrow::Schema> schema) override;
	void UpdateShflBatch(std::string tbl_name, std::shared_ptr<arrow::Schema> schema) override;
	void UpdateShflBatchPartition(std::string tbl_name, int b, int p, 
																std::shared_ptr<arrow::Table> table, 
																RawBatchVector raw_batches);
	void InitExecution() override;
	void UpdateExecBatch() override;
	void ExecutePartition(int b, int p) override;
	row_size_t GetResNumRows(int in_batch, int in_partition) override;

	uint64_t GetWorkerUsedMem() override;
	std::shared_mutex used_mem_size_mutex_;
	uint64_t used_mem_size_ = 0;		// in bytes
	uint64_t flight_used_mem_ = 0;
	uint64_t mpi_inp_used_mem_ = 0;
	uint64_t res_used_mem_ = 0;

	uint64_t mpi_inp_mem_limit_;
	uint64_t res_mem_limit_;
	inline uint64_t GetTableSize(std::shared_ptr<arrow::Table> table);
	void FreeInput(int b, int p);
	void FreeIntermediates(int in_batch, int in_partition) override;

	bool print_res_;
	arrow::Status CommitResult(std::shared_ptr<arrow::Table> res) override;
	arrow::Result<std::shared_ptr<arrow::Table>> AggregateFinalResults(int num_batches) override;
	arrow::Result<std::shared_ptr<arrow::Table>>
	MapFinalRes(std::shared_ptr<arrow::Table> table, std::shared_ptr<arrow::Table> map);
};

CPU_Worker::CPU_Worker(int id, bool print_res) 
	: Base_Worker(id), numArgs_(0), numResults_(0), print_res_(print_res)
{
	worker_type_ = "CPU";
	if (stage_no_ > 0 && stage_no_ < num_stages_ - 1) {
		// InitExecution();
		runtime_ctxt_.id=42;
	}
	// mpi_inp_mem_limit_ = 1024 * 1024 * 1024;		// default to 2G
	// mpi_inp_mem_limit_ *= 2;
	// res_mem_limit_ = 1024 * 1024 * 1024;				// default to 2G
	// res_mem_limit_ *= 2;
	if (std::getenv("L_CPU")) {
		double l_cpu = std::stof(std::getenv("L_CPU")) * 1024*1024*1024;
		mpi_inp_mem_limit_ = l_cpu * INP_MEM_RATIO;
		res_mem_limit_ = l_cpu * (1 - INP_MEM_RATIO);
	}
	else {
		assert(false && "memory limit should be set");
	}
}

// #define PRINT_CPU_USED_MEM
CPU_Worker::~CPU_Worker()
{
	if (stage_no_ > 0 && stage_no_ < num_stages_ - 1) {
		#ifdef PRINT_CPU_USED_MEM
		std::cout << "flight used mem " << flight_used_mem_ / (1024*1024) << "MB" << std::endl;
		std::cout << "mpi input used mem " << mpi_inp_used_mem_ / (1024*1024) << "MB" << std::endl;
		std::cout << "res used mem " << res_used_mem_ / (1024*1024) << "MB" << std::endl;
		#endif
	}
}

inline uint64_t CPU_Worker::GetTableSize(std::shared_ptr<arrow::Table> table)
{
	uint64_t size = table->num_columns() * table->num_rows() * sizeof(int32_t);
	return size;
}

arrow::Status CPU_Worker::GetFlightTables(std::vector<std::string> toget_table_names)
{
	// Direcly new a DataBase, the old one will be deprecated auto
	auto db = std::make_unique<runtime::ArrowDirDatabase>();
	db->setWriteback(false);
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
		// 	std::cout << table_name << " has " << table->num_rows() << " rows" << std::endl;
		// 	table_name = table_name.substr(0, sep_pos);
		// 	#ifdef DBG_PRINT
		// 	std::cout << table->schema()->ToString() << std::endl;
		// 	#endif
		// }

		// Preprocess the table into i32
		ARROW_ASSIGN_OR_RAISE(auto new_schema, GetSchemaFromPlan(table_name));
		arrow::RecordBatchVector new_batches;
		for (auto batch : batches) {
			arrow::ArrayVector preproc_cols;
			for (auto col_name : new_schema->field_names()) {
				auto col = batch->GetColumnByName(col_name);
				ARROW_ASSIGN_OR_RAISE(auto new_col, Preprocess(col));
				preproc_cols.push_back(new_col);
			}
			auto new_batch = arrow::RecordBatch::Make(new_schema, batch->num_rows(), preproc_cols);
			new_batches.push_back(new_batch);
		}

		ARROW_ASSIGN_OR_RAISE(auto table, arrow::Table::FromRecordBatches(new_batches));
		db->addATable(table_name, table);
		uint64_t table_size = GetTableSize(table);
		std::unique_lock ms_lock(used_mem_size_mutex_);
		used_mem_size_ += table_size;
		flight_used_mem_ += table_size;
	}

	ARROW_RETURN_NOT_OK(CommitPreproc());
	runtime_ctxt_.db = std::move(db);
	return arrow::Status::OK();
}

void CPU_Worker::InitExecution()
{
	std::string llvm_path 
		= plan_dir_ + "/subq" + subq_id_str_ + "stage" + std::to_string(stage_no_) + ".mlir";
	if (!loadLLVM(llvm_path)) {
		assert(false && "Cannot load LLVM code");
	}
	// Initialize LLVM targets.
	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();
	auto targetTriple = llvm::sys::getDefaultTargetTriple();
	std::string errorMessage;
	const auto* target = llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
	if (!target) {
		assert(false && "Cannot get a LLVM target");
	}

	// llvm::InitializeNativeTarget();
	// llvm::InitializeNativeTargetAsmPrinter();
	
	engine_ = std::make_unique<WrappedExecutionEngine>(mlir_module_.get(), RunMode::DEFAULT);
	if (!engine_->succeeded()) {
		assert(false && "Cannot get a LLVM engine");
	}
	// Set runtime context
	typedef uint8_t* (*myfunc)(void*);
	auto fn = (myfunc) engine_->getSetContextPtr();
	fn(&runtime_ctxt_);
	assert(numResults_ == 1);
	main_fn_ = (mainFunc) engine_->getMainFuncPtr();
	
	if (stage_no_ < num_stages_ - 1) {
		const auto& stage_info = exec_plan_["stage_kernel_info"][stage_no_];
		assert(stage_info.find("result_name") != stage_info.end());
		res_name_ = stage_info["result_name"];
	}
	else {
		res_name_ = "result";
	}

	// std::unique_lock res_lock(res_mutex_);
	partition_res_tables_.resize(num_inp_probe_batches_);
	partition_res_status_.resize(num_inp_probe_batches_);
	for (int b = 0; b < num_inp_probe_batches_; b++) {
		partition_res_tables_[b].resize(num_inp_partitions_);
		partition_res_status_[b].resize(num_inp_partitions_, 0);
	}
}

void CPU_Worker::ExecutePartition(int b, int p) 
{
	std::string build_tbl_name = (input_shlf_tbls_[0] == probe_tbl_name_)? input_shlf_tbls_[1] : input_shlf_tbls_[0];
	runtime_ctxt_.db->addATable(build_tbl_name, named_partition_table_[build_tbl_name][0][p]);
	runtime_ctxt_.db->addATable(probe_tbl_name_, named_partition_table_[probe_tbl_name_][b][p]);
	// Run main function
	uint8_t* res = main_fn_();
	FreeInput(b, p);

	auto table = *(std::shared_ptr<arrow::Table>*) res;
	std::unique_lock lock(res_mutex_);
	partition_res_tables_[b][p] = table;
	partition_res_status_[b][p] = 1;
	lock.unlock();
	uint64_t res_table_size = GetTableSize(table);
	while (res_used_mem_ + res_table_size > res_mem_limit_) {
		// block until ...
	}
	std::unique_lock ms_lock(used_mem_size_mutex_);
	used_mem_size_ += res_table_size;
	res_used_mem_ += res_table_size;

#ifdef BW_PROFILE
	exec_rows_.push_back(named_partition_table_[probe_tbl_name_][b][p]->num_rows());
#endif
}

void CPU_Worker::FreeInput(int b, int p)
{
	// Free raw batches
	RawBatchVector raw_batches = named_partition_raw_batches_[probe_tbl_name_][b][p];
	for (RawBatch& batch : raw_batches) {
		for (uint8_t* col : batch) {
			free(col);
		}
	}
	// Update the used mem size
	auto table = named_partition_table_[probe_tbl_name_][b][p];
	uint64_t table_size = GetTableSize(table);
	table.reset();
	std::unique_lock ms_lock(used_mem_size_mutex_);
	used_mem_size_ -= table_size;
	mpi_inp_used_mem_ -= table_size;
}

void CPU_Worker::FreeIntermediates(int in_batch, int in_partition)
{
	auto table = partition_res_tables_[in_batch][in_partition];
	uint64_t table_size = GetTableSize(table);
	table.reset();
	std::unique_lock ms_lock(used_mem_size_mutex_);
	used_mem_size_ -= table_size;
	res_used_mem_ -= table_size;
}

row_size_t CPU_Worker::GetResNumRows(int in_batch, int in_partition)
{
	std::shared_lock lock(res_mutex_);
	return partition_res_tables_[in_batch][in_partition]->num_rows();
}

uint64_t CPU_Worker::GetWorkerUsedMem()
{
	std::cout << "flight used mem " << flight_used_mem_ / (1024*1024) << "MB" << std::endl;
	std::cout << "mpi input used mem " << mpi_inp_used_mem_ / (1024*1024) << "MB" << std::endl;
	std::cout << "res used mem " << res_used_mem_ / (1024*1024) << "MB" << std::endl;

	return used_mem_size_;		// KB to bytes
}

arrow::Status CPU_Worker::ShflSend(int in_batch, int in_partition,
																	 std::shared_ptr<arrow::Schema> send_schema, int dest_start_id)
{
	std::shared_lock lock(res_mutex_);
	auto table = partition_res_tables_[in_batch][in_partition];
	lock.unlock();
	ARROW_RETURN_NOT_OK(ShflSend(table, send_schema, dest_start_id));

	return arrow::Status::OK();
}

// #define CPU_SHFL_DBG_PRINT
arrow::Status CPU_Worker::ShflSend(std::shared_ptr<arrow::Table> table, 
																	 std::shared_ptr<arrow::Schema> send_schema, int dest_start_id)
{
	int num_cols = table->num_columns();
	
	arrow::TableBatchReader batch_reader(table);
	ARROW_ASSIGN_OR_RAISE(auto batches, batch_reader.ToRecordBatches());
	int num_batches = batches.size();
	#ifdef CPU_SHFL_DBG_PRINT
	std::cout << "res #batch: " << num_batches << " size: " << table->num_rows() << std::endl;
	#endif
	if (num_batches == 0 || table->num_rows() == 0) {		// a CPU only bug
		row_size_t zero = 0;
		int req_size = num_out_partitions_ * num_cols;
		MPI_Request send_shfl_reqs[req_size];
		int req_idx = 0;			// Unique local request index for each column
		for (int p = 0; p < num_out_partitions_; p++) {
			int tgt_worker_id = dest_start_id + (p % num_shfl_tgt_workers_); 
			MPI_CHECK(MPI_Send(&zero, 1, MPI_ROW_SIZE_T, tgt_worker_id, 0, MPI_COMM_WORLD));
			for (int c = 0; c < num_cols; c++) {
				int mpi_tag = MPI_TAGC + c + p*num_cols + id_*MPI_TAGC;
				MPI_CHECK(MPI_Send(&zero, 1, MPI_BUF_SIZE_T, tgt_worker_id, 0, MPI_COMM_WORLD));
				MPI_CHECK(MPI_Isend(&zero, 0, MPI_UINT8_T, tgt_worker_id,
														mpi_tag, MPI_COMM_WORLD, &send_shfl_reqs[req_idx++]));
			}
		}
		MPI_CHECK(MPI_Waitall(req_size, send_shfl_reqs, MPI_STATUS_IGNORE));
		return arrow::Status::OK();
	}
	assert(num_batches == 1);

	std::vector<std::unique_ptr<arrow::Int32Builder>> partition_indices_builder;
	for (int p = 0; p < num_out_partitions_; p++) {
		partition_indices_builder.push_back(std::make_unique<arrow::Int32Builder>());
	}

	// Reorder columns according to schema (from plan) and preprocess to int32
#ifdef CPU_SHFL_PROFILE
	timer_start("ShflSend-Preprocess");
#endif
	for (int b = 0; b < num_batches; b++) {
		auto batch = batches[b];
		arrow::ArrayVector preproc_cols;
		for (auto col_name : send_schema->field_names()) {
			auto col = batch->GetColumnByName(col_name);
			ARROW_ASSIGN_OR_RAISE(col, Preprocess(col));
			preproc_cols.push_back(col);
		}
		batches[b] = arrow::RecordBatch::Make(send_schema, batch->num_rows(), preproc_cols);
	}
#ifdef CPU_SHFL_PROFILE
	timer_end("ShflSend-Preprocess");
#endif

	int req_size = num_batches * num_out_partitions_ * num_cols;
	// MPI_Request send_shfl_size_reqs[req_size];
	MPI_Request send_shfl_reqs[req_size];
	int req_idx = 0;			// Unique local request index for each column

	std::vector<arrow::RecordBatchVector> batch_partitions;
	std::vector<std::vector<row_size_t>> batch_partition_num_rows;
	if (num_out_partitions_ > 1) {
		for (int b = 0; b < num_batches; b++) {
			auto batch = batches[b];
			// Build indices for each partition
			auto part_col = std::static_pointer_cast<arrow::Int32Array>(batch->GetColumnByName(partition_col_));
			auto part_col_buf = reinterpret_cast<const int32_t*>(part_col->data()->buffers[1]->data());
			std::vector<std::vector<int32_t>> all_partition_indices(num_out_partitions_);
			for (int p = 0; p < num_out_partitions_; p++) {
				all_partition_indices[p].reserve(part_col->length() / num_out_partitions_);
			}
			#ifdef CPU_SHFL_PROFILE
			timer_start("ShflSend-Partition Indexing");
			#endif
			// TODO: multi-threads accelerate ?
			for (int32_t i = 0; i < part_col->length(); i++) {
				int32_t tgt = part_col_buf[i] % num_out_partitions_;
				all_partition_indices[tgt].push_back(i);
			}
			#ifdef CPU_SHFL_PROFILE
			timer_end("ShflSend-Partition Indexing");
			#endif
			// Materialize each partition
			arrow::RecordBatchVector partitions;
			std::vector<row_size_t> partition_num_rows;
			for (int p = 0; p < num_out_partitions_; p++) {
				ARROW_RETURN_NOT_OK(partition_indices_builder[p]->AppendValues(all_partition_indices[p]));
				ARROW_ASSIGN_OR_RAISE(auto partition_indices, partition_indices_builder[p]->Finish());
				arrow::Datum part;
				#ifdef CPU_SHFL_PROFILE
				timer_start("ShflSend-Partition Take");
				#endif
				ARROW_ASSIGN_OR_RAISE(part, arrow::compute::Take(batch, partition_indices, arrow::compute::TakeOptions::NoBoundsCheck()));
				auto partition = part.record_batch();
				#ifdef CPU_SHFL_PROFILE
				timer_end("ShflSend-Partition Take");
				#endif
				partitions.push_back(partition);
				partition_num_rows.push_back(partition->num_rows());
			}
			batch_partitions.push_back(partitions);
			batch_partition_num_rows.push_back(partition_num_rows);
		}

		for (int b = 0; b < num_batches; b++) {
			for (int p = 0; p < num_out_partitions_; p++) {
				int tgt_worker_id = dest_start_id + (p % num_shfl_tgt_workers_); 
				auto partition = batch_partitions[b][p];
				MPI_CHECK(MPI_Send(&batch_partition_num_rows[b][p], 1, MPI_ROW_SIZE_T, tgt_worker_id, 
														0, MPI_COMM_WORLD));
				for (int c = 0; c < num_cols; c++) {
					auto col = partition->column(c);
					auto data_buf = col->data()->buffers[1];
					buf_size_t buf_size = data_buf->size();
					int mpi_tag = MPI_TAGC + c + p*num_cols + b*num_out_partitions_*num_cols + id_*MPI_TAGC;
					// MPI_CHECK(MPI_Isend(&buf_size, 1, MPI_BUF_SIZE_T, tgt_worker_id,
					// 										mpi_tag, MPI_COMM_WORLD, &send_shfl_size_reqs[req_idx]));
					MPI_CHECK(MPI_Send(&buf_size, 1, MPI_BUF_SIZE_T, tgt_worker_id, 0, MPI_COMM_WORLD));
					// MPI_CHECK(MPI_Send(data_buf->data(), buf_size, MPI_UINT8_T, tgt_worker_id,
					// 										mpi_tag, MPI_COMM_WORLD));
					MPI_CHECK(MPI_Isend(data_buf->data(), buf_size, MPI_UINT8_T, tgt_worker_id,
															mpi_tag, MPI_COMM_WORLD, &send_shfl_reqs[req_idx]));
					req_idx += 1;
				}
			}
		}
	}
	else {
		for (int b = 0; b < num_batches; b++) {
			auto batch = batches[b];
			int tgt_worker_id = dest_start_id;
			row_size_t num_rows = batch->num_rows();
			MPI_CHECK(MPI_Send(&num_rows, 1, MPI_ROW_SIZE_T, tgt_worker_id, 
													0, MPI_COMM_WORLD));
			for (int c = 0; c < num_cols; c++) {
				auto col = batch->column(c);
				auto data_buf = col->data()->buffers[1];
				buf_size_t buf_size = data_buf->size();
				int mpi_tag = MPI_TAGC + c + b*num_out_partitions_*num_cols + id_*MPI_TAGC;
				MPI_CHECK(MPI_Send(&buf_size, 1, MPI_BUF_SIZE_T, tgt_worker_id, 0, MPI_COMM_WORLD));
				MPI_CHECK(MPI_Isend(data_buf->data(), buf_size, MPI_UINT8_T, tgt_worker_id,
														mpi_tag, MPI_COMM_WORLD, &send_shfl_reqs[req_idx]));
				req_idx += 1;
			}
		}
	}
	assert(req_idx == req_size);
	// MPI_CHECK(MPI_Waitall(req_size, send_shfl_size_reqs, MPI_STATUS_IGNORE));
	MPI_CHECK(MPI_Waitall(req_size, send_shfl_reqs, MPI_STATUS_IGNORE));

#ifdef BW_PROFILE
	shfl_rows_.push_back(table->num_rows());
	shfl_bytes_.push_back(table->num_rows() * num_cols * sizeof(int32_t));
#endif

	return arrow::Status::OK();
}

bool CPU_Worker::loadLLVM(std::string file) 
{
	mlir::DialectRegistry registry;
	registry.insert<mlir::LLVM::LLVMDialect>();
	mlir_ctxt_.appendDialectRegistry(registry);
	mlir::registerLLVMDialectTranslation(mlir_ctxt_);

	llvm::SourceMgr sourceMgr;
	llvm::DebugFlag = false;
	mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &mlir_ctxt_);
	if (loadMLIR(file, mlir_ctxt_, mlir_module_))
			return false;
	// The LLVM file will not be passed to lowering, check its numArg and numRes here
	mlir::ModuleOp moduleOp = mlir_module_.get();
	if (auto mainFunc = moduleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>("main")) {
			numArgs_ = mainFunc.getNumArguments();
			numResults_ = mainFunc.getNumResults();
	}
	return true;
}

// Executed only by the initial shuffle workers
arrow::Status CPU_Worker::GetAndShfl()
{
	ARROW_ASSIGN_OR_RAISE(auto schema, GetSchemaFromPlan(shfl_table_name_));
	std::string request;
	const int num_all_batches = exec_plan_["num_batches"][shfl_table_name_];
	const int num_iters = (num_all_batches-1) / num_same_shfl_workers_ + 1;
	int batch_idx = local_shfl_id_;
	for (int i = 0; i < num_iters; i++) {
		timer_start("Flight get");
		request = MakeFlightRequest(shfl_table_name_, schema, false, batch_idx);
		arrow::RecordBatchVector batches;
		if (batch_idx < num_all_batches) {
			ARROW_ASSIGN_OR_RAISE(auto stream, client_->DoGet(arrow::flight::Ticket{request}));
			ARROW_ASSIGN_OR_RAISE(batches, stream->ToRecordBatches());
		}
		else {
			ARROW_ASSIGN_OR_RAISE(auto empty, arrow::RecordBatch::MakeEmpty(schema));
			batches.push_back(empty);
		}
		ARROW_ASSIGN_OR_RAISE(auto table, arrow::Table::FromRecordBatches(batches));
		timer_end("Flight get");

		timer_start("ShflSend");
		ARROW_RETURN_NOT_OK(ShflSend(table, schema, shfl_tgt_start_id_));
		timer_end("ShflSend");
		
		batch_idx += num_same_shfl_workers_;
	}

	// int ending = 114514;
	// for (int w = 0; w < num_shfl_tgt_workers_; w++) {
	// 	MPI_Send(&ending, 1, MPI_INT, shfl_tgt_start_id_+w, 0, MPI_COMM_WORLD);
	// }
	
	ARROW_RETURN_NOT_OK(CommitPreproc());
	return arrow::Status::OK();
}

void CPU_Worker::InitShflRecv(std::string tbl_name, std::shared_ptr<arrow::Schema> schema)
{
	int num_inp_batches = (tbl_name == probe_tbl_name_)? num_inp_probe_batches_ : 1;
	std::unique_lock named_partition_lock_(named_partition_mutex_);
	named_partition_table_[tbl_name].resize(num_inp_batches);
	named_partition_raw_batches_[tbl_name].resize(num_inp_batches);
	named_partition_ready_[tbl_name].resize(num_inp_batches);
	for (int b = 0; b < num_inp_batches; b++) {
		named_partition_table_[tbl_name][b].resize(num_inp_partitions_);
		named_partition_raw_batches_[tbl_name][b].resize(num_inp_partitions_);
		named_partition_ready_[tbl_name][b].resize(num_inp_partitions_, 0);
	}
}

void CPU_Worker::UpdateShflBatch(std::string tbl_name, std::shared_ptr<arrow::Schema> schema)
{
	std::unique_lock named_partition_lock_(named_partition_mutex_);
	named_partition_table_[tbl_name].emplace_back(std::vector<std::shared_ptr<arrow::Table>>(num_inp_partitions_));
	named_partition_raw_batches_[tbl_name].emplace_back(std::vector<RawBatchVector>(num_inp_partitions_));
	named_partition_ready_[tbl_name].emplace_back(std::vector<uint8_t>(num_inp_partitions_, 0));
}

void CPU_Worker::UpdateExecBatch()
{
	std::unique_lock lock(res_mutex_);
	partition_res_tables_.emplace_back(std::vector<std::shared_ptr<arrow::Table>>(num_inp_partitions_));
	partition_res_status_.emplace_back(std::vector<uint8_t>(num_inp_partitions_, 0));
}

void CPU_Worker::UpdateShflBatchPartition(std::string tbl_name, int b, int p, 
																					std::shared_ptr<arrow::Table> table, 
																					RawBatchVector raw_batches)
{
	std::unique_lock named_partition_lock_(named_partition_mutex_);
	named_partition_raw_batches_[tbl_name][b][p] = raw_batches;
	named_partition_table_[tbl_name][b][p] = table;
	named_partition_ready_[tbl_name][b][p] = 1;
}

arrow::Status CPU_Worker::ShflRecv(std::string tbl_name, std::shared_ptr<arrow::Schema> schema,
																	 int src_start_id, int num_src_worker)
{
	int num_cols = schema->num_fields();

	// Each src worker produces 1 batch
	int num_all_batches = num_src_worker;
	
	// Prepare for raw batches
	std::vector<row_size_t> num_rows_srcw(num_src_worker, 0);

	std::vector<RawBatch> raw_batches(num_all_batches);
	std::vector<std::vector<buf_size_t>> raw_batch_col_size(num_all_batches);
	for (int b = 0; b < num_all_batches; b++) {
		raw_batch_col_size[b].resize(num_cols, 0);
		raw_batches[b].resize(num_cols);
	}
		
	int req_size = num_src_worker * num_cols;
	for (int p = 0; p < num_inp_partitions_; p++) {
		MPI_Request recv_shfl_reqs[req_size];
		int req_idx = 0;
		for (int w = 0; w < num_src_worker; w++) {
			int src_worker_id = src_start_id + w;
#ifdef CPU_SHFL_PROFILE
			timer_start("ShflRecv-Sync size");
#endif
			MPI_CHECK(MPI_Recv(&num_rows_srcw[w], 1, MPI_ROW_SIZE_T, src_worker_id,
													0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
#ifdef CPU_SHFL_PROFILE
			timer_end("ShflRecv-Sync size");
#endif
			total_proc_size_ += num_rows_srcw[w];
			RawBatch& raw_batch = raw_batches[w];
			// Receive column by column
			for (int c = 0; c < num_cols; c++) {
				buf_size_t recv_size = 0;
				int mpi_tag = MPI_TAGC + c + (id_in_stage_+p*num_workers_same_stage_)*num_cols + src_worker_id*MPI_TAGC;
				MPI_CHECK(MPI_Recv(&recv_size, 1, MPI_BUF_SIZE_T, src_worker_id,
														0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
				raw_batch_col_size[w][c] = recv_size;
				if (stage_no_ < num_stages_ - 1) {
#ifdef CPU_SHFL_PROFILE
					timer_start("ShflRecv-Wait input buf");
#endif
					while (mpi_inp_used_mem_ + recv_size > mpi_inp_mem_limit_) {
						// block until ...
					}
#ifdef CPU_SHFL_PROFILE
					timer_end("ShflRecv-Wait input buf");
#endif
				}
				raw_batch[c] = (uint8_t*) malloc(recv_size);
				std::unique_lock ms_lock(used_mem_size_mutex_);
				used_mem_size_ += recv_size;
				mpi_inp_used_mem_ += recv_size;
				ms_lock.unlock();
				// MPI_CHECK(MPI_Recv(raw_batch[c], recv_size, MPI_UINT8_T, src_worker_id,
				// 										mpi_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
				MPI_CHECK(MPI_Irecv(raw_batch[c], recv_size, MPI_UINT8_T, src_worker_id,
														mpi_tag, MPI_COMM_WORLD, &recv_shfl_reqs[req_idx]));
				req_idx += 1;
			}
		}
		assert(req_idx == req_size);
#ifdef CPU_SHFL_PROFILE
		timer_start("ShflRecv-WaitAll");
#endif
		MPI_CHECK(MPI_Waitall(req_size, recv_shfl_reqs, MPI_STATUS_IGNORE));
#ifdef CPU_SHFL_PROFILE
		timer_end("ShflRecv-WaitAll");
#endif
		arrow::RecordBatchVector batches;
		for (int b = 0; b < num_all_batches; b++) {
			const row_size_t batch_num_rows = num_rows_srcw[b];
			if (batch_num_rows == 0) {
				continue;
			}
			auto batch = RawBatchToArrow(batch_num_rows, raw_batches[b], raw_batch_col_size[b], schema);
			batches.push_back(batch);
		}
		std::shared_ptr<arrow::Table> table;
		if (batches.size() == 0) {
			ARROW_ASSIGN_OR_RAISE(table, arrow::Table::MakeEmpty(schema));
		} else {
			ARROW_ASSIGN_OR_RAISE(table, arrow::Table::FromRecordBatches(batches));
		}
		const int batch_pos = curr_recv_batches_[tbl_name] - 1;
		UpdateShflBatchPartition(tbl_name, batch_pos, p, table, raw_batches);

		#ifdef CPU_SHFL_DBG_PRINT
		for (int b = 0; b < num_all_batches; b++) {
			std::cout << "==========Batch " << b << "==========" << std::endl;
			std::cout << batches[b]->ToString() << std::endl;
		}
		#endif
	}

	return arrow::Status::OK();
}

std::shared_ptr<arrow::RecordBatch>
CPU_Worker::RawBatchToArrow(row_size_t num_rows, RawBatch batch, 
														std::vector<buf_size_t> col_size,
														std::shared_ptr<arrow::Schema> schema)
{
	int num_cols = batch.size();
	assert(num_cols == schema->num_fields());

	// std::shared_ptr<arrow::DataType> int32_type = arrow::int32();
	// schema = arrow::schema({
	// 		arrow::field("p_partkey", int32_type),
	// 		arrow::field("p_brand1", int32_type),
	// 		arrow::field("p_category", int32_type)
  // });
	
	int valid_size = (num_rows-1)/8+1;
	uint8_t* validity = (uint8_t*) malloc(valid_size);
	// Set all data as valid now
	for (int i = 0; i < valid_size; i++) {
		validity[i] = 0xff;
	}
	auto valid_buf = arrow::Buffer::Wrap(validity, valid_size);
	
	arrow::ArrayDataVector col_data_vec;
	for (int c = 0; c < num_cols; c++) {
		auto col_buf = arrow::Buffer::Wrap(batch[c], col_size[c]);
		auto type = schema->field(c)->type();
		auto col_data = arrow::ArrayData::Make(type, num_rows, {valid_buf, col_buf});
		col_data_vec.push_back(col_data);
	}
	auto record_batch = arrow::RecordBatch::Make(schema, num_rows, col_data_vec);

	return record_batch;
}

arrow::Result<std::shared_ptr<arrow::Array>>
CPU_Worker::Preprocess(std::shared_ptr<arrow::Array> col)
{
	std::shared_ptr<arrow::Array> toint;
	auto type = col->type_id();
	// std::cout << type << std::endl;
	if (type == arrow::Type::INT32 || type == arrow::Type::DATE32 || type == arrow::Type::DATE64) {
		return col;
	}
	else if (type == arrow::Type::DECIMAL128) {
		ARROW_RETURN_NOT_OK(
			DecimalArray2Int(std::static_pointer_cast<arrow::Decimal128Array>(col), &toint)
		);
	}
	else if (type == arrow::Type::STRING) {
		ARROW_RETURN_NOT_OK(
			StringArray2Int(std::static_pointer_cast<arrow::StringArray>(col), &toint)
		);
	}
	else if (type == arrow::Type::FIXED_SIZE_BINARY) {
		ARROW_RETURN_NOT_OK(
			FixedBinArray2Int(std::static_pointer_cast<arrow::FixedSizeBinaryArray>(col), &toint)
		);
		// std::cout << col->ToString() << std::endl;
		// std::cout << toint->ToString() << std::endl;
	}
	else {
		assert(false);
	}
	return toint;
}

arrow::Result<std::shared_ptr<arrow::Table>> CPU_Worker::AggregateFinalResults(int num_batches)
{
	std::vector<std::shared_ptr<arrow::Table>> tables;
	for (int b = 0; b < num_batches; b++) {
		for (int p = 0; p < num_inp_partitions_; p++) {
			assert(named_partition_ready_[probe_tbl_name_][b][p] == 1 && "The data should be ready");
			auto table = named_partition_table_[probe_tbl_name_][b][p];
			tables.push_back(table);
		}
	}
	ARROW_ASSIGN_OR_RAISE(auto merged_table, arrow::ConcatenateTables(tables));
	std::cout << "Merged size: " << merged_table->num_rows() << std::endl;
	// std::cout << merged_table->ToString() << std::endl;
	// Group again if needed
	const auto last_knl_info = exec_plan_["stage_kernel_info"][num_stages_-2];
	if (last_knl_info.find("aggr_col") != last_knl_info.end()) {
		std::vector<arrow::FieldRef> gb_key_fields;
		for (std::string k : last_knl_info["groupby"]) {
			gb_key_fields.emplace_back(k);
		}
		std::vector<arrow::compute::Aggregate> aggrs;
		for (std::string c : last_knl_info.at("aggr_res_col")) {
			aggrs.emplace_back("hash_sum", arrow::FieldRef(c), c);
		}
		ARROW_ASSIGN_OR_RAISE(auto grouped, 
													arrow::acero::TableGroupBy(merged_table, aggrs, gb_key_fields));
		// std::cout << grouped->ToString() << std::endl;
		// Map strings back
		ARROW_ASSIGN_OR_RAISE(auto stream, 
			client_->DoGet(arrow::flight::Ticket{"@"+subq_id_str_+std::to_string(id_)}));
		ARROW_ASSIGN_OR_RAISE(auto map_rbs, stream->ToRecordBatches());
		ARROW_ASSIGN_OR_RAISE(auto map_table, arrow::Table::FromRecordBatches(map_rbs));
		if (map_table->num_rows() == 0) {
			return merged_table;
		}
		arrow::compute::Aggregate distinct_aggr("hash_distinct", "int", "int");
		ARROW_ASSIGN_OR_RAISE(auto distinct_map, 
													arrow::acero::TableGroupBy(map_table, {}, {"int", "str"}));
		ARROW_ASSIGN_OR_RAISE(auto maped, MapFinalRes(grouped, distinct_map));
		// Process sort
		std::shared_ptr<arrow::Table> final;
		if (last_knl_info.find("sort") != last_knl_info.end()) {
			std::vector<arrow::compute::SortKey> sort_keys;
			for (int c = 0; c < last_knl_info["sort"]["name"].size(); c++) {
				const std::string name = last_knl_info["sort"]["name"][c];
				const int order = last_knl_info["sort"]["order"][c];
				sort_keys.emplace_back(arrow::FieldRef(name),
					order? arrow::compute::SortOrder::Ascending : arrow::compute::SortOrder::Descending
				);
			}
			auto sort_opt = arrow::compute::SortOptions::Defaults();
			sort_opt.sort_keys = sort_keys;
			ARROW_ASSIGN_OR_RAISE(auto sort_indices,
														arrow::compute::CallFunction("sort_indices", {maped}, &sort_opt))
			ARROW_ASSIGN_OR_RAISE(auto sorted_datum,
														arrow::compute::Take(maped, sort_indices.array()));
			final = sorted_datum.table();
		}
		else {
			final = maped;
		}
		if (print_res_)
			std::cout << final->ToString() << std::endl;
		std::cout << "Final result size: " << final->num_rows() << std::endl;
		return final;
	}
	return merged_table;
}

arrow::Result<std::shared_ptr<arrow::Table>>
CPU_Worker::MapFinalRes(std::shared_ptr<arrow::Table> table, std::shared_ptr<arrow::Table> map)
{
	auto batch_reader = arrow::TableBatchReader(table);
	ARROW_ASSIGN_OR_RAISE(auto batches, batch_reader.ToRecordBatches());
	auto map_batch_reader = arrow::TableBatchReader(map);
	ARROW_ASSIGN_OR_RAISE(auto map_batches, map_batch_reader.ToRecordBatches());
	assert(batches.size() == 1 && map_batches.size() == 1 && "only map when table is small");
	auto batch = batches[0];
	auto map_batch = map_batches[0];

	auto map_int_col = std::static_pointer_cast<arrow::Int32Array>(map_batch->column(0));
	auto map_str_col = std::static_pointer_cast<arrow::StringArray>(map_batch->column(1));
	std::unordered_map<int, std::string> int2str;
	for (int i = 0; i < map_batch->num_rows(); i++) {
		int2str[map_int_col->Value(i)] = map_str_col->GetString(i);
	}

	const auto col_names = table->ColumnNames();
	const std::string final_res_name = 
		exec_plan_["stage_kernel_info"][num_stages_-2]["result_name"];
	const std::vector<std::string> final_col_names = table_schema_[final_res_name]["name"];
	const std::vector<std::string> final_col_type0 = table_schema_[final_res_name]["type0"];
	std::unordered_map<std::string, bool> cols_to_map;
	for (int c = 0; c < final_col_names.size(); c++) {
		if (final_col_type0[c] == "string" || final_col_type0[c] == "fixed_bin") {
			cols_to_map[final_col_names[c]] = true;
		}
		else {
			cols_to_map[final_col_names[c]] = false;
		}
	}

	std::vector<std::shared_ptr<arrow::Array>> new_cols;
	const int num_rows = batch->num_rows();
	for (int c = 0; c < batch->num_columns(); c++) {
		auto col_name = col_names[c];
		auto col = batch->column(c);
		if (cols_to_map.at(col_name)) {
			std::shared_ptr<arrow::StringArray> new_col;
			arrow::StringBuilder str_builder;
			auto int_col = std::static_pointer_cast<arrow::Int32Array>(col);
			for (int i = 0; i < num_rows; i++) {
				ARROW_RETURN_NOT_OK(str_builder.Append(int2str.at(int_col->Value(i))));
			}
			ARROW_RETURN_NOT_OK(str_builder.Finish(&new_col));
			new_cols.push_back(new_col);
		}
		else {
			new_cols.push_back(col);
		}
	}
	auto new_batch = arrow::RecordBatch::Make(batch->schema(), batch->num_rows(), new_cols);
	return arrow::Table::FromRecordBatches({new_batch});
}

arrow::Status CPU_Worker::CommitResult(std::shared_ptr<arrow::Table> res) {
	const std::string final_res_name = 
		exec_plan_["stage_kernel_info"][num_stages_-2]["result_name"];
	auto descriptor = arrow::flight::FlightDescriptor::Path({final_res_name});
	std::unique_ptr<arrow::flight::FlightStreamWriter> writer;
	std::unique_ptr<arrow::flight::FlightMetadataReader> metadata_reader;
	ARROW_ASSIGN_OR_RAISE(auto put_stream, client_->DoPut(descriptor, res->schema()));
	writer = std::move(put_stream.writer);
	metadata_reader = std::move(put_stream.reader);

	// std::cout << "Commit res size " << res->num_rows() << std::endl;
	ARROW_RETURN_NOT_OK(writer->WriteTable(*res));
	ARROW_RETURN_NOT_OK(writer->Close());

	return arrow::Status::OK();
}

arrow::Status RunClient(int id, bool do_shfl, bool print_res)
{
	CPU_Worker worker(id, print_res);

	arrow::flight::Location location;
	std::string server_ip(std::getenv("SR_IP"));
	ARROW_ASSIGN_OR_RAISE(location, arrow::flight::Location::ForGrpcTcp(server_ip, 36433));
	ARROW_RETURN_NOT_OK(worker.ConnectServer(location));
	
	if (do_shfl) {
		ARROW_RETURN_NOT_OK(worker.GetAndShfl());
	}
	else {
		ARROW_RETURN_NOT_OK(worker.GetData());
	
		worker.StartExecute();

		ARROW_RETURN_NOT_OK(worker.Finish());
	}
	worker.PrintStats();
	return arrow::Status::OK();
}

int main(int argc, char** argv)
{
	// Parse command line args
	cxxopts::Options options("cpu_worker", "CPU worker");
	options.add_options()
		("shfl", "Worker doing partitioning", cxxopts::value<bool>()->default_value("false"))
		("no_print_res", "Print final results", cxxopts::value<bool>()->default_value("false"))
		("h,help", "Print usage")
	;
	auto result = options.parse(argc, argv);
	if (result.count("help")) {
		std::cout << options.help() << std::endl;
		exit(0);
	}
	auto do_shfl = result["shfl"].as<bool>();
	auto print_res = ! result["no_print_res"].as<bool>();

	int mpi_id, num_procs, provided;
	// MPI_CHECK(MPI_Init(&argc, &argv));
	MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));
	MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));

	// Modify output formats
	std::streambuf* original_cout_buf = std::cout.rdbuf();
	std::string prefix = "[CPU Worker " + std::to_string(mpi_id) + "] ";
	PrefixBuf prefixBuf(original_cout_buf, prefix);
	std::cout.rdbuf(&prefixBuf);

	auto status = RunClient(mpi_id, do_shfl, print_res);
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