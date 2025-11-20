#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/flight/api.h>
#include <arrow/filesystem/api.h>
#include <arrow/ipc/api.h>
#include <arrow/compute/api.h>

#define NDEBUG

class GPU_ShuffleServer : public arrow::flight::FlightServerBase
{
public:
	explicit GPU_ShuffleServer(std::unordered_map<std::string, arrow::RecordBatchVector> named_batches,
														 std::vector<std::vector<int>> batch_partition_res_size,
														 std::vector<std::vector<int*>> batch_partition_res_probe_indices,
														 std::vector<std::vector<int*>> batch_partition_res_build_indices) 
		: named_batches_(named_batches), batch_partition_res_size_(batch_partition_res_size),
			batch_partition_res_probe_indices_(batch_partition_res_probe_indices),
			batch_partition_res_build_indices_(batch_partition_res_build_indices)
	{
		num_partition_ = batch_partition_res_size_[0].size();
		partition_finished_ = std::vector<int>(num_partition_, 0);
		// std::cout << "Shfl server num_p: " << num_partition_ << std::endl;
		// std::cout << "Shfl server num_p: " << partition_finished.size() << std::endl;
	}

	arrow::Status DoGet(const arrow::flight::ServerCallContext&,
                      const arrow::flight::Ticket& request,
                      std::unique_ptr<arrow::flight::FlightDataStream>* stream) override {
    std::string partition_name = request.ticket;
		// The request for a partition will be in the form of "table_name:0"
		size_t sep_pos = partition_name.find(':');
		if (sep_pos != std::string::npos) {
			std::string table_name = partition_name.substr(0, sep_pos);
			int paritition_id = std::stoi(partition_name.substr(sep_pos+1));
			if (paritition_id >= num_partition_) {
				return arrow::Status::Invalid("Shuffle request ", partition_name, " exceeds partition number");
			}
			else if (table_name!="part" && table_name!="lineorder") {
				return arrow::Status::Invalid("Shuffle request ", partition_name, " is not intermediate table");
			}
			// Pass batches of requsted partition
			// The batches will be extracted based on indices in this DoGet, such that different partition can be extracted in parallel(?)
			arrow::RecordBatchVector req_partition;
			int num_batches = batch_partition_res_size_.size();
			for (int b = 0; b < num_batches; b++) {
				std::shared_ptr<arrow::RecordBatch> partition;
				if (table_name == "lineorder") {
					ARROW_ASSIGN_OR_RAISE(partition, 
																GetBatchFromRawIndices(batch_partition_res_probe_indices_[b][paritition_id],
																											batch_partition_res_size_[b][paritition_id],
																											b, table_name
																));
				}
				else {
					ARROW_ASSIGN_OR_RAISE(partition, 
																GetBatchFromRawIndices(batch_partition_res_build_indices_[b][paritition_id],
																											batch_partition_res_size_[b][paritition_id],
																											0, table_name
																));
				}
				req_partition.push_back(partition);
			}
			ARROW_ASSIGN_OR_RAISE(auto batch_reader, arrow::RecordBatchReader::Make(req_partition));
			*stream = std::unique_ptr<arrow::flight::FlightDataStream>(
				new arrow::flight::RecordBatchStream(batch_reader)
			);
			// std::cout << "Finish passing partition " << paritition_id << std::endl;
			partition_finished_[paritition_id] += 1;
		}
		else {
			return arrow::Status::Invalid("Shuffle request ", partition_name, " is not partition");
		}
		
		return arrow::Status::OK();
	}

	bool AllFinished() {
		for (auto f : partition_finished_) {
			if (f < 2) {
				return false;
			}
		}
		return true;
	}

private:
	std::unordered_map<std::string, arrow::RecordBatchVector> named_batches_;
	std::vector<std::vector<int>> batch_partition_res_size_;
	std::vector<std::vector<int*>> batch_partition_res_probe_indices_;
	std::vector<std::vector<int*>> batch_partition_res_build_indices_;
	int num_partition_;
	std::vector<int> partition_finished_;
	
	arrow::Result<std::shared_ptr<arrow::RecordBatch>>
	GetBatchFromRawIndices(int* indices, int size, int batch_idx, std::string table);
};

arrow::Result<std::shared_ptr<arrow::RecordBatch>>
GPU_ShuffleServer::GetBatchFromRawIndices(int* indices, int size, int batch_idx, std::string table)
{
	// First convert the indices to an arrow::Array (Segfault...)
	// auto indices_buf = arrow::Buffer::Wrap(indices, size * sizeof(int));
	// std::shared_ptr<arrow::ArrayData> indices_data = std::make_shared<arrow::ArrayData>();
	// indices_data->type = arrow::int32();
	// indices_data->buffers.push_back(indices_buf);
	// indices_data->length = size;
	// auto indices_array = arrow::MakeArray(indices_data);
	
	arrow::Int32Builder builder;
	for (int i = 0; i < size; i++) {
		ARROW_RETURN_NOT_OK(builder.Append(indices[i]));
	}
	std::shared_ptr<arrow::Array> indices_array;
	ARROW_RETURN_NOT_OK(builder.Finish(&indices_array));

	// std::cout << size << "/" << named_batches_[table][batch_idx]->num_rows() << std::endl;
	// Then take from source batches
	ARROW_ASSIGN_OR_RAISE(auto partition, 
												arrow::compute::Take(named_batches_[table][batch_idx], indices_array));
	return partition.record_batch();
}