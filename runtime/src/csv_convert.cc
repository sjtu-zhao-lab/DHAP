#include <iostream>
#include <fstream>
#include <filesystem>
#include <random>
#include <regex>

#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/compute/api.h>

#include "runtime/metadata.h"
#include <nlohmann/json.hpp>

size_t AsInt(std::variant<size_t, std::string> intOrStr) {
	if (std::holds_alternative<size_t>(intOrStr)) {
		return std::get<size_t>(intOrStr);
	} else {
		return std::stoll(std::get<std::string>(intOrStr));
	}
}

std::shared_ptr<arrow::DataType> CreateDataType(const runtime::ColumnType& columnType) {
	if (columnType.base == "bool") return arrow::boolean();
	if (columnType.base == "int") {
		switch (AsInt(columnType.modifiers.at(0))) {
			case 8: return arrow::int8();
			case 16: return arrow::int16();
			case 32: return arrow::int32();
			case 64: return arrow::int64();
		}
	}
	if (columnType.base == "float") {
		switch (AsInt(columnType.modifiers.at(0))) {
			case 16: return arrow::float16();
			case 32: return arrow::float32();
			case 64: return arrow::float64();
		}
	}
	if (columnType.base == "date") {
		return std::get<std::string>(columnType.modifiers.at(0)) == "day" ?
			arrow::date32() :
			arrow::date64();
	}
	if (columnType.base == "string") return arrow::utf8();
	if (columnType.base == "char") return arrow::fixed_size_binary(AsInt(columnType.modifiers.at(0)));
	if (columnType.base == "decimal") return arrow::decimal(AsInt(columnType.modifiers.at(0)), AsInt(columnType.modifiers.at(1)));
	throw std::runtime_error("unsupported type");
}

std::shared_ptr<arrow::Schema> CreateSchema(std::shared_ptr<runtime::TableMetaData> metaData) 
{
	arrow::FieldVector fields;
	for (auto c : metaData->getOrderedColumns()) {
		auto& columnMetaData = metaData->getColumnMetaData(c);
		fields.push_back(std::make_shared<arrow::Field>(c, CreateDataType(columnMetaData->getColumnType())));
	}
	return std::make_shared<arrow::Schema>(fields);
}

arrow::Status DumpTable(std::shared_ptr<arrow::Table> table, std::string table_path)
{
	ARROW_ASSIGN_OR_RAISE(auto out_file, arrow::io::FileOutputStream::Open(table_path));
	ARROW_ASSIGN_OR_RAISE(auto batch_writer, arrow::ipc::MakeFileWriter(out_file, table->schema()));
	ARROW_RETURN_NOT_OK(batch_writer->WriteTable(*table));
	ARROW_RETURN_NOT_OK(batch_writer->Close());
	ARROW_RETURN_NOT_OK(out_file->Close());
	return arrow::Status::OK();
}

arrow::Status GenerateSample(std::shared_ptr<arrow::Table> table, std::string sample_path)
{
	if (std::getenv("DHAP_NOT_SAMPLE")) {
		return arrow::Status::OK();
	}
	std::cout << "generating samples" << std::endl;
	const int sample_size = table->num_rows()>1000? 1000 : table->num_rows();
	arrow::Int64Builder indice_builder;
	std::mt19937 mt(114);
	std::uniform_int_distribution<int64_t> dist(0, table->num_rows()-1);

	for (int i = 0; i < sample_size; i++) {
		ARROW_RETURN_NOT_OK(indice_builder.Append(dist(mt)));
	}
	ARROW_ASSIGN_OR_RAISE(auto indices, indice_builder.Finish());
	ARROW_ASSIGN_OR_RAISE(auto sample, arrow::compute::Take(table, *indices));
	ARROW_RETURN_NOT_OK(DumpTable(sample.table(), sample_path));
	std::cout << "generating samples fin" << std::endl;

	return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Table>> ConvertFloatToInt(std::shared_ptr<arrow::Table> table) {
  arrow::FieldVector new_fields;
  std::vector<std::shared_ptr<arrow::ChunkedArray>> new_columns;

  // Loop through columns in the table
  for (int i = 0; i < table->num_columns(); ++i) {
    auto column = table->column(i);
    auto field = table->schema()->field(i);
    auto data_type = field->type();

    // If column is of type float, cast to int
    if (data_type->id() == arrow::Type::FLOAT || data_type->id() == arrow::Type::DOUBLE) {
      std::cout << "Casting float column '" << field->name() << "' to int." << std::endl;

      arrow::compute::CastOptions options;
      options.to_type = arrow::int32();
			options.allow_float_truncate = true;

      ARROW_ASSIGN_OR_RAISE(
        auto int_column,
        arrow::compute::Cast(column, options)
      );

      new_columns.push_back(int_column.chunked_array());
      new_fields.push_back(arrow::field(field->name(), arrow::int32()));
    } else {
      // If it's not a float column, just add it as is
      new_columns.push_back(column);
      new_fields.push_back(field);
    }
  }
	auto new_schema = std::make_shared<arrow::Schema>(new_fields);
	return arrow::Table::Make(new_schema, new_columns);
}

arrow::Status ConvertCSV(const std::string json, const std::string filepath, const std::string db_path,
												 int block_size, std::unordered_map<std::string, int>& table_num_split,
												 std::unordered_map<std::string, int>& table_num_batches,
												 std::unordered_map<std::string, std::shared_ptr<runtime::TableMetaData>>& table_metadata,
												 std::shared_ptr<arrow::io::FileOutputStream>& out_file,
												 std::shared_ptr<arrow::ipc::RecordBatchWriter>& batch_writer)
{
	size_t last_slash = filepath.find_last_of('/');
	auto filename = filepath.substr(last_slash+1);

	std::string table_name;
	int cur_split = -1, num_split = -1;
	std::regex pattern("(.*)-(\\d+).tbl");
  std::smatch match;
	if (std::regex_search(filename, match, pattern)) {
		table_name = match[1].str();
		cur_split = std::stoi(match[2].str());
		num_split = table_num_split.at(table_name);
	}
	if (cur_split == -1) {
		size_t last_dot = filename.find_last_of('.');
		table_name = filename.substr(0, last_dot);
		assert(filename.substr(last_dot) == ".tbl");
	}
	std::cout << table_name << std::endl;

	const std::string out_filename = db_path + table_name + ".arrow";
	
	auto metadata = runtime::TableMetaData::create(json, table_name, std::shared_ptr<arrow::RecordBatch>());
	if (metadata->getOrderedColumns().size() == 0) {
		std::cout << table_name << " does not exist in metadata, skip " << std::endl;
		return arrow::Status::OK();
	}
	auto schema = CreateSchema(metadata);
	// std::cout << schema->ToString() << std::endl;

	ARROW_ASSIGN_OR_RAISE(auto file, arrow::io::ReadableFile::Open(filepath));
	
	auto read_opt = arrow::csv::ReadOptions::Defaults();
	// That will determine the size of output record batches (as no limit in `WriteTable`)
	// For SSB (100B/row), 1<<20 (1MB) will make record batches with 10k rows
	// read_opt.block_size = 1 << 28;
	// read_opt.block_size = 1 << 30;			// 10M rows per RB
	// read_opt.block_size = 1 << 26;			// 10M rows per RB
	read_opt.block_size = (1 << block_size) - 1;

	auto parse_opt = arrow::csv::ParseOptions::Defaults();
	parse_opt.delimiter = '|'; 
	auto cvt_opt = arrow::csv::ConvertOptions::Defaults();
	cvt_opt.null_values.push_back("");
	cvt_opt.strings_can_be_null = true;
	for (auto f : schema->fields()) {
		read_opt.column_names.push_back(f->name());
		cvt_opt.column_types.insert({f->name(), f->type()});
	}
	ARROW_ASSIGN_OR_RAISE(
		auto reader,
		arrow::csv::TableReader::Make(
			arrow::io::default_io_context(), file, read_opt, parse_opt, cvt_opt)
	);

	ARROW_ASSIGN_OR_RAISE(auto table, reader->Read());
	ARROW_ASSIGN_OR_RAISE(table, ConvertFloatToInt(table));

	arrow::TableBatchReader batch_reader(table);
	ARROW_ASSIGN_OR_RAISE(auto batches, batch_reader.ToRecordBatches());
	int num_b = batches.size();
	std::cout << "Got a table with " << table->num_rows() << " and " << num_b << " batches" << std::endl;

	if (table_metadata.contains(table_name)) {
		table_num_batches[table_name] += num_b;
		size_t new_rows = table_metadata[table_name]->getNumRows() + table->num_rows();
		table_metadata[table_name]->setNumRows(new_rows);
	}
	else {
		table_num_batches[table_name] = num_b;
		metadata->setNumRows(table->num_rows());
		table_metadata[table_name] = metadata;
	}

	if (cur_split == -1) {
		ARROW_RETURN_NOT_OK(DumpTable(table, out_filename));
	}
	else {
		if (cur_split == 0) {
			ARROW_ASSIGN_OR_RAISE(out_file, arrow::io::FileOutputStream::Open(out_filename, true));	
			ARROW_ASSIGN_OR_RAISE(batch_writer, arrow::ipc::MakeFileWriter(out_file, table->schema()));
		}
		assert(out_file && batch_writer);
		for (auto batch : batches) {
			ARROW_RETURN_NOT_OK(batch_writer->WriteRecordBatch(*batch));
		}
		if (cur_split+1 == num_split) {
			ARROW_RETURN_NOT_OK(batch_writer->Close());
			ARROW_RETURN_NOT_OK(out_file->Close());
		}
	}

	if (cur_split == -1) {
		ARROW_RETURN_NOT_OK(GenerateSample(table, out_filename+".sample"));
	}
	else if (cur_split == 0) {
		ARROW_ASSIGN_OR_RAISE(auto table0, arrow::Table::FromRecordBatches({batches[0]}));
		ARROW_RETURN_NOT_OK(GenerateSample(table0, out_filename+".sample"));
	}

	return arrow::Status::OK();
}

void WriteMetaData(std::string filename, std::unordered_map<std::string, std::shared_ptr<runtime::TableMetaData>>& table_metadata)
{
	std::ofstream ostream(filename);
	ostream << "{ \"tables\": {";
	bool first = true;
	for (auto t : table_metadata) {
		if (first) {
			first = false;
		} else {
			ostream << ",";
		}
		ostream << "\"" << t.first << "\":" << t.second->serialize(false);
	}
	ostream << "} }";
}

void UpdateMetaData(std::string filename, std::unordered_map<std::string, int>& num_batches)
{
	nlohmann::json metadata;
	std::ifstream md_file(filename);
	md_file >> metadata;
	md_file.close();
	for (auto kv : num_batches) {
		std::string table = kv.first;
		int num_b = kv.second;
		metadata["tables"][table]["num_batches"] = num_b;
		for (auto& col : metadata["tables"][table]["columns"]) {
			if (col["type"]["base"] == "float") {
				col["type"]["base"] = "int";
			}
		}
	}
	std::ofstream out_file(filename);
	out_file << std::setw(2) << metadata;
}

// Usage: ./csv_converter <source_dataset_name> <target_dataset_name> [20] 
// The block size will be 1 << 20 (2**20)
// Before running, make sure `mkdir <target_dataset_name>` and copy an
// metadata.json in it
int main(int argc, char** argv)
{
	// Load metadata generated by LingoDB (CREATE)
	std::string src_dataset = argv[1];
	std::string tgt_dataset = argv[2];
	std::string source_path = src_dataset + "/";
	std::string database_path = tgt_dataset + "/";
	int block_size = 31;
	if (argc > 3) {
		block_size = std::stoi(argv[3]);
	}

	std::string metadata_path = database_path + "metadata.json";
	std::string json;
	std::ifstream t(metadata_path);
	json = std::string((std::istreambuf_iterator<char>(t)),
											std::istreambuf_iterator<char>());
	std::unordered_map<std::string, int> table_num_batches;
	std::unordered_map<std::string, std::shared_ptr<runtime::TableMetaData>> table_metadata;

	std::vector<std::string> sorted_path;
	std::unordered_map<std::string, int> table_num_split;
	for (const auto& p : std::filesystem::directory_iterator(source_path)) {
    std::string path = p.path();
		if (path.find("metadata") != std::string::npos) {
			continue;
		}
		sorted_path.push_back(path);
		size_t last_slash = path.find_last_of('/');
		auto filename = path.substr(last_slash+1);

		std::regex pattern("(.*)-(\\d+).tbl");
		std::smatch match;
		if (std::regex_search(filename, match, pattern)) {
			auto table_name = match[1].str();
			int cur_split = std::stoi(match[2].str());
			if (!table_num_split.contains(table_name) || cur_split+1 > table_num_split.at(table_name)) {
				table_num_split[table_name] = cur_split + 1;
			}
		}
	}
	std::sort(sorted_path.begin(), sorted_path.end());

	std::shared_ptr<arrow::io::FileOutputStream> out_file = nullptr;
	std::shared_ptr<arrow::ipc::RecordBatchWriter> batch_writer = nullptr;
	for (const auto& path : sorted_path) {
		std::cout << path << std::endl;
		size_t last_slash = path.find_last_of('/');
		auto filename = path.substr(last_slash+1);

		auto status = ConvertCSV(json, path, database_path, block_size, 
														 table_num_split, table_num_batches, table_metadata,
														 out_file, batch_writer);
		if (!status.ok()) {
			std::cout << status << std::endl;
			return EXIT_FAILURE;
		}
	}

	std::string output_metadata_path = database_path + "/metadata.json";
	WriteMetaData(output_metadata_path, table_metadata);
	UpdateMetaData(output_metadata_path, table_num_batches);

	return EXIT_SUCCESS;
}