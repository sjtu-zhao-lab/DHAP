#include "mlir/Conversion/LoopToGDSA/CUDAGenerator.h"
#include "mlir/Dialect/RelAlg/IR/util.h"

namespace mlir {
namespace gdsa {

CUDAGenerator::CUDAGenerator(std::string subq_id_str, mlir::MLIRContext& mlir_ctxt,
														std::string file_name, int max_stage)
	: subq_id_str_(subq_id_str), mlir_ctxt_(mlir_ctxt), file_name_(file_name), max_stage_(max_stage)
{
	attr_manager = mlir_ctxt_.getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
	if (subq_id_str == "0") {
		knl_content = 	
			"#include <cooperative_groups.h>\n"
			"#include <thrust/iterator/discard_iterator.h>\n"
			"#include \"cuco/static_multimap.cuh\"\n"
			"#include \"cuco/static_map.cuh\"\n"
			"#include \"knl_util.cuh\"\n"
			"using pair_type = cuco::pair<hash_value_type, size_type>;\n"
			"namespace cg = cooperative_groups;\n\n";
	}
}

CUDAGenerator::~CUDAGenerator()
{
	const auto mode = (subq_id_str_ == "0") ? std::ios::out : std::ios::app;
	std::ofstream knl_file(file_name_, mode);
	knl_file << knl_content << std::endl;
}

void CUDAGenerator::genGet(GetElement get_op)
{
	std::string indent(space_indent, ' ');
	auto get_col = mlir::relalg::getColumnFromAttr(get_op.col());
	auto [table, col_name] = attr_manager.getName(get_col);
	std::string idx_name = value2name[get_op.idx()];
	std::string res_name = col_name + "_at_" + idx_name;
	RegularName(res_name);
	std::string batches_size = "batches_size_" + table;
	std::string num_batches = "num_batches_" + table;
	std::string get_params = ListToCommaSep({idx_name, col_name, batches_size, num_batches});
	knl_content += indent + "int " + res_name + " = get_from_batches" + get_params + ";\n";
	value2name[get_op.elem()] = res_name;
}

void CUDAGenerator::genEq(Equal eq_op)
{
	std::string indent(space_indent, ' ');
	std::string left_name = value2name[eq_op.left()];
	mlir::Attribute right_attr = eq_op.right();
	std::string right;
	if (auto str_attr = right_attr.dyn_cast<mlir::StringAttr>()) {
		std::string right_str = str_attr.getValue().str();
		int right_hash = str_hash(right_str);
		right = std::to_string(right_hash);
	}
	else if (auto int_attr = right_attr.dyn_cast<mlir::IntegerAttr>()) {
		int right_int = int_attr.getInt();
		right = std::to_string(right_int);
	}
	std::string res_name = left_name + "_eq";
	knl_content += indent + "bool " + res_name + " = (" + left_name + " == " + right + ");\n";
	value2name[eq_op.res()] = res_name;
}

void CUDAGenerator::genIf(If if_op)
{
	std::string indent(space_indent, ' ');
	std::string cond_name = value2name[if_op.cond()];
	knl_content += indent + "if (" + cond_name + ") {\n";
	space_indent += 2;
	for (mlir::Operation& in_if_op : if_op.getBody()->getOperations()) {
		Generate(&in_if_op);
	}
	space_indent -= 2;
	knl_content += indent + "}\n";
	bool update_num_groups = 	if_op.getOperation()->hasAttr("update_num_groups");
	if (update_num_groups) {
		knl_content += indent + "else { atomicAdd(d_num_groups, 1); }\n";
	}
}

void CUDAGenerator::genInsertMap(InsertMap insert_op)
{
	std::string indent(space_indent, ' ');
	std::string to_ins_name = value2name[insert_op.pair()];
	if (insert_op.map().getType() == MultiMapType::get(&mlir_ctxt_)) { // multimap (join)
		knl_content += indent + "ht.insert" + ListToCommaSep({"curr_cg", to_ins_name}) + ";\n";
	}
	else if (insert_op.map().getType() == MapType::get(&mlir_ctxt_)) {	// aggregation
		knl_content += indent + "groupby_ht_mv.insert" 
			+ ListToCommaSep({"curr_cg", to_ins_name, "groupby_key_hash", "groupby_key_eq"}) + ";\n";
	}
}

void CUDAGenerator::genMakePair(MakePair makep_op)
{
	std::string indent(space_indent, ' ');
	std::string key_name = value2name[makep_op.key()]; 
	std::string val_name = value2name[makep_op.val()];
	std::string pair_name = key_name + "_" + val_name;
	knl_content += indent + "auto " + pair_name + " = cuco::make_pair" 
												+ ListToCommaSep({key_name, val_name}) + ";\n";
	value2name[makep_op.pair()] = pair_name;
}

void CUDAGenerator::genMapCount(MapCount mapc_op)
{
	std::string indent(space_indent, ' ');
	auto probe_key_col = mlir::relalg::getOpAttrCols(mapc_op, "keys");
	assert(probe_key_col.size() == 1);
	mlir::relalg::Column* build_key_col = probe2build_key[probe_key_col[0]];
	auto [build_table, build_key] = attr_manager.getName(build_key_col);
	auto [_, probe_key] = attr_manager.getName(probe_key_col[0]);
	std::string ht = "ht_" + build_table;
	std::string pair = value2name[mapc_op.pair()];
	std::string pair_eq = build_key + "_eq_" + probe_key;
	std::string pair_cnt_params = ListToCommaSep({"curr_cg", pair, pair_eq});
	std::string count_name = probe_key + "_probe_cnt";
	knl_content += indent + "int " + count_name
							+  " = " + ht + ".pair_count" + pair_cnt_params + ";\n";
	value2name[mapc_op.count()] = count_name;
}

void CUDAGenerator::genMapFind(MapFind mapf_op)
{
	std::string indent(space_indent, ' ');
	if (mapf_op.map().getType() == MultiMapType::get(&mlir_ctxt_)) {	// join
		auto probe_key_col = mlir::relalg::getOpAttrCols(mapf_op, "probe_key");
		mlir::relalg::Column* build_key_col = probe2build_key[probe_key_col[0]];
		auto [build_table, build_key] = attr_manager.getName(build_key_col);
		auto [_, probe_key] = attr_manager.getName(probe_key_col[0]);
		std::string ht = "ht_" + build_table;
		std::string pair = value2name[mapf_op.pair()];
		std::string pair_eq = build_key + "_eq_" + probe_key;
		const std::string discard = "thrust::make_discard_iterator()";
		std::string find_build_idx = build_table + "_join_idx";
		std::string sm_decl = "  __shared__ uint32_t " + find_build_idx + "[num_probing_cgs][buffer_size];\n";
		knl_content.insert(before_while_pos, sm_decl);
		// knl_content += indent + "uint32_t " + find_build_idx + ";\n";
		std::string pair_retrieve_params = 
			ListToCommaSep({"curr_cg", pair, discard, discard, discard, 
											find_build_idx+"[probing_cg_id]", pair_eq});
		knl_content += indent + ht + ".pair_retrieve" + pair_retrieve_params + ";\n";
		// Define a ref to avoid `[]` in variable names
		std::string find_build_idx_ref = find_build_idx+"_ref";
		knl_content += indent + "uint32_t& " + find_build_idx_ref
								+ " = " + find_build_idx+"[probing_cg_id][0]" + ";\n";
		value2name[mapf_op.res()] = find_build_idx_ref;
	}
	else if (mapf_op.map().getType() == MapType::get(&mlir_ctxt_)) {	// aggregation
		std::string pair = value2name[mapf_op.pair()];
		std::string find_params = ListToCommaSep({"curr_cg", pair, "groupby_key_hash", "groupby_key_eq"});
		knl_content += indent + "const auto aggr_find_res = groupby_ht_v.find" + find_params + ";\n";
		value2name[mapf_op.res()] = "aggr_find_res";
	}
}

void CUDAGenerator::genIncrease(Increase inc_op)
{
	std::string indent(space_indent, ' ');
	knl_content += indent + "idx += loop_stride;\n";
}

void CUDAGenerator::genAnyThread(AnyThread any_op)
{
	std::string indent(space_indent, ' ');
	std::string any_input = value2name[any_op.input()];
	std::string any_out = "any_" + any_input;
	knl_content += indent + "int " + any_out + " = curr_cg.any(" + any_input + ");\n";
	value2name[any_op.res()] = any_out;
}
 
void CUDAGenerator::genSingleThread(SingleThread single_op)
{
	std::string indent(space_indent, ' ');
	knl_content += indent + "if (curr_cg.thread_rank() == 0) {\n";
	space_indent += 2;
	for (mlir::Operation& in_single_op : single_op.getBody()->getOperations()) {
		Generate(&in_single_op);
	}
	space_indent -= 2;
	knl_content += indent + "}\n";
}

void CUDAGenerator::genMakeAggrKey(MakeAggrKey make_aggr_op)
{
	std::string indent(space_indent, ' ');
	std::vector<std::string> keys_name;
	for (mlir::Value key : make_aggr_op.inputs()) {
		keys_name.push_back(value2name[key]);
	}
	knl_content += indent + "groupby_ht_keyT aggr_key" + ListToCommaSep(keys_name, "{}") + ";\n";
	value2name[make_aggr_op.res()] = "aggr_key";
}

void CUDAGenerator::genGetSecond(GetSecond get_sec_op)
{
	std::string indent(space_indent, ' ');
	std::string pair = value2name[get_sec_op.pair()];
	knl_content += indent + "uint32_t " + pair+"_second = " + pair + "->second;\n";
	value2name[get_sec_op.sec()] = pair+"_second";
}

void CUDAGenerator::genNotEqual(NotEqual neq_op)
{
	std::string indent(space_indent, ' ');
	std::string left_name = value2name[neq_op.left()];
	std::string right_name;
	if (neq_op.right()) {
		right_name = value2name[neq_op.right()];
	}
	else {
		right_name = std::to_string(neq_op.right1().dyn_cast<mlir::IntegerAttr>().getInt());
	}
	std::string res_name = left_name + "_neq_" + right_name;
	knl_content += indent + "bool " + res_name + " = (" + left_name + " != " + right_name + ");\n";
	value2name[neq_op.res()] = res_name;
}

void CUDAGenerator::genAtomicAdd(AtomicAdd add_op)
{
	std::string indent(space_indent, ' ');
	mlir::relalg::Column* col = mlir::relalg::getColumnFromAttr(add_op.col());
	auto [table, col_name] = attr_manager.getName(col);
	std::string idx = value2name[add_op.idx()];
	std::string val = value2name[add_op.val()];
	std::string to_add_pos = col_name + "_at_" + idx + "_ptr";
	std::string get_ptr_params = ListToCommaSep({idx, col_name, "batches_size_"+table, "num_batches_"+table});
	knl_content += indent + "int* " + to_add_pos + " = get_ptr_from_batches" + get_ptr_params + ";\n";
	std::string atomic_add_params = ListToCommaSep({to_add_pos, val});
	knl_content += indent + "atomicAdd" + atomic_add_params + ";\n";
}

void CUDAGenerator::genMod(Mod mod_op)
{
	std::string indent(space_indent, ' ');
	std::string left = value2name[mod_op.left()];
	std::string right = mod_op.right().dyn_cast<mlir::StringAttr>().getValue().str();
	assert(right == "PARAM: num_partitions");
	std::string mod_res = left + "_mod";
	knl_content += indent + "int " + mod_res + " = " + left + " % " + "num_partitions" + ";\n";
	value2name[mod_op.res()] = mod_res;
}

void CUDAGenerator::genAtomicAdd1(AtomicAdd1 add1_op)
{
	std::string indent(space_indent, ' ');
	std::string col = add1_op.col().dyn_cast<mlir::StringAttr>().getValue().str();
	std::string idx = value2name[add1_op.idx()];
	assert(col == "PARAM: num_partition_res");
	col = "num_partition_res";
	std::string res = col + "_" + idx + "_add1";
	knl_content += indent + "uint32_t " + res + " = atomicAdd" + ListToCommaSep({col+"+"+idx, "1"}) + ";\n";
	if (add1_op.res()){
		value2name[add1_op.res()] = res;
	}
}

void CUDAGenerator::genGetGlobalIndex(GetGlobalIndex get_gidx_op)
{
	std::string indent(space_indent, ' ');
	std::string partition_id = value2name[get_gidx_op.p_id()];
	std::string idx_in_partition = value2name[get_gidx_op.idx_in_p()];
	knl_content += indent + "uint32_t global_idx = " + partition_id + "*partition_limit_size + "
							+  idx_in_partition + ";\n";
	value2name[get_gidx_op.global_idx()] = "global_idx";
}

void CUDAGenerator::genMaterialize(Materialize mater_op)
{
	std::string indent(space_indent, ' ');
	std::string idx = value2name[mater_op.pos()];
	for (mlir::Value res : mater_op.results()) {
		std::string res_name = value2name[res];
		mlir::Operation* def_op = res.getDefiningOp();
		auto get_op = llvm::dyn_cast<GetElement>(def_op);
		assert(get_op && "materialzied results must be from `get`");
		mlir::relalg::Column* col = mlir::relalg::getColumnFromAttr(get_op.col());
		auto [_, col_name] = attr_manager.getName(col);
		std::string res_col = col_name + "_res";
		knl_content += indent + res_col+"["+idx+"] = " + res_name + ";\n";
	}
}

void CUDAGenerator::genConst(Const const_op)
{
	std::string indent(space_indent, ' ');
	if (auto int_attr = const_op.v().dyn_cast<mlir::IntegerAttr>()) {
		int v = int_attr.getInt();
		auto name = "c"+std::to_string(v);
		knl_content += indent + "const int " + name + " = " + std::to_string(v) + ";\n";
		value2name[const_op.res()] = name;
	}
	else {
		assert(false);
	}
}

void CUDAGenerator::genBinary(mlir::Operation* bin_op)
{
	std::string indent(space_indent, ' ');
	std::string aop, sop;
	if (mlir::isa<Add>(bin_op)) {
		aop = "+";
		sop = "ADD";
	}
	else if (mlir::isa<Sub>(bin_op)) {
		aop = "-";
		sop = "SUB";
	}
	else if (mlir::isa<Mul>(bin_op)) {
		aop = "*";
		sop = "MUL";
	}
	std::string left_name = value2name[bin_op->getOperand(0)];
	std::string right_name = value2name[bin_op->getOperand(1)];
	std::string res_name = left_name + sop + right_name;
	value2name[bin_op->getResult(0)] = res_name;

	knl_content += indent + "int " + res_name + " = " + left_name + aop + right_name + ";\n";
}

void CUDAGenerator::genStore(StoreElement store_op)
{
	std::string indent(space_indent, ' ');
	mlir::relalg::Column* col = mlir::relalg::getColumnFromAttr(store_op.col());
	auto [table, col_name] = attr_manager.getName(col);

  std::string idx_name = value2name[store_op.idx()];
	std::string store_pos = col_name + "_at_" + idx_name;
	std::string batches_size = "batches_size_" + table;
	std::string num_batches = "num_batches_" + table;
	std::string get_ptr_params = ListToCommaSep({idx_name, col_name, batches_size, num_batches});

	knl_content += indent + "int* " + store_pos + " = get_ptr_from_batches" + get_ptr_params + ";\n";
	knl_content += indent + "*"+store_pos + " = " + value2name[store_op.value()] + ";\n";
}

void CUDAGenerator::Generate(mlir::Operation* op)
{
	llvm::TypeSwitch<mlir::Operation*>(op)
		.Case<GetElement>([&](auto x) { genGet(x); })
		.Case<Equal>([&](auto x) { genEq(x); })
		.Case<If>([&](auto x) { genIf(x); })
		.Case<InsertMap>([&](auto x) { genInsertMap(x); })
		.Case<MakePair>([&](auto x) { genMakePair(x); })
		.Case<MapCount>([&](auto x) { genMapCount(x); })
		.Case<MapFind>([&](auto x) { genMapFind(x); })
		.Case<Increase>([&](auto x) { genIncrease(x); })
		.Case<AnyThread>([&](auto x) { genAnyThread(x); })
		.Case<SingleThread>([&](auto x) { genSingleThread(x); })
		.Case<MakeAggrKey>([&](auto x) { genMakeAggrKey(x); })
		.Case<GetSecond>([&](auto x) { genGetSecond(x); })
		.Case<NotEqual>([&](auto x) { genNotEqual(x); })
		.Case<AtomicAdd>([&](auto x) { genAtomicAdd(x); })
		.Case<Mod>([&](auto x) { genMod(x); })
		.Case<AtomicAdd1>([&](auto x) { genAtomicAdd1(x); })
		.Case<GetGlobalIndex>([&](auto x) { genGetGlobalIndex(x); })
		.Case<Materialize>([&](auto x) { genMaterialize(x); })
		.Case<Const>([&](auto x) { genConst(x); })
		.Case<Add, Sub, Mul>([&](auto x) { genBinary(x); })
		.Case<StoreElement>([&](auto x) { genStore(x); })
		.Default([](auto x) {});
		// .Default([](auto x) { assert(false && "should not unroll"); });
}

void CUDAGenerator::GenerateBuildKnl(mlir::Operation* op)
{
	auto while_op = llvm::dyn_cast<mlir::gdsa::While>(op);
	std::vector<mlir::relalg::Column*> build_key_col = mlir::relalg::getOpAttrCols(op, "build_keys");
	std::vector<mlir::relalg::Column*> build_key_plds = mlir::relalg::getOpAttrCols(op, "build_payloads");
	assert(build_key_col.size() == 1);
	build_key2plds[build_key_col[0]] = build_key_plds;
	auto [table, build_key] = attr_manager.getName(build_key_col[0]);
	std::string knl_name = "subq"+subq_id_str_+"_build_" + table;

	// Print the function name
	knl_content += knl_prefix + knl_name + "(\n";
	// Add parameters
	knl_content += "join_ht_mview ht,\n";
	knl_content +=  "int**" + build_key + ",\n";
	if (op->hasAttr("filter")) {
		std::vector<mlir::relalg::Column*> filter_col = mlir::relalg::getOpAttrCols(op, "filter");
		assert(filter_col.size() == 1);
		auto [_, filter] = attr_manager.getName(filter_col[0]);
		knl_content += "int** " + filter + ",\n";
	}
	knl_content += "uint32_t* batches_size_" + table + ",\n"
							+  "uint32_t total_size,\n"
							+  "int num_batches_" + table + "\n";
	// Start kernel body
	space_indent = 2;
	std::string indent(space_indent, ' ');
	knl_content += ")\n{\n" + knl_start;
	knl_content += indent + "while (idx < total_size) {\n";
	value2name[while_op.getBody()->getArgument(0)] = "idx";
	space_indent += 2;
	for (mlir::Operation& in_while_op : while_op.getBody()->getOperations()) {
		Generate(&in_while_op);
	}
	knl_content += indent + "}\n";
	knl_content += "}\n\n";
	value2name.clear();
}

void CUDAGenerator::PrepareProbeParams(mlir::gdsa::While while_op)
{
	mlir::Operation* op = while_op.getOperation();
	std::vector<mlir::relalg::Column*> build_key_cols = mlir::relalg::getOpAttrCols(op, "build_keys");
	std::vector<mlir::relalg::Column*> aggr_cols = mlir::relalg::getOpAttrCols(op, "aggr_col");
	// For build tables
	for (auto build_key_col : build_key_cols) {
		auto [build_table, build_key] = attr_manager.getName(build_key_col);
		// Hash table for different build tables
		knl_content += "join_ht_view ht_" + build_table +",\n";
		auto build_pld_cols = build_key2plds[build_key_col];
		// Build payloads: table (num_baches & batches size) + cols
		if (build_pld_cols.size() > 0) {
			knl_content += "int num_batches_" + build_table + ",\n" +
										 "uint32_t* batches_size_" + build_table + ",\n";
			for (auto build_pld_col : build_pld_cols) {
				// If aggr_col is in build payloads, rename to avoid duplication
				auto [_, build_pld] = attr_manager.getName(build_pld_col);
				if (std::find(aggr_cols.begin(), aggr_cols.end(), build_pld_col) != aggr_cols.end()) {
					knl_content += "int** " + build_pld + "0,\n";
				}
				else {
					knl_content += "int** " + build_pld + ",\n";
				}
			}
		}
	}
	// For probe table
	knl_content +=  "uint32_t total_size,\n";
	std::vector<mlir::relalg::Column*> probe_key_cols = mlir::relalg::getOpAttrCols(op, "probe_keys");
	assert(probe_key_cols.size() == build_key_cols.size());
	// Probe keys and eq
	bool print_probe_batch_param = false;
	for (int k = 0; k < probe_key_cols.size(); k++) {
		auto build_key_col = build_key_cols[k];
		auto probe_key_col = probe_key_cols[k];
		auto [build_table, build_key] = attr_manager.getName(build_key_col);
		auto [probe_table, probe_key] = attr_manager.getName(probe_key_col);
		if (!print_probe_batch_param) {
			knl_content += "int num_batches_" + probe_table + ",\n" 
									+  "uint32_t* batches_size_" + probe_table + ",\n";
			print_probe_batch_param = true;
		}
		knl_content += "int** " + probe_key + ", "
								+  "batch2_pair_equality " + build_key + "_eq_" + probe_key + ", \n";
		probe2build_key[probe_key_col] = build_key_col;
	}
	// Probe payloads
	mlir::relalg::Column* partition_col = NULL;
	if (op->hasAttr("partition_col")) {
		partition_col = mlir::relalg::getColumnFromAttr(op->getAttr("partition_col"));
	}
	std::vector<mlir::relalg::Column*> probe_pld_cols = mlir::relalg::getOpAttrCols(op, "probe_payloads");
	for (auto probe_pld_col : probe_pld_cols) {
		auto [probe_table, probe_pld] = attr_manager.getName(probe_pld_col);
		knl_content += "int** " + probe_pld + ",\n";
	}
	// Partition col at last
	int stage_id = op->getAttr("stage").dyn_cast<mlir::IntegerAttr>().getInt();
	if (partition_col || stage_id == max_stage_) {
		knl_content += "int num_partitions,\n";
		knl_content += "uint32_t partition_limit_size,\n";
		knl_content += "uint32_t* num_partition_res,\n";
	}
	// Output payloads
	if (aggr_cols.empty()) {
		for (auto probe_pld_col : probe_pld_cols) {
			auto [probe_table, probe_pld] = attr_manager.getName(probe_pld_col);
			knl_content += "int* " + probe_pld + "_res,\n";
		}
		for (auto build_key_col : build_key_cols) {
			auto build_pld_cols = build_key2plds[build_key_col];
			for (auto build_pld_col : build_pld_cols) {
				auto [_, build_pld] = attr_manager.getName(build_pld_col);
				knl_content += "int* " + build_pld + "_res,\n";
			}
		}
	}

	// For map
	std::vector<mlir::relalg::Column*> computed_cols = mlir::relalg::getOpAttrCols(op, "computed_cols");
	if (!computed_cols.empty()) {
		auto [map_table, aggr_col_name] = attr_manager.getName(computed_cols[0]);
		knl_content += "int num_batches_" + map_table + ",\n" 
								+  "uint32_t* batches_size_" + map_table + ",\n";
		for (auto computed_col : computed_cols) {
			auto [table, computed_col_name] = attr_manager.getName(computed_col);
			assert(table == map_table && "only 1 map tables");
			knl_content += "int** " + computed_col_name + ",\n";
		}
	}
	// For aggregation
	for (auto aggr_col : aggr_cols) {
		auto [_, aggr_col_name] = attr_manager.getName(aggr_col);
		// If the aggr_col is not in the mapped results, rename
		if (std::find(computed_cols.begin(), computed_cols.end(), aggr_col) == computed_cols.end()) {
			knl_content += "int** " + aggr_col_name + ",\n";
		}
		else {
			knl_content += "int** " + aggr_col_name + "_,\n";
		}
		std::string aggr_info	= "gb_ht_view groupby_ht_v, "
														"gb_ht_mview groupby_ht_mv, "
														"uint32_t* d_num_groups,\n"
														"batch_groupby_key_hasher groupby_key_hash, "
														"batch_groupby_key_equality groupby_key_eq\n";
		knl_content += aggr_info;
	}	
	// Remove the ending `,` if there is results
	if (knl_content.rfind(",\n") == knl_content.length() - 2) {
		knl_content.erase(knl_content.length() - 2, 1);
	}
}

void CUDAGenerator::GenerateProbeKnl(mlir::Operation* op)
{
	auto while_op = llvm::dyn_cast<mlir::gdsa::While>(op);
	int stage = op->getAttr("stage").dyn_cast_or_null<mlir::IntegerAttr>().getInt();
	std::string knl_name = "subq"+subq_id_str_+"_probe" + std::to_string(stage);
	knl_content += knl_prefix + knl_name + "(\n";
	// Add parameters
	PrepareProbeParams(while_op);
	// Start kernel body
	space_indent = 2;
	std::string indent(space_indent, ' ');
	knl_content += ")\n{\n" + knl_start + probe_knl_start;
	before_while_pos = knl_content.length();
	knl_content += indent + "while (idx < total_size) {\n";
	value2name[while_op.getBody()->getArgument(0)] = "idx";
	space_indent += 2;
	for (mlir::Operation& in_while_op : while_op.getBody()->getOperations()) {
		Generate(&in_while_op);
	}
	knl_content += indent + "}\n";
	knl_content += "}\n\n";
	value2name.clear();
	probe2build_key.clear();
}

} // namespace gdsa
} // namespace mlir