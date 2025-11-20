#include "mlir/Conversion/RelAlgToLoop/UnrollRelAlg.h"
#include "mlir/Dialect/RelAlg/IR/util.h"

using namespace mlir::relalg;

namespace mlir {
namespace relalg {

void GetRequiredColumns(mlir::Operation* op, mlir::Operation* consumer,
												std::unordered_map<mlir::Operation*, mlir::relalg::ColumnSet>& op_req_cols)
{
	// auto& attr_manager = op->getContext()->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
	if (consumer) {
		Operator op1 = op;
		assert(op1 && "Operation that has consumer must be Operator");
		assert(op_req_cols.find(consumer) != op_req_cols.end() && 
					 "Consumer must have been got");
		mlir::relalg::ColumnSet consumer_cs = op1.getAvailableColumns().intersect(op_req_cols[consumer]);
		// mlir::relalg::ColumnSet op_avail;
		// if (op->hasAttr("new_inp")) {
		// 	std::string build_table = op->getAttr("build_table").dyn_cast<mlir::StringAttr>().getValue().str();
		// 	std::string old_inp = op->getAttr("old_inp").dyn_cast<mlir::StringAttr>().getValue().str();
		// 	std::string new_inp = op->getAttr("new_inp").dyn_cast<mlir::StringAttr>().getValue().str();
		// 	for (const mlir::relalg::Column* x : op1.getAvailableColumns()) {
		// 		auto [table, col_name] = attr_manager.getName(x);
		// 		if (table == old_inp) {		// Probe cols from original probe table
		// 			auto new_ref = attr_manager.createRef(new_inp, col_name);
		// 			op_avail.insert(new_ref.getColumnPtr().get());
		// 		}
		// 		else if (table != build_table) {		// Probe cols from intermediates
		// 			auto new_ref = attr_manager.createRef(new_inp, col_name);
		// 			op_avail.insert(new_ref.getColumnPtr().get());
		// 		}
		// 		else {
		// 			op_avail.insert(x);
		// 		}
		// 	}
		// }
		// else {
		// 	op_avail = op1.getAvailableColumns();
		// }
		// mlir::relalg::ColumnSet consumer_cs = op_avail.intersectOnColName(op->getContext(), op_req_cols[consumer]);
		op_req_cols[op] = consumer_cs.insert(op1.getUsedColumns());

		for (auto child : op1.getChildren()) {
			GetRequiredColumns(child.getOperation(), op, op_req_cols);
		}
	}
	else {
		auto op1 = llvm::dyn_cast<mlir::relalg::MaterializeOp>(op);
		assert(op1 && "Only MaterializeOp has no consumer");
		op_req_cols[op] = ColumnSet::fromArrayAttr(op1.cols());
		mlir::Operation* child = op1.rel().getDefiningOp();
		GetRequiredColumns(child, op, op_req_cols);
	}
}

void Unroll(mlir::Operation* op, mlir::OpBuilder& builder,
						std::unordered_map<mlir::Operation*, mlir::relalg::ColumnSet>& op_req_cols)
{
	llvm::TypeSwitch<mlir::Operation*>(op)
		.Case<SelectionOp>([&](auto x) { UnrollSelection(x, builder, op_req_cols); })
		.Case<InnerJoinOp>([&](auto x) { UnrollJoin(x, builder, op_req_cols); })
		.Case<AggregationOp>([&](auto x) { UnrollAggr(x, builder, op_req_cols); })
		.Case<MapOp>([&](auto x) { UnrollMap(x, builder, op_req_cols); })
		.Default([](auto x) { assert(false && "should not unroll"); });
}

void UnrollSelection(SelectionOp sel_op, mlir::OpBuilder& builder,
											std::unordered_map<mlir::Operation*, mlir::relalg::ColumnSet>& op_req_cols)
{
	mlir::MLIRContext* mlir_ctxt = builder.getContext();
	mlir::Type ts_type = mlir::relalg::TupleStreamType::get(mlir_ctxt);
	
	mlir::Operation* src_def_op = sel_op.rel().getDefiningOp();
	assert(mlir::isa<mlir::relalg::BaseTableOp>(src_def_op) && "only select from base table");
	auto src_base_table = llvm::dyn_cast<mlir::relalg::BaseTableOp>(src_def_op);
	std::string sel_src_name(src_base_table.table_identifier().begin(), src_base_table.table_identifier().end());
	auto sel_loop = builder.create<mlir::loop::ForOp>(
		sel_op.getLoc(), ts_type, sel_op.rel(), op_req_cols[sel_op].asRefArrayAttr(builder.getContext()),
		sel_src_name, sel_src_name, 0
	);
	mlir::Value sel_loop_res = sel_loop.res();
	sel_op.result().replaceAllUsesWith(sel_loop_res);
	
	mlir::Block* sel_blk = new mlir::Block;
	sel_loop.region().push_back(sel_blk);
	mlir::OpBuilder builder1(sel_blk, sel_blk->end());
	
	mlir::Type arg_type = sel_op.predicate().front().getArgumentTypes().front();
	mlir::Type tuple_type = mlir::relalg::TupleType::get(builder1.getContext());
	assert(arg_type == tuple_type);
	mlir::BlockArgument blk_arg = sel_blk->addArgument(arg_type, sel_op.getLoc());
	
	std::unordered_map<mlir::Operation*, mlir::Value> old2new_res;
	mlir::relalg::ColumnSet filter_cols;
	for (mlir::Operation& op : sel_op.predicate().front()) {
		builder1.setInsertionPointToEnd(sel_blk);
		llvm::TypeSwitch<mlir::Operation*>(&op)
			.Case<mlir::relalg::GetColumnOp>([&](auto getcol_op) {
				CloneGetCol(builder1, getcol_op, blk_arg, old2new_res);
				filter_cols.insert(mlir::relalg::ColumnSet::from(getcol_op.attr()));
				sel_loop.getOperation()->setAttr("filter", filter_cols.asRefArrayAttr(mlir_ctxt));
			})
			.Case<mlir::db::ConstantOp>([&](auto const_op) {
				CloneConstOp(builder1, const_op, old2new_res);
			})
			.Case<mlir::db::CmpOp>([&](auto cmp_op) {
				mlir::Operation* left_def_op = cmp_op.left().getDefiningOp();
				mlir::Value new_left = old2new_res[left_def_op];
				mlir::Operation* right_def_op = cmp_op.right().getDefiningOp();
				mlir::Value new_right = old2new_res[right_def_op];
				old2new_res[cmp_op] = builder1.create<mlir::db::CmpOp>(
					sel_op.getLoc(), cmp_op.predicate(), new_left, new_right
				);
			})
			.Case<mlir::relalg::ReturnOp>([&](auto ret_op) {
				assert(ret_op.results().size() == 1 && "The return in selection must have 1 result");
				std::vector<mlir::Value> new_res;
				for (mlir::Value res : ret_op.results()) {
					mlir::Operation* res_def_op = res.getDefiningOp();
					new_res.push_back(old2new_res[res_def_op]);
				}
				// builder1.create<mlir::relalg::ReturnOp>(ret_op.getLoc(), new_res);
				auto sel_if = builder1.create<mlir::loop::IfOp>(ret_op.getLoc(), new_res[0]);
				mlir::Block* sel_if_blk = new mlir::Block;
				sel_if.region().push_back(sel_if_blk);
				OpBuilder builder2(sel_if.getBodyRegion());
				builder2.create<mlir::loop::YieldOp>(ret_op.getLoc(), blk_arg);
				// builder1.create<mlir::relalg::ReturnOp>(ret_op.getLoc());
			})
			.Default([](auto x) {
				assert(false && "not allowed op within selection");
			});
	}
	// builder1.create<mlir::loop::YieldOp>(sel_op.getLoc());
}

// #define DBG_UNROLL
void UnrollJoin(InnerJoinOp join_op, mlir::OpBuilder& builder,
								std::unordered_map<mlir::Operation*, mlir::relalg::ColumnSet>& op_req_cols)
{
	mlir::MLIRContext* mlir_ctxt = builder.getContext();
	auto& attr_manager = mlir_ctxt->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
	mlir::Type ts_type = mlir::relalg::TupleStreamType::get(mlir_ctxt);
	mlir::Type tuple_type = mlir::relalg::TupleType::get(mlir_ctxt);

	mlir::Value build_tbl = join_op.left();
	mlir::Value probe_tbl = join_op.right();
	bool shfl_build = false, shfl_probe = false;
	for (mlir::Operation* build_tbl_user : build_tbl.getUsers()) {
		if (llvm::dyn_cast<mlir::loop::Shuffle>(build_tbl_user)) {
			shfl_build = true;
			break;
		}
	}
	for (mlir::Operation* probe_tbl_user : probe_tbl.getUsers()) {
		if (llvm::dyn_cast<mlir::loop::Shuffle>(probe_tbl_user)) {
			shfl_probe = true;
			break;
		}
	}
	
	mlir::relalg::ColumnSet build_avail, probe_avail;
	std::string build_name, probe_name, join_id;
	auto build_tbl_src_op = build_tbl.getDefiningOp();
	auto probe_tbl_src_op = probe_tbl.getDefiningOp();
	// llvm::outs() << build_tbl.getDefiningOp()->getName() << "\n";
	// llvm::outs() << probe_tbl.getDefiningOp()->getName() << "\n";
	if (auto x = llvm::dyn_cast<mlir::loop::ForOp>(build_tbl_src_op)) {
		build_avail = mlir::relalg::ColumnSet::fromArrayAttr(x.req_cols());
		build_name = x.name_res();
	}
	else if (Operator x = build_tbl_src_op) {
		build_avail = x.getAvailableColumns();
		auto base_table = llvm::dyn_cast<mlir::relalg::BaseTableOp>(build_tbl_src_op);
		assert(base_table && "if not loop, must build from base table");
		build_name = base_table.table_identifier();
	}
	if (auto x = llvm::dyn_cast<mlir::loop::ForOp>(probe_tbl_src_op)) {
		probe_avail = mlir::relalg::ColumnSet::fromArrayAttr(x.req_cols());
		probe_name = x.name_res();
		join_id = std::to_string(std::stoi(probe_name.substr(probe_name.find("_")+1)) + 1);
	}
	else if (Operator x = probe_tbl_src_op) {
		probe_avail = x.getAvailableColumns();
		auto base_table = llvm::dyn_cast<mlir::relalg::BaseTableOp>(probe_tbl_src_op);
		assert(base_table && "if not loop, must probe from base table");
		probe_name = base_table.table_identifier();
		join_id = "1";
	}

	// Get join keys
	auto [build_keys, probe_keys, keyTypes, build_key_attrs, can_save] 
		= mlir::relalg::HashJoinUtils::analyzeHJPred(&(join_op.predicate().front()), build_avail, probe_avail);
	auto build_req_cols = build_avail.intersect(op_req_cols[join_op]);
	auto build_payloads = build_avail.intersect(op_req_cols[join_op]);
	// auto build_req_cols = op_req_cols[join_op].intersectOnColName(mlir_ctxt, build_avail);
	// auto build_payloads = build_avail.intersect(op_req_cols[join_op]);
	for (mlir::relalg::ColumnSet& build_k : build_key_attrs) {
		build_payloads.remove(build_k);
	}
	auto probe_req_cols = probe_avail.intersect(op_req_cols[join_op]);
	// auto probe_req_cols = op_req_cols[join_op].intersectOnColName(mlir_ctxt, probe_avail);
	// Add the build payload of last join to the requested columns
	mlir::relalg::ColumnSet last_build_payloads, last_probe_payloads;
	probe_tbl_src_op->walk([&](mlir::Operation* x) {
		if (auto probe = llvm::dyn_cast<mlir::loop::ProbeHashTable>(x)) {
			auto last_build_payloads_attr = probe.ht().getDefiningOp()->getAttr("payloads").dyn_cast<mlir::ArrayAttr>();
			auto last_probe_payloads_attr = probe.getOperation()->getAttr("payloads").dyn_cast<mlir::ArrayAttr>();
			last_build_payloads = mlir::relalg::ColumnSet::fromArrayAttr(last_build_payloads_attr);
			last_probe_payloads = mlir::relalg::ColumnSet::fromArrayAttr(last_probe_payloads_attr);
		}
	});
	std::string origin_probe_name;
	for (const auto* probe_key_col : probe_keys) {
		auto [table, col] = attr_manager.getName(probe_key_col);
		origin_probe_name = table;
	}
	// Replace the table name with probe table
	probe_req_cols.insert(last_probe_payloads);
	for (const auto* last_build_pld : last_build_payloads) {
		auto [table, col] = attr_manager.getName(last_build_pld);
		auto last_build_pld_attr = attr_manager.createRef(origin_probe_name, col);
		last_build_pld_attr.getColumn().type = last_build_pld->type;
		probe_req_cols.insert(last_build_pld_attr.getColumnPtr().get());
	}
	auto probe_payloads = probe_req_cols;
	probe_payloads.remove(probe_keys);

	#ifdef DBG_UNROLL
	llvm::outs() << "All req cols: \n";
	op_req_cols[join_op].dump(mlir_ctxt);
	llvm::outs() << "\n";
	llvm::outs() << "Build keys: \n";
	build_keys.dump(mlir_ctxt);
	llvm::outs() << "\n";
	llvm::outs() << "Build payloads: \n";
	build_payloads.dump(mlir_ctxt);
	llvm::outs() << "\n";
	llvm::outs() << "Build avail: \n";
	build_avail.dump(mlir_ctxt);
	llvm::outs() << "\n";
	llvm::outs() << "Probe avail: \n";
	probe_avail.dump(mlir_ctxt);
	llvm::outs() << "\n";
	llvm::outs() << "Probe keys: \n";
	probe_keys.dump(mlir_ctxt);
	llvm::outs() << "\n";
	llvm::outs() << "Probe requset: \n";
	probe_req_cols.dump(mlir_ctxt);
	llvm::outs() << "\n";
	llvm::outs() << "Probe payloads: \n";
	probe_payloads.dump(mlir_ctxt);
	llvm::outs() << "\n";
	llvm::outs() << "Build attrs?\n";
	for (mlir::relalg::ColumnSet& cs : build_key_attrs) {
		cs.dump(mlir_ctxt);
		llvm::outs() << "\n";
	}
	llvm::outs() << "\n";
	#endif

	// auto& attr_manaeger = mlir_ctxt->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
	// for (const mlir::relalg::Column* c : op_req_cols[join_op]) {
	// 	auto [tbl_name, col_name] = attr_manaeger.getName(c);
	// 	llvm::outs() << tbl_name << " " << col_name << "\n";
	// }
	// llvm::outs() << "\n";

	// Lower the build of join hash table
	auto create_ht = builder.create<mlir::loop::CreateHashTable>(
		join_op.getLoc(), mlir::loop::JoinHashtableType::get(mlir_ctxt),
		build_keys.asRefArrayAttr(mlir_ctxt), build_payloads.asRefArrayAttr(mlir_ctxt)
	);
	std::string build_name_s = shfl_build? "_s" : "";
	auto build_ht_for = builder.create<mlir::loop::ForOp>(
		join_op.getLoc(), mlir::Type{}, build_tbl, build_req_cols.asRefArrayAttr(mlir_ctxt),
		build_name, "", 0
	);
	assert(join_op.getOperation()->hasAttr("stage"));
	build_ht_for.getOperation()->setAttr("stage", join_op.getOperation()->getAttr("stage"));
	build_ht_for.getOperation()->setAttr("build_keys", build_keys.asRefArrayAttr(mlir_ctxt));
	build_ht_for.getOperation()->setAttr("build_payloads", build_payloads.asRefArrayAttr(mlir_ctxt));
	mlir::Block* build_ht_blk = new mlir::Block;
	build_ht_for.region().push_back(build_ht_blk);
	mlir::OpBuilder builder1(build_ht_for.getBodyRegion());
	mlir::BlockArgument build_blk_arg = build_ht_blk->addArgument(tuple_type, join_op.getLoc());
	builder1.create<mlir::loop::UpdateHashTable>(
		join_op.getLoc(), create_ht.ht(), build_blk_arg,
		build_keys.asRefArrayAttr(mlir_ctxt), build_payloads.asRefArrayAttr(mlir_ctxt)
	);

	// Lowering the probe of join hash table
	std::string probe_name_s = shfl_probe? "_s" : "";
	auto probe_for = builder.create<mlir::loop::ForOp>(
		join_op.getLoc(), ts_type, probe_tbl, probe_req_cols.asRefArrayAttr(mlir_ctxt),
		probe_name, "join_"+join_id, 1
	);
	probe_for.getOperation()->setAttr("stage", join_op.getOperation()->getAttr("stage"));
	probe_for.getOperation()->setAttr("probe_keys", probe_keys.asRefArrayAttr(mlir_ctxt));
	probe_for.getOperation()->setAttr("probe_payloads", probe_payloads.asRefArrayAttr(mlir_ctxt));
	// Add build keys to efficiently get correct order of build
	probe_for.getOperation()->setAttr("build_keys", build_keys.asRefArrayAttr(mlir_ctxt));
	join_op.result().replaceAllUsesWith(probe_for.res());
	mlir::Block* probe_blk = new mlir::Block;
	probe_for.region().push_back(probe_blk);
	mlir::OpBuilder builder2(probe_for.getBodyRegion());
	mlir::BlockArgument probe_blk_arg = probe_blk->addArgument(tuple_type, join_op.getLoc());
	mlir::Value probe_res = builder2.create<mlir::loop::ProbeHashTable>(
		join_op.getLoc(), mlir::loop::ProbeResultType::get(mlir_ctxt), create_ht.ht(), probe_blk_arg,
		probe_keys.asRefArrayAttr(mlir_ctxt), probe_payloads.asRefArrayAttr(mlir_ctxt)
	);
	// If probe hit
	auto probe_if_hit = builder2.create<mlir::loop::IfOp>(join_op.getLoc(), probe_res);
	mlir::Block* probe_hit_blk = new mlir::Block;
	probe_if_hit.region().push_back(probe_hit_blk);
	mlir::OpBuilder builder3(probe_if_hit.getBodyRegion());
	builder3.create<mlir::loop::YieldOp>(join_op.getLoc(), probe_res);
}

void UnrollMap(MapOp map_op, mlir::OpBuilder& builder,
								std::unordered_map<mlir::Operation*, mlir::relalg::ColumnSet>& op_req_cols)
{
	mlir::MLIRContext* mlir_ctxt = builder.getContext();
	mlir::Type ts_type = mlir::relalg::TupleStreamType::get(mlir_ctxt);	
	mlir::Type tuple_type = mlir::relalg::TupleType::get(mlir_ctxt);	
	auto& attr_manager = mlir_ctxt->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();

	auto computed_cols = map_op.computed_cols();
	std::string res_tbl_name;
	std::vector<std::string> res_col_names;
	for (auto col_ref : computed_cols) {
		mlir::relalg::Column* col = col_ref.dyn_cast_or_null<ColumnDefAttr>().getColumnPtr().get();
		auto [table, col_name] = attr_manager.getName(col);
		res_tbl_name = table;
		res_col_names.push_back(col_name);
	}
	
	mlir::Operation* src_def_op = map_op.rel().getDefiningOp();
	assert(mlir::isa<mlir::loop::ForOp>(src_def_op) && "only map from loop result");
	auto map_loop = builder.create<mlir::loop::ForOp>(map_op.getLoc(), 
		ts_type, map_op.rel(), op_req_cols[map_op].asRefArrayAttr(builder.getContext()),
		src_def_op->getAttr("name_res").dyn_cast_or_null<mlir::StringAttr>(),
		builder.getStringAttr(res_tbl_name), 4
	);
	map_loop->setAttr("computed_cols", map_op.computed_cols());
	map_op.result().replaceAllUsesWith(map_loop.res());

	mlir::Block* map_blk = new mlir::Block;
	map_loop.region().push_back(map_blk);
	mlir::OpBuilder builder1(map_blk, map_blk->end());
	mlir::Type arg_type = map_op.predicate().front().getArgumentTypes().front();
	assert(arg_type == tuple_type);
	mlir::BlockArgument blk_arg = map_blk->addArgument(arg_type, map_op.getLoc());

	std::unordered_map<mlir::Operation*, mlir::Value> old2new_res;
	for (mlir::Operation& op : map_op.predicate().front()) {
		builder1.setInsertionPointToEnd(map_blk);
		llvm::TypeSwitch<mlir::Operation*>(&op)
			.Case<mlir::relalg::GetColumnOp>([&](auto getcol_op) {
				CloneGetCol(builder1, getcol_op, blk_arg, old2new_res);
			})
			.Case<mlir::db::ConstantOp>([&](auto const_op) {
				CloneConstOp(builder1, const_op, old2new_res);
			})
			.Case<mlir::db::SubOp>([&](auto sub_op) {
				CloneBinaryOp(builder1, sub_op, old2new_res);
			})
			.Case<mlir::db::MulOp>([&](auto mul_op) {
				CloneBinaryOp(builder1, mul_op, old2new_res);
			})
			.Case<mlir::relalg::ReturnOp>([&](auto ret_op) {
				assert(ret_op.results().size() == 1 && "The return in map must have 1 result");
				llvm::SmallVector<mlir::Value, 4> new_map_res;
				for (mlir::Value res : ret_op.results()) {
					new_map_res.push_back(old2new_res[res.getDefiningOp()]);
				}
				mlir::Value updated_tuple = builder1.create<mlir::loop::UpdateOp>(
					ret_op.getLoc(), tuple_type, blk_arg, new_map_res, map_op.computed_cols() 
				);
				builder1.create<mlir::loop::YieldOp>(ret_op.getLoc(), updated_tuple);
			})
			.Default([](auto x) {
				assert(false && "not allowed op within map");
			});
	}
}

void UnrollAggr(AggregationOp aggr_op, mlir::OpBuilder& builder,
								std::unordered_map<mlir::Operation*, mlir::relalg::ColumnSet>& op_req_cols)
{
	mlir::MLIRContext* mlir_ctxt = builder.getContext();
	mlir::Type ts_type = mlir::relalg::TupleStreamType::get(mlir_ctxt);	
	mlir::Type tuple_type = mlir::relalg::TupleType::get(mlir_ctxt);	

	auto aggr_tbl = aggr_op.rel();
	auto group_by_cols = mlir::relalg::ColumnSet::fromArrayAttr(aggr_op.group_by_cols());
	auto group_by_cols1 = mlir::relalg::ColumnSet::fromArrayAttr(aggr_op.group_by_cols());
	auto computed_cols = mlir::relalg::ColumnSet::fromArrayAttr(aggr_op.computed_cols());
	auto input_cols = op_req_cols[aggr_op];
	input_cols.remove(group_by_cols);
	input_cols.remove(computed_cols);

	auto create_ht = builder.create<mlir::loop::CreateHashTable>(
		aggr_op.getLoc(), mlir::loop::AggregationHashtableType::get(mlir_ctxt),
		group_by_cols.asRefArrayAttr(mlir_ctxt), input_cols.asRefArrayAttr(mlir_ctxt)
	);

	mlir::Operation* aggr_tbl_src_op = aggr_tbl.getDefiningOp();
	assert(mlir::isa<mlir::loop::ForOp>(aggr_tbl_src_op));
	mlir::relalg::ColumnSet aggr_ht_for_req_cols = group_by_cols1.insert(input_cols);
	auto build_aggr_ht_for = builder.create<mlir::loop::ForOp>(
		aggr_op.getLoc(), mlir::Type{}, aggr_op.rel(), aggr_ht_for_req_cols.asRefArrayAttr(mlir_ctxt),
		llvm::dyn_cast<mlir::loop::ForOp>(aggr_tbl_src_op).name_res(), "", 2
	);
	build_aggr_ht_for.getOperation()->setAttr("groupby_keys", group_by_cols.asRefArrayAttr(mlir_ctxt));
	build_aggr_ht_for.getOperation()->setAttr("aggr_col", input_cols.asRefArrayAttr(mlir_ctxt));
	
	// op_req_cols[aggr_op].dump(mlir_ctxt);
	// llvm::outs() << "\n";
	// llvm::outs() << aggr_op.group_by_cols().getType() << "\n";
	// for (auto attr : aggr_op.group_by_cols()) {
	// 	llvm::outs() << attr.getType() << " " << attr << "\n";
	// }

	mlir::Block* build_aggr_ht_blk = new mlir::Block;
	build_aggr_ht_for.region().push_back(build_aggr_ht_blk);
	mlir::OpBuilder builder1(build_aggr_ht_for.getBodyRegion());
	mlir::BlockArgument build_blk_arg = build_aggr_ht_blk->addArgument(tuple_type, aggr_op.getLoc());
	builder1.create<mlir::loop::UpdateHashTable>(
		aggr_op.getLoc(), create_ht.ht(), build_blk_arg,
		group_by_cols.asRefArrayAttr(mlir_ctxt), input_cols.asRefArrayAttr(mlir_ctxt)
	);
	// builder1.create<mlir::loop::YieldOp>(aggr_op.getLoc());

	auto retrieve_aggr_ht_for = builder.create<mlir::loop::ForOp>(
		aggr_op.getLoc(), ts_type, create_ht.ht(), aggr_ht_for_req_cols.asRefArrayAttr(mlir_ctxt),
		"", "", 3
	);
	aggr_op.result().replaceAllUsesWith(retrieve_aggr_ht_for.res());
	mlir::Block* retrieve_aggr_ht_blk = new mlir::Block;
	retrieve_aggr_ht_for.region().push_back(retrieve_aggr_ht_blk);
	mlir::OpBuilder builder2(retrieve_aggr_ht_for.getBodyRegion());
	mlir::BlockArgument retrieve_blk_arg = retrieve_aggr_ht_blk->addArgument(tuple_type, aggr_op.getLoc());
	builder2.create<mlir::loop::YieldOp>(aggr_op.getLoc(), retrieve_blk_arg);

}

} // end namespace relalg
} // end namespace mlir 