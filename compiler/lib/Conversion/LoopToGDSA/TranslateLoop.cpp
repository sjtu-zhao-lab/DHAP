#include "mlir/Conversion/LoopToGDSA/TranslateLoop.h"
#include "mlir/Dialect/RelAlg/IR/util.h"


using namespace mlir::loop;

namespace mlir {
namespace loop {
	
void LoopTranslator::LoopToWhile(mlir::Operation* op, mlir::OpBuilder& builder)
{
	mlir::MLIRContext* mlir_ctxt = builder.getContext();
	mlir::Type row_idx_type = mlir::gdsa::RowIndexType::get(mlir_ctxt);
	
	auto for_op = llvm::dyn_cast<mlir::loop::ForOp>(op);
	auto while_op = builder.create<mlir::gdsa::While>(for_op.getLoc(), "PARAM: size", for_op.req_cols());
	std::vector<std::string> to_copy_attrs = {
		"type", "filter", "stage", "build_keys", "build_payloads", "probe_keys", "probe_payloads", 
		"partition_col", "aggr_col", "groupby_keys_build", "groupby_keys_probe", "computed_cols"
	};
	copyAttrIfHas(op, while_op.getOperation(), to_copy_attrs);

	mlir::Block* while_blk = new mlir::Block;
	while_op.region().push_back(while_blk);
	mlir::OpBuilder while_builder(while_op.getBodyRegion());
	mlir::BlockArgument while_idx = while_blk->addArgument(row_idx_type, for_op.getLoc());

	for (mlir::Operation& op1 : for_op.region().front()) {
		if (auto get_col_op = llvm::dyn_cast<mlir::relalg::GetColumnOp>(&op1)) {
			mlir::Value elem = while_builder.create<mlir::gdsa::GetElement>(
				get_col_op.getLoc(), mlir::IntegerType::get(mlir_ctxt, 32), get_col_op.attr(), while_idx
			);
			// get_col_op.res().replaceAllUsesWith(elem);
			old2new_res_[&op1] = elem;
		}
		else if (auto cmp_op = llvm::dyn_cast<mlir::db::CmpOp>(&op1)) {
			CmpToGDSA(cmp_op, while_builder);
		}
		else if (auto if_op = llvm::dyn_cast<mlir::loop::IfOp>(&op1)) {
			NestedIfToGDSA(if_op, while_builder, while_idx);
		}
		else if (auto update_op = llvm::dyn_cast<mlir::loop::UpdateHashTable>(&op1)) {
			auto ht_type = update_op.ht().getType();
			if (ht_type == mlir::loop::JoinHashtableType::get(mlir_ctxt)) {
				UpdateJoinHTToGDSA(update_op, while_builder, while_idx);
			}
			else {
				assert(ht_type == mlir::loop::AggregationHashtableType::get(mlir_ctxt));	
				UpdateAggrHTToGDSA(update_op, while_builder, while_idx);
			}
		}
		else if (auto probe_op = llvm::dyn_cast<mlir::loop::ProbeHashTable>(&op1)) {
			ProbeHTToGDSA(probe_op, while_builder, while_idx);
		}
		else if (auto yield = llvm::dyn_cast<mlir::loop::YieldOp>(&op1)) {
			YiledToGDSA(yield, while_builder, while_idx);
		}
	}

	while_builder.create<mlir::gdsa::Increase>(for_op.getLoc(), while_idx);
}

void LoopTranslator::CmpToGDSA(mlir::db::CmpOp cmp_op, mlir::OpBuilder& builder) 
{
	// llvm::outs() << "getcol " << get_col_op.attr().name() << "\n";
	mlir::Value left;
	mlir::Attribute right;
	for (auto x : cmp_op.getOperation()->getOperands()) {
		mlir::Operation* x_def_op = x.getDefiningOp();
		if (auto const_op = llvm::dyn_cast<mlir::db::ConstantOp>(x_def_op)) {
			right = const_op.value();
		}
		else {
			assert(llvm::dyn_cast<mlir::relalg::GetColumnOp>(x_def_op));
			left = old2new_res_[x_def_op];
		}
	}
	mlir::Value eq = builder.create<mlir::gdsa::Equal>(
		cmp_op.getLoc(), mlir::IntegerType::get(builder.getContext(), 1), left, right
	);
	old2new_res_[cmp_op.getOperation()] = eq;
}

void LoopTranslator::UpdateJoinHTToGDSA(mlir::loop::UpdateHashTable update_op, mlir::OpBuilder& builder,
																				mlir::BlockArgument& while_idx)
{
	auto mlir_ctxt = builder.getContext();
	auto old_create = llvm::dyn_cast<mlir::loop::CreateHashTable>(update_op.ht().getDefiningOp());
	auto create_map = builder.create<mlir::gdsa::CreateMap>(
		update_op.getLoc(), mlir::gdsa::MultiMapType::get(mlir_ctxt)
	);
	create_map.getOperation()->moveBefore(old_create);
	old2new_res_[old_create.getOperation()] = create_map.map();
	// old_create.ht().replaceAllUsesWith(create_map.map());
	assert(update_op.keys().size() == 1 && "support only 1 build key");
	for (auto key : update_op.keys()) {
		// auto key_col_ref = key.dyn_cast_or_null<mlir::relalg::ColumnRefAttr>();
		mlir::Value key_elem = builder.create<mlir::gdsa::GetElement>(
			update_op.getLoc(), mlir::IntegerType::get(mlir_ctxt, 32), key, while_idx
		);
		mlir::Value pair = builder.create<mlir::gdsa::MakePair>(
			update_op.getLoc(), mlir::gdsa::PairType::get(mlir_ctxt), key_elem, while_idx
		);
		builder.create<mlir::gdsa::InsertMap>(
			update_op.getLoc(), create_map.map(), pair, update_op.payloads()
		);
	}
}

void LoopTranslator::UpdateAggrHTToGDSA(mlir::loop::UpdateHashTable update_op, mlir::OpBuilder& builder,
											 									mlir::BlockArgument& while_idx)
{
	auto mlir_ctxt = builder.getContext();
	auto old_create = llvm::dyn_cast<mlir::loop::CreateHashTable>(update_op.ht().getDefiningOp());
	auto create_map = builder.create<mlir::gdsa::CreateMap>(
		update_op.getLoc(), mlir::gdsa::MapType::get(mlir_ctxt)
	);
	create_map.getOperation()->moveBefore(old_create);
	std::vector<mlir::Value> group_keys;
	for (auto group_key_attr : update_op.keys()) {
		mlir::relalg::Column* group_key_col = 
			group_key_attr.dyn_cast<mlir::relalg::ColumnRefAttr>().getColumnPtr().get();
		// When the key is not defined in this while loop, it is not index of build table but of probe table
		// So just use the `while_idx`
		mlir::Operation* current_while
			= builder.getBlock()->front().getParentOfType<mlir::gdsa::While>();
		if (!col2build_idx.contains(group_key_col)) {
			group_keys.push_back(while_idx);
		}
		else {
			if (col2build_idx[group_key_col].isa<mlir::BlockArgument>()) {
				group_keys.push_back(col2build_idx[group_key_col]);
			}
			else {
				mlir::Operation* group_key_res_parent_while
					= col2build_idx[group_key_col].getDefiningOp()->getParentOfType<mlir::gdsa::While>();
				if (current_while == group_key_res_parent_while) {
					group_keys.push_back(col2build_idx[group_key_col]);
				}
				else {
					group_keys.push_back(while_idx);
				}
			}
		}
	}
	mlir::Value aggr_group_key = builder.create<mlir::gdsa::MakeAggrKey>(
		update_op.getLoc(), mlir::gdsa::AggrKeyType::get(mlir_ctxt), group_keys
	);
	// Decide aggr_idx by aggr_col is in build or probe table
	mlir::Value aggr_idx;
	for (auto aggr_col_ref : update_op.payloads()) {
		mlir::relalg::Column* aggr_col = 
			aggr_col_ref.dyn_cast<mlir::relalg::ColumnRefAttr>().getColumnPtr().get();
		if (col2build_idx.contains(aggr_col)) {
			aggr_idx = col2build_idx[aggr_col];
		}
		else {
			aggr_idx = while_idx;
		}
	}
	mlir::Value aggr_pair = builder.create<mlir::gdsa::MakePair>(
		update_op.getLoc(), mlir::gdsa::PairType::get(mlir_ctxt), aggr_group_key, aggr_idx
	);
	builder.create<mlir::gdsa::InsertMap>(
		update_op.getLoc(), create_map.map(), aggr_pair, update_op.payloads()
	);
	mlir::Value res_pair = builder.create<mlir::gdsa::MapFind>(
		update_op.getLoc(), mlir::gdsa::PairType::get(mlir_ctxt), create_map.map(), aggr_group_key
	);
	auto single_op = builder.create<mlir::gdsa::SingleThread>(update_op.getLoc());
	mlir::Block* single_blk = new mlir::Block;
	single_op.getBodyRegion().push_back(single_blk);
	mlir::OpBuilder single_builder(single_op.getBodyRegion());
	mlir::Value tgt_idx = single_builder.create<mlir::gdsa::GetSecond>(
		update_op.getLoc(), while_idx.getType(), res_pair
	);
	mlir::Value tgt_eq_idx = single_builder.create<mlir::gdsa::NotEqual>(update_op.getLoc(), 
		builder.getI1Type(), tgt_idx, aggr_idx,builder.getIntegerAttr(builder.getI64Type(), 0)
	);
	auto if_op = single_builder.create<mlir::gdsa::If>(update_op.getLoc(), tgt_eq_idx);
	// Update hash table size due to CUCO bug ...
	if_op.getOperation()->setAttr("update_num_groups", builder.getBoolAttr(true));
	mlir::Block* if_blk = new mlir::Block;
	if_op.region().push_back(if_blk);
	mlir::OpBuilder if_builder(if_op.getBodyRegion());
	assert(update_op.payloads().size() == 1 && "only 1 aggregation column now");
	for (auto aggr_col_ref : update_op.payloads()) {
		mlir::Value aggr_value = if_builder.create<mlir::gdsa::GetElement>(
			update_op.getLoc(), builder.getI32Type(), aggr_col_ref, aggr_idx
		);
		if_builder.create<mlir::gdsa::AtomicAdd>(update_op.getLoc(),
			mlir::Type{}, aggr_col_ref, tgt_idx, aggr_value
		);
		if_builder.create<mlir::gdsa::End>(update_op.getLoc());
	}
	single_builder.create<mlir::gdsa::End>(update_op.getLoc());
}

void LoopTranslator::ProbeHTToGDSA(mlir::loop::ProbeHashTable probe_op, mlir::OpBuilder& builder,
									 								 mlir::BlockArgument& while_idx)
{
	auto mlir_ctxt = builder.getContext();
	assert(probe_op.keys().size() == 1 && "support only 1 probe key");
	// std::vector<mlir::Value> probe_maps;
	for (auto key : probe_op.keys()) {
		mlir::Value key_elem = builder.create<mlir::gdsa::GetElement>(
			probe_op.getLoc(), mlir::IntegerType::get(mlir_ctxt, 32), key, while_idx
		);
		mlir::Value key_pair = builder.create<mlir::gdsa::MakePair>(
			probe_op.getLoc(), mlir::gdsa::PairType::get(mlir_ctxt), key_elem, while_idx
		);
		assert(old2new_res_.find(probe_op.ht().getDefiningOp()) != old2new_res_.end() &&
					 "res of old create hash table is not recorded");
		mlir::Value probe_map = old2new_res_[probe_op.ht().getDefiningOp()];
		// probe_maps.push_back(probe_map);
		mlir::Value count = builder.create<mlir::gdsa::MapCount>(probe_op.getLoc(), 
			mlir::IntegerType::get(mlir_ctxt, 8), probe_map, key_pair, probe_op.keys(), probe_op.payloads()
		);
		mlir::Value neq = builder.create<mlir::gdsa::NotEqual>(probe_op.getLoc(), 
			builder.getI1Type(), count, nullptr, builder.getIntegerAttr(builder.getI64Type(), 0)
			// probe_op.getLoc(), mlir::IntegerType::get(mlir_ctxt, 1), count, builder.getIntegerAttr(builder.getI64Type(), 1)
		);
		mlir::Value any_eq = builder.create<mlir::gdsa::AnyThread>(probe_op.getLoc(), builder.getI32Type(), neq);
		old2new_res_[probe_op.getOperation()] = any_eq;
	}
	for (auto pld : probe_op.payloads()) {
		auto col = mlir::relalg::getColumnFromAttr(pld);
		col2build_idx[col] = while_idx;
	}
	// // Set stage ID of this while (if not set) and the build while
	// mlir::Operation* curr_while = builder.getBlock()->getParentOp();
	// // Only for the first probe in this while (the builder's parent is while)
	// if (llvm::dyn_cast<mlir::gdsa::While>(curr_while)) {
	// 	curr_while->setAttr("stage", builder.getI32IntegerAttr(curr_stage_));
	// 	// Set the stage of build whiles
	// 	for (mlir::Value probe_map : probe_maps) {
	// 		for (mlir::Operation* map_user : probe_map.getUsers()) {
	// 			if (auto insert_map = llvm::dyn_cast<mlir::gdsa::InsertMap>(map_user)) {
	// 				auto build_while = map_user->getParentOfType<mlir::gdsa::While>();
	// 				build_while.getOperation()->setAttr("stage", builder.getI32IntegerAttr(curr_stage_));
	// 			}
	// 		}
	// 	}
	// 	curr_stage_++;
	// }
}

void LoopTranslator::YiledToGDSA(mlir::loop::YieldOp yield_op, mlir::OpBuilder& builder,
									 							 mlir::BlockArgument& while_idx)
{
	auto mlir_ctxt = builder.getContext();
	auto value = yield_op.results();
	auto parent_for = yield_op.getOperation()->getParentOfType<mlir::loop::ForOp>();
	mlir::Type value_type = value.getType();
	if (value_type == mlir::loop::ProbeResultType::get(mlir_ctxt)) {
		auto single_op = builder.create<mlir::gdsa::SingleThread>(yield_op.getLoc());
		mlir::Block* single_blk = new mlir::Block;
		single_op.getBodyRegion().push_back(single_blk);
		mlir::OpBuilder single_builder(single_op.getBodyRegion());

		// Get payloads that need to be materialized
		auto last_probe_op = llvm::dyn_cast<mlir::loop::ProbeHashTable>(value.getDefiningOp());
		std::unordered_map<mlir::relalg::Column*, mlir::Value> materialize_vals;
		// Materialize build payloads
		for (auto build_pld_ref : build_payloads_in_for[parent_for]) {
			mlir::relalg::Column* build_col = build_pld_ref.getColumnPtr().get();
			mlir::Value build_idx = col2build_idx[build_col];
			mlir::Value build_pld_value = single_builder.create<mlir::gdsa::GetElement>(
				yield_op.getLoc(), single_builder.getI32Type(), build_pld_ref, build_idx
			);
			materialize_vals[build_col] = build_pld_value;
		}
		// Prepare to materialze probe payloads (and partition col)
		auto probe_plds = last_probe_op.payloads();
		for (auto probe_pld : probe_plds) {
			auto probe_pld_ref = probe_pld.dyn_cast<mlir::relalg::ColumnRefAttr>();
			mlir::Value probe_pld_value = single_builder.create<mlir::gdsa::GetElement>(
				yield_op.getLoc(), single_builder.getI32Type(), probe_pld_ref, while_idx
			);
			materialize_vals[probe_pld_ref.getColumnPtr().get()] = probe_pld_value;
		}
		// Get partition col and partition number
		int user_for_of_for_num = 0;
		for (mlir::Operation* user : parent_for.res().getUsers()) {
			// If parent_for is used by materialize, it is the last join (no group by)
			if (llvm::dyn_cast<mlir::relalg::MaterializeOp>(user)) {
				mlir::Value partition_id = single_builder.create<mlir::gdsa::Const>(
					yield_op.getLoc(), builder.getI32Type(), single_builder.getI32IntegerAttr(0) 
				);
				mlir::Value in_partition_idx = single_builder.create<mlir::gdsa::AtomicAdd1>(
					yield_op.getLoc(), mlir::gdsa::RowIndexType::get(mlir_ctxt), 
					mlir::StringAttr::get(mlir_ctxt, "PARAM: num_partition_res"), partition_id
				).res();
				mlir::Value global_idx = single_builder.create<mlir::gdsa::GetGlobalIndex>(
					yield_op.getLoc(), mlir::gdsa::RowIndexType::get(mlir_ctxt), partition_id, in_partition_idx
				);
				std::vector<mlir::Value> to_materialize;
				for (auto val : materialize_vals) {
					to_materialize.push_back(val.second);
				}
				single_builder.create<mlir::gdsa::Materialize>(yield_op.getLoc(), global_idx, to_materialize);
			}
			auto user_for = llvm::dyn_cast<mlir::loop::ForOp>(user);
			if (!user_for) {
				continue;
			}
			int probe_in_user_for_num = 0;
			for (auto fisrt_probe_in_user_for : user_for.getBody()->getOps<mlir::loop::ProbeHashTable>()) {
				int partition_col_num = 0;
				for (auto partition_col_ref : fisrt_probe_in_user_for.keys()) {
					mlir::relalg::Column* partition_col 
						= partition_col_ref.dyn_cast<mlir::relalg::ColumnRefAttr>().getColumnPtr().get();
					assert(materialize_vals.find(partition_col) != materialize_vals.end() && 
								 "probe payload must contain the partition col");
					mlir::Value partition_id = single_builder.create<mlir::gdsa::Mod>(
						yield_op.getLoc(), builder.getI32Type(), materialize_vals[partition_col], 
						mlir::StringAttr::get(mlir_ctxt, "PARAM: num_partitions")
					);
					mlir::Value in_partition_idx = single_builder.create<mlir::gdsa::AtomicAdd1>(
						yield_op.getLoc(), mlir::gdsa::RowIndexType::get(mlir_ctxt), 
						mlir::StringAttr::get(mlir_ctxt, "PARAM: num_partition_res"), partition_id
					).res();
					mlir::Value global_idx = single_builder.create<mlir::gdsa::GetGlobalIndex>(
						yield_op.getLoc(), mlir::gdsa::RowIndexType::get(mlir_ctxt), partition_id, in_partition_idx
					);
					std::vector<mlir::Value> to_materialize;
					for (auto val : materialize_vals) {
						to_materialize.push_back(val.second);
					}
					single_builder.create<mlir::gdsa::Materialize>(yield_op.getLoc(), global_idx, to_materialize);
					partition_col_num++;
				}
				assert(partition_col_num == 1 && "there must be 1 and only 1 partition col");
				probe_in_user_for_num++;
			}
			assert(probe_in_user_for_num == 1 && "there must be 1 and only 1 probe in user loop");
			user_for_of_for_num++;
		}
		assert(user_for_of_for_num <= 1 && "result of ForOp with Yield can only be used once");
		// assert(parent_for.res().getUsers().size() == 1);
		single_builder.create<mlir::gdsa::End>(yield_op.getLoc());
	}
	else if (value_type == mlir::relalg::TupleType::get(mlir_ctxt)) {
		mlir::Type parent_for_input_type = parent_for.table().getType();
		assert(parent_for_input_type == mlir::loop::AggregationHashtableType::get(mlir_ctxt) && 
						"yielded tuple can only come from aggregatoin hash table");
		// Not handle it now
		mlir::Operation* curr_while = builder.getBlock()->getParentOp();
		auto tmp_res = builder.create<mlir::gdsa::CreateResult>(yield_op.getLoc(), mlir::relalg::TupleStreamType::get(mlir_ctxt));
		tmp_res.getOperation()->moveBefore(curr_while);
		parent_for.res().replaceAllUsesWith(tmp_res.res());
	}
	else {
		assert(false && "no other type can be yielded");
	}
}

void LoopTranslator::GetColToGDSA(mlir::relalg::GetColumnOp getcol, mlir::OpBuilder& builder,
																	mlir::Value& idx)
{
	mlir::relalg::Column* col = getcol.attr().getColumnPtr().get();
	assert(col2build_idx.contains(col));
	mlir::Value elem = builder.create<mlir::gdsa::GetElement>(
		getcol.getLoc(), builder.getI32Type(), getcol.attr(), idx
	);
	old2new_res_[getcol] = elem;
}

void LoopTranslator::BinaryToGDSA(mlir::Operation* bin_op, mlir::OpBuilder& builder)
{
	auto left = old2new_res_[bin_op->getOperand(0).getDefiningOp()];
	auto right = old2new_res_[bin_op->getOperand(1).getDefiningOp()];
	mlir::Value res;
	if (mlir::isa<mlir::db::SubOp>(bin_op)) {
		res = builder.create<mlir::gdsa::Sub>(bin_op->getLoc(), builder.getI32Type(), left, right);
	}
	else if (mlir::isa<mlir::db::AddOp>(bin_op)) {
		res = builder.create<mlir::gdsa::Add>(bin_op->getLoc(), builder.getI32Type(), left, right);
	}
	else if (mlir::isa<mlir::db::MulOp>(bin_op)) {
		res = builder.create<mlir::gdsa::Mul>(bin_op->getLoc(), builder.getI32Type(), left, right);
	}
	else {
		assert(false && "not allowed binary op");
	}
	old2new_res_[bin_op] = res;
}

void LoopTranslator::UpdateToGDSA(mlir::loop::UpdateOp update_op, mlir::OpBuilder& builder,
																	mlir::BlockArgument& while_idx)
{
	int update_i = 0;
	for (mlir::Value update_val : update_op.values()) {
		auto new_val = old2new_res_[update_val.getDefiningOp()];
		builder.create<mlir::gdsa::StoreElement>(
			update_op.getLoc(), new_val, update_op.cols()[update_i], while_idx
		);
		update_i += 1;
	}
}

void LoopTranslator::ConstToGDSA(mlir::db::ConstantOp const_op, mlir::OpBuilder& builder)
{
	mlir::Value v = builder.create<mlir::gdsa::Const>(
		const_op.getLoc(), builder.getI32Type(), const_op.value()
	);
	old2new_res_[const_op] = v;
}

void LoopTranslator::NestedIfToGDSA(mlir::loop::IfOp if_op, mlir::OpBuilder& builder,
																		mlir::BlockArgument& while_idx)
{
	auto mlir_ctxt = builder.getContext();
	mlir::Operation* cond_def_op = if_op.cond().getDefiningOp();
	mlir::Value new_cond;
	assert(old2new_res_.find(cond_def_op) != old2new_res_.end());
	new_cond = old2new_res_[cond_def_op];
	mlir::Operation* new_cond_def = new_cond.getDefiningOp();
	auto any_cond = llvm::dyn_cast<mlir::gdsa::AnyThread>(new_cond_def);
	auto neq_cond = llvm::dyn_cast<mlir::gdsa::NotEqual>(new_cond_def);
	if (any_cond) {
		neq_cond = llvm::dyn_cast<mlir::gdsa::NotEqual>(any_cond.input().getDefiningOp());
		assert(neq_cond && "input of `any` can only be `neq`");
	}

	auto new_if = builder.create<mlir::gdsa::If>(if_op.getLoc(), new_cond);
	mlir::Block* if_blk = new mlir::Block;
	new_if.region().push_back(if_blk);
	mlir::OpBuilder if_builder(new_if.region());
	
	if (auto map_count = llvm::dyn_cast<mlir::gdsa::MapCount>(neq_cond.left().getDefiningOp())) {
		auto probe_op = llvm::dyn_cast<mlir::loop::ProbeHashTable>(cond_def_op);
		assert(probe_op);
		auto map_find_op = if_builder.create<mlir::gdsa::MapFind>(probe_op.getLoc(),
			mlir::gdsa::RowIndexType::get(mlir_ctxt), map_count.map(), map_count.pair()
		);
		map_find_op.getOperation()->setAttr("probe_key", probe_op.keys());
		mlir::Value find_res = map_find_op.res();
		// Record the result: row index of build table (for each column)
		auto create_ht = llvm::dyn_cast<mlir::loop::CreateHashTable>(probe_op.ht().getDefiningOp());
		auto build_keys = create_ht.keys();
		assert(build_keys.size() == 1 && "only 1 build key");
		for (auto build_key : build_keys) {
			mlir::relalg::ColumnRefAttr col_ref = build_key.dyn_cast_or_null<mlir::relalg::ColumnRefAttr>();
			assert(col_ref && "must be mlir::relalg::ColumnRefAttr");
			mlir::relalg::Column* col = col_ref.getColumnPtr().get();
			assert(col2build_idx.find(col) == col2build_idx.end() && "it should be the first insert");
			col2build_idx[col] = find_res;
		}
		// Both build key and payload share the same result index
		auto build_plds = create_ht.payloads();
		auto parent_for = probe_op.getOperation()->getParentOfType<mlir::loop::ForOp>();
		for (auto build_pld : build_plds) {
			mlir::relalg::ColumnRefAttr col_ref = build_pld.dyn_cast_or_null<mlir::relalg::ColumnRefAttr>();
			assert(col_ref && "must be mlir::relalg::ColumnRefAttr");
			mlir::relalg::Column* col = col_ref.getColumnPtr().get();
			assert(col2build_idx.find(col) == col2build_idx.end() && "it should be the first insert");
			col2build_idx[col] = find_res;
			build_payloads_in_for[parent_for].push_back(col_ref);
		}
	}
	for (mlir::Operation& in_if_op : if_op.region().front()) {
		if (auto if1 = llvm::dyn_cast<mlir::loop::IfOp>(&in_if_op)) {
			NestedIfToGDSA(if1, if_builder, while_idx);
		}
		else if (auto update_ht = llvm::dyn_cast<mlir::loop::UpdateHashTable>(&in_if_op)) {
			auto ht_type = update_ht.ht().getType();
			if (ht_type == mlir::loop::JoinHashtableType::get(mlir_ctxt)) {
				UpdateJoinHTToGDSA(update_ht, if_builder, while_idx);
			}
			else {
				assert(ht_type == mlir::loop::AggregationHashtableType::get(mlir_ctxt));
				UpdateAggrHTToGDSA(update_ht, if_builder, while_idx);
			}
		}
		else if (auto probe_ht = llvm::dyn_cast<mlir::loop::ProbeHashTable>(&in_if_op)) {
			ProbeHTToGDSA(probe_ht, if_builder, while_idx);
		}
		else if (auto yield = llvm::dyn_cast<mlir::loop::YieldOp>(&in_if_op)) {
			YiledToGDSA(yield, if_builder, while_idx);
		}
		else if (auto getcol = llvm::dyn_cast<mlir::relalg::GetColumnOp>(&in_if_op)) {
			GetColToGDSA(getcol, if_builder, col2build_idx.at(getcol.attr().getColumnPtr().get()));
		}
		else if (auto update = llvm::dyn_cast<mlir::loop::UpdateOp>(&in_if_op)) {
			UpdateToGDSA(update, if_builder, while_idx);
		}
		else if (mlir::isa<mlir::db::SubOp, mlir::db::AddOp, mlir::db::MulOp>(&in_if_op)) {
			BinaryToGDSA(&in_if_op, if_builder);
		}
		else if (auto const_op = llvm::dyn_cast<mlir::db::ConstantOp>(&in_if_op)) {
			ConstToGDSA(const_op, if_builder);
		}
		else {
			in_if_op.dump();
			assert(false && "no other ops should be in loop.if");
		}
	}
	if_builder.create<mlir::gdsa::End>(if_op.getLoc());
}

} // namespace gdsa
} // namespace mlir