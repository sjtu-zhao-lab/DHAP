#include "mlir/Conversion/RelAlgToLoop/FuseLoop.h"
#include "mlir/Dialect/RelAlg/IR/util.h"

using namespace mlir::relalg;

namespace mlir {
namespace relalg {

mlir::Block* GetMostInnerBlock(mlir::loop::ForOp for_op)
{
	mlir::Block* blk = for_op.getBody();
	while (!blk->getOps<mlir::loop::IfOp>().empty()) {
		int if_i = 0;
		for (auto if_op : blk->getOps<mlir::loop::IfOp>()) {
			assert(if_i == 0 && "At most 1 If in each for");
			blk = if_op.getBody();
		}
	}
	return blk;
}

bool TryFuse(mlir::Operation* op, mlir::OpBuilder& builder)
{
	mlir::MLIRContext* mlir_ctxt = builder.getContext();
	auto& attr_manager = mlir_ctxt->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
	auto for_op = llvm::dyn_cast<mlir::loop::ForOp>(op);
	// Fuse the map
	if (auto comp_cols = op->getAttr("computed_cols")) {
		mlir::Operation* map_src_def_op = for_op.table().getDefiningOp();
		if (auto src_for = llvm::dyn_cast<mlir::loop::ForOp>(map_src_def_op)) {
			src_for->setAttr("computed_cols", comp_cols);
			auto src_for_blk_arg = src_for.getBody()->getArgument(0);
			mlir::Block* most_in_blk = GetMostInnerBlock(src_for);
			auto most_in_builder = mlir::OpBuilder(most_in_blk, most_in_blk->end());
			auto src_yield = llvm::dyn_cast<mlir::loop::YieldOp>(&(most_in_blk->back()));
			src_yield.erase();
			
			std::unordered_map<mlir::Operation*, mlir::Value> old2new_res;
			for (mlir::Operation& in_op : for_op.region().front()) {
				llvm::TypeSwitch<mlir::Operation*>(&in_op)
					.Case<mlir::relalg::GetColumnOp>([&](auto getcol_op) {
						CloneGetCol(most_in_builder, getcol_op, src_for_blk_arg, old2new_res);
					})
					.Case<mlir::db::ConstantOp>([&](auto const_op) {
						CloneConstOp(most_in_builder, const_op, old2new_res);
					})
					.Case<mlir::db::SubOp>([&](auto sub_op) {
						CloneBinaryOp(most_in_builder, sub_op, old2new_res);
					})
					.Case<mlir::db::MulOp>([&](auto mul_op) {
						CloneBinaryOp(most_in_builder, mul_op, old2new_res);
					})
					.Case<mlir::loop::UpdateOp>([&](auto update_op) {
						llvm::SmallVector<mlir::Value, 4> new_update;
						for (auto v : update_op.values()) {
							new_update.push_back(old2new_res[v.getDefiningOp()]);
						}
						old2new_res[update_op] = most_in_builder.create<mlir::loop::UpdateOp>(
							update_op.getLoc(), update_op.output().getType(),
							src_for_blk_arg, new_update, update_op.cols()
						);
					})
					.Case<mlir::loop::YieldOp>([&](auto yield_op) {
						mlir::Operation* new_yield = most_in_builder.create<mlir::loop::YieldOp>(
							yield_op->getLoc(), old2new_res[yield_op.results().getDefiningOp()]
						);
						new_yield->setAttr("computed_cols", comp_cols);
					})
					.Default([](auto x) {
						assert(false && "not allowed op within map");
					});
			}
			for_op.res().replaceAllUsesWith(src_for.res());
			return true;
		}
	}

	// Fuse the loop that builds hash table to the source loop
	auto update_ops = for_op.getBody()->getOps<mlir::loop::UpdateHashTable>();
	int update_i = 0;
	// auto join_ht_type = mlir::loop::JoinHashtableType::get(mlir_ctxt);
	for (auto update_op : update_ops) {
		assert(update_i == 0 && "At most 1 UpdateHashTable in each for");
		mlir::Operation* build_tbl_def_op = for_op.table().getDefiningOp();
		if (auto src_for = llvm::dyn_cast<mlir::loop::ForOp>(build_tbl_def_op)) {
			if (op->hasAttr("stage")) {
				build_tbl_def_op->setAttr("stage", op->getAttr("stage"));
			}
			if (op->hasAttr("build_keys")) {
				build_tbl_def_op->setAttr("build_keys", op->getAttr("build_keys"));
			}
			if (op->hasAttr("build_payloads")) {
				build_tbl_def_op->setAttr("build_payloads", op->getAttr("build_payloads"));
			}
			// For aggregation hash table build
			if (op->hasAttr("aggr_col")) {
				build_tbl_def_op->setAttr("aggr_col", op->getAttr("aggr_col"));
			}
			if (op->hasAttr("groupby_keys")) {
				auto all_groupby_keys = mlir::relalg::ColumnSet::fromArrayAttr(op->getAttr("groupby_keys").dyn_cast<mlir::ArrayAttr>());
				auto aggr_cols = mlir::relalg::ColumnSet::fromArrayAttr(op->getAttr("aggr_col").dyn_cast<mlir::ArrayAttr>());
				// Cancel overlapped output probe payloads
				auto probe_plds = mlir::relalg::ColumnSet::fromArrayAttr(
					build_tbl_def_op->getAttr("probe_payloads").dyn_cast<mlir::ArrayAttr>()
				);
				probe_plds.remove(all_groupby_keys);
				probe_plds.remove(aggr_cols);
				build_tbl_def_op->setAttr("probe_payloads", probe_plds.asRefArrayAttr(mlir_ctxt));
				// To discriminate groupby keys from build table and probe table
				mlir::relalg::ColumnSet groupby_keys_build;
				build_tbl_def_op->walk([&](mlir::Operation* x){
					if (auto probe = llvm::dyn_cast_or_null<mlir::loop::ProbeHashTable>(x)) {
						auto create_ht = llvm::dyn_cast<mlir::loop::CreateHashTable>(probe.ht().getDefiningOp());
						auto join_build_payloads = mlir::relalg::ColumnSet::fromArrayAttr(create_ht.payloads());
						// Remove build payloads (in groupby keys) of build hash table ForOp
						for (mlir::Operation* ht_user : create_ht.ht().getUsers()) {
							if (auto update_ht = llvm::dyn_cast<mlir::loop::UpdateHashTable>(ht_user)) {
								auto join_build_for = ht_user->getParentOfType<mlir::loop::ForOp>();
								auto build_gb_keys = join_build_payloads.intersect(all_groupby_keys);
								groupby_keys_build.insert(build_gb_keys);

								join_build_payloads.remove(build_gb_keys);
								join_build_for.getOperation()->setAttr(
									"build_payloads", join_build_payloads.asRefArrayAttr(mlir_ctxt)
								);
							}
						}
					}
				});
				build_tbl_def_op->setAttr("groupby_keys", all_groupby_keys.asRefArrayAttr(mlir_ctxt));
				build_tbl_def_op->setAttr("groupby_keys_build", groupby_keys_build.asRefArrayAttr(mlir_ctxt));
				all_groupby_keys.remove(groupby_keys_build);
				build_tbl_def_op->setAttr("groupby_keys_probe", all_groupby_keys.asRefArrayAttr(mlir_ctxt));
			}
			build_tbl_def_op->setAttr("name_res", op->getAttr("name_res"));
			// To make the input of filter also end with `_s`
			if (for_op.name_inp().str().ends_with("_s")) {
				build_tbl_def_op->setAttr("name_inp", op->getAttr("name_inp"));
			}
			mlir::Block* most_in_blk = GetMostInnerBlock(src_for);
			auto most_in_builder = mlir::OpBuilder(most_in_blk, most_in_blk->end());
			update_op.ht().getDefiningOp()->moveBefore(build_tbl_def_op);
			if (auto src_yield = llvm::dyn_cast_or_null<mlir::loop::YieldOp>(&(most_in_blk->back()))) {
				most_in_builder.create<mlir::loop::UpdateHashTable>(
					update_op.getLoc(), update_op.ht(), src_yield.results(),
					update_op.keys(), update_op.payloads()
				);
				src_yield.erase();
			}
			else {
				assert(false && "the src loop must end with yield");
			}
			
			return true;
		}
		update_i++;
	}

	// Fuse the probe loops if its operands is not shuffled
	auto probe_ops = for_op.getBody()->getOps<mlir::loop::ProbeHashTable>();
	int probe_i = 0;
	for (auto probe_op : probe_ops) {
		assert(probe_i == 0 && "At most 1 ProbeHashTable in each for");
		assert(for_op.res() && "Probe loop has result");
		for (mlir::Operation* probe_tbl_user : for_op.table().getUsers()) {
			// llvm::outs() << "Probe tbl user: " << probe_tbl_user->getName() << "\n";
			if (llvm::dyn_cast<mlir::loop::Shuffle>(probe_tbl_user)) {
				// A probe after shuffle, do not fuse
				return false;
			}
		}
		// Fuse to the source loop
		mlir::Operation* probe_tbl_def_op = for_op.table().getDefiningOp();
		if (auto src_for = llvm::dyn_cast<mlir::loop::ForOp>(probe_tbl_def_op)) {
			// Modify the attibutets
			assert(op->getAttr("stage") == probe_tbl_def_op->getAttr("stage"));
			// Collect probe keys and build keys
			mlir::ArrayAttr src_probe_keys = probe_tbl_def_op->getAttr("probe_keys").dyn_cast<mlir::ArrayAttr>();
			llvm::SmallVector<mlir::Attribute, 8> src_probe_keys_vec(src_probe_keys.getValue().begin(), 
																															 src_probe_keys.getValue().end());
			mlir::ArrayAttr curr_probe_keys = op->getAttr("probe_keys").dyn_cast<mlir::ArrayAttr>();
			llvm::SmallVector<mlir::Attribute, 2> curr_probe_keys_vec(curr_probe_keys.getValue().begin(), 
																															 	curr_probe_keys.getValue().end());
			src_probe_keys_vec.append(curr_probe_keys_vec.begin(), curr_probe_keys_vec.end());
			probe_tbl_def_op->setAttr("probe_keys", builder.getArrayAttr(src_probe_keys_vec));
			mlir::ArrayAttr src_build_keys = probe_tbl_def_op->getAttr("build_keys").dyn_cast<mlir::ArrayAttr>();
			llvm::SmallVector<mlir::Attribute, 8> src_build_keys_vec(src_build_keys.getValue().begin(), 
																															 src_build_keys.getValue().end());
			mlir::ArrayAttr curr_build_keys = op->getAttr("build_keys").dyn_cast<mlir::ArrayAttr>();
			llvm::SmallVector<mlir::Attribute, 2> curr_build_keys_vec(curr_build_keys.getValue().begin(), 
																															 	curr_build_keys.getValue().end());
			src_build_keys_vec.append(curr_build_keys_vec.begin(), curr_build_keys_vec.end());
			probe_tbl_def_op->setAttr("build_keys", builder.getArrayAttr(src_build_keys_vec));
			// Reduce probe payloads
			mlir::relalg::ColumnSet build_plds_in_to_fuse_loop;
			probe_tbl_def_op->walk([&](mlir::Operation* x) {
				if (auto probe_op = llvm::dyn_cast<mlir::loop::ProbeHashTable>(x)) {
					auto create_ht = llvm::dyn_cast<mlir::loop::CreateHashTable>(probe_op.ht().getDefiningOp());
					auto build_plds = mlir::relalg::ColumnSet::fromArrayAttr(create_ht.payloads());
					build_plds_in_to_fuse_loop.insert(build_plds);
				}
			});
			auto reduced_probe_plds = 
				mlir::relalg::ColumnSet::fromArrayAttr(op->getAttr("probe_payloads").dyn_cast<mlir::ArrayAttr>());
			// As build payloads has been renamed
			for (const auto* probe_pld_col : reduced_probe_plds) {
				auto probe_pld_col_name = attr_manager.getColName(probe_pld_col);
				for (const auto* build_pld_col : build_plds_in_to_fuse_loop) {
					auto build_pld_col_name = attr_manager.getColName(build_pld_col);
					if (probe_pld_col_name == build_pld_col_name) {
						reduced_probe_plds.remove(probe_pld_col);
					}
				}
			}
			probe_tbl_def_op->setAttr("probe_payloads", reduced_probe_plds.asRefArrayAttr(mlir_ctxt));
			// Rename results
			probe_tbl_def_op->setAttr("name_res", op->getAttr("name_res"));
			// Move the CreateHashTable and build loop before probe, if not yet
			mlir::Operation* create_ht = probe_op.ht().getDefiningOp();
			if (!create_ht->isBeforeInBlock(probe_tbl_def_op)) {
				create_ht->moveBefore(probe_tbl_def_op);
				// Find the build loop
				for (mlir::Operation* ht_user : probe_op.ht().getUsers()) {
					if (llvm::dyn_cast<mlir::loop::UpdateHashTable>(ht_user)) {
						auto build_for = ht_user->getParentOfType<mlir::loop::ForOp>();
						// Move the build basetable before create_ht first
						mlir::Operation* build_basetable = build_for.table().getDefiningOp();
						if (!build_basetable->isBeforeInBlock(create_ht)) {
							build_basetable->moveBefore(create_ht);
						}
						// Then move the build for
						build_for.getOperation()->moveAfter(create_ht);
					}
				}
			}
			// Fuse it
			auto src_for_blk_arg = src_for.getBody()->getArgument(0);
			mlir::Block* most_in_blk = GetMostInnerBlock(src_for);
			auto most_in_builder = mlir::OpBuilder(most_in_blk, most_in_blk->begin());
			auto new_probe = most_in_builder.create<mlir::loop::ProbeHashTable>(
				probe_op.getLoc(), mlir::loop::ProbeResultType::get(mlir_ctxt), 
				probe_op.ht(), src_for_blk_arg, probe_op.keys(), reduced_probe_plds.asRefArrayAttr(mlir_ctxt)
			);
			probe_op.res().replaceAllUsesWith(new_probe.res());
			for (mlir::Operation* probe_res_user : new_probe.res().getUsers()) {
				if (llvm::dyn_cast<mlir::loop::IfOp>(probe_res_user)) {
					most_in_builder.clone(*probe_res_user);
				}
			}
			// Erase the last Yield
			most_in_blk->back().erase();

			for_op.res().replaceAllUsesWith(src_for.res());
			return true;
		}
		
		probe_i++;
	}

	return false;
}

} // end namespace relalg
} // end namespace mlir 