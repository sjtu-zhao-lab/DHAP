
#include "mlir/Conversion/RelAlgToLoop/UnrollRelAlg.h"
#include "mlir/Dialect/RelAlg/IR/util.h"
#include <fstream>

using namespace mlir::relalg;

namespace mlir {
namespace relalg {

void InsertShflBtwJoins(nlohmann::json& plan, std::vector<mlir::relalg::InnerJoinOp> joins)
{
	// get shfl_points from plan file
	std::vector<int> shfl_points = plan["shfl_points"];

	int stage = 0;
	for (size_t i = 0; i < joins.size(); i++) {
		auto join_op = joins[i];
		OpBuilder builder(join_op);
		if (std::find(shfl_points.begin(), shfl_points.end(), i) != shfl_points.end()) {
			// For the build table, get its basetable op and shuffle
			mlir::Operation* build_def_op = join_op.left().getDefiningOp();
			while (!llvm::dyn_cast<mlir::relalg::BaseTableOp>(build_def_op)) {
				assert(build_def_op->getNumOperands() == 1);
				build_def_op = build_def_op->getOperand(0).getDefiningOp();
			}
			auto build_basetable = llvm::dyn_cast<mlir::relalg::BaseTableOp>(build_def_op);
			builder.create<mlir::loop::Shuffle>(join_op.getLoc(), build_basetable.result());
			
			mlir::Operation* probe_def_op = join_op.right().getDefiningOp();
			assert(llvm::dyn_cast<mlir::relalg::BaseTableOp>(probe_def_op) || llvm::dyn_cast<mlir::relalg::InnerJoinOp>(probe_def_op));
			builder.create<mlir::loop::Shuffle>(join_op.getLoc(), join_op.right());
			stage++;
		}
		join_op.getOperation()->setAttr("stage", builder.getI32IntegerAttr(stage));
	}
}

// void ReplaceJoinProbeTableIds(std::vector<mlir::relalg::InnerJoinOp> joins)
// {
// 	OpBuilder builder(joins[0]);
// 	auto& attr_manager = builder.getContext()->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
// 	for (size_t i = 0; i < joins.size(); i++) {
// 		auto join_op = joins[i];
// 		mlir::Operation* join_op0 = join_op.getOperation();
// 		// The build table must be from a selection or basetable
// 		mlir::Operation* build_def_op = join_op.left().getDefiningOp();
// 		assert(llvm::dyn_cast<BaseTableOp>(build_def_op) || llvm::dyn_cast<SelectionOp>(build_def_op));
// 		mlir::Operation* base_build_tbl = build_def_op;
// 		if (!llvm::dyn_cast<BaseTableOp>(build_def_op)) {
// 			auto sel_op = llvm::dyn_cast<SelectionOp>(build_def_op);
// 			base_build_tbl = sel_op.rel().getDefiningOp();
// 		}
// 		std::string base_build_tbl_name 
// 			= base_build_tbl->getAttr("table_identifier").dyn_cast<mlir::StringAttr>().getValue().str();
// 		join_op0->setAttr("build_table", builder.getStringAttr(base_build_tbl_name));
// 		// For probe table
// 		auto probe_tbl = join_op.right();
// 		mlir::Operation* base_probe_tbl = probe_tbl.getDefiningOp();
// 		while (!llvm::dyn_cast<BaseTableOp>(base_probe_tbl)) {
// 			auto probe_def_join = llvm::dyn_cast<InnerJoinOp>(base_probe_tbl);
// 			assert(probe_def_join && "probe table must be from basetable or join");
// 			base_probe_tbl = probe_def_join.right().getDefiningOp();
// 		}
// 		std::string base_probe_tbl_name 
// 			= base_probe_tbl->getAttr("table_identifier").dyn_cast<mlir::StringAttr>().getValue().str();
		
// 		if (probe_tbl.getDefiningOp() != base_probe_tbl) {
// 			assert(i > 0);
// 			std::string probe_name = "join_" + std::to_string(i);
// 			for (mlir::Operation* probe_tbl_user : probe_tbl.getUsers()) {
// 				if (llvm::dyn_cast<mlir::loop::Shuffle>(probe_tbl_user)) {
// 					probe_name += "_s";
// 					break;
// 				}
// 			}
// 			join_op0->setAttr("old_inp", builder.getStringAttr(base_probe_tbl_name));
// 			join_op0->setAttr("new_inp", builder.getStringAttr(probe_name));
// 			// join_op0->walk([&](mlir::Operation* x){
// 			// 	if (auto get_col = llvm::dyn_cast<GetColumnOp>(x)) {
// 			// 		mlir::relalg::Column* old_col = get_col.attr().getColumnPtr().get();
// 			// 		auto [old_table, old_col_name] = attr_manager.getName(old_col);
// 			// 		// Replace the name of old probe table
// 			// 		if (old_table == base_probe_tbl_name) {
// 			// 			x->setAttr("attr", attr_manager.createRef(probe_name, old_col_name));
// 			// 		}
// 			// 	}
// 			// });
// 		}
// 	}
// }


} // end namespace relalg
} // end namespace mlir 