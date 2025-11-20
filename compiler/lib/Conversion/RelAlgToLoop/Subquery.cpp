#include "mlir/Conversion/RelAlgToLoop/RelAlgToLoopPass.h"
#include "mlir/Conversion/RelAlgToLoop/UpdateCols.h"
#include "mlir/Dialect/RelAlg/IR/util.h"

namespace {

bool isEndingJoin(mlir::relalg::InnerJoinOp join)
{
	if (std::getenv("DHAP_NAIVE")) {
		return true;
	}
	mlir::Value join_res = join.result();
	for (auto user : join_res.getUsers()) {
		if (auto join_user = llvm::dyn_cast_or_null<mlir::relalg::InnerJoinOp>(user)) {
			if (join_res == join_user.left()) {
				return true;
			}
		}
	}
	return false;
}

void UpdateFinalMaterialize(mlir::OpBuilder& builder, 
														mlir::relalg::ColumnManager& attr_manager,
														mlir::relalg::MaterializeOp mater_op,
														std::string new_table_name,
														const std::vector<std::string>& new_col_names) {
	llvm::SmallVector<mlir::Attribute, 8> new_cols_ref;
	for (auto col_ref : mater_op.cols()) {
		auto col = mlir::relalg::getColumnFromAttr(col_ref);
		const auto col_name = attr_manager.getColName(col);
		bool updated = false;
		for (auto new_col_name : new_col_names) {
			if (col_name == new_col_name) {
				updated = true;
				new_cols_ref.push_back(attr_manager.createRef(new_table_name, col_name));
				break;
			}
		}
		if (!updated) new_cols_ref.push_back(col_ref);
	}
	mater_op->setAttr("cols", builder.getArrayAttr(new_cols_ref));
}

// Should work for the complex graph case
mlir::ModuleOp buildSubQueryModule(
	int subq_id, mlir::OpBuilder& builder, mlir::FunctionType main_type,
	llvm::SmallVector<mlir::relalg::InnerJoinOp, 8>& ending_joins,
	llvm::SmallVector<mlir::relalg::InnerJoinOp, 8>& joins_same_subq,
	std::unordered_map<int, llvm::SmallVector<mlir::Attribute, 8>>& subq_res_cols,
	std::unordered_map<std::string, std::string>& col2res) {
	auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
	builder.setInsertionPointToStart(module.getBody());
	mlir::func::FuncOp func = builder.create<mlir::func::FuncOp>(
		builder.getUnknownLoc(), "main", main_type
	);
	builder.createBlock(&func.getBody());

	std::unordered_map<mlir::Operation*, mlir::Operation*> old2new;
	// Copy joins to new module
	Operator last_join = joins_same_subq.back();
	mlir::relalg::CloneChildrenUntil(old2new, builder, last_join, 
		[&](Operator op) {
			if (op == last_join) return false;
			for (auto& e : ending_joins) {
				if (op == e) return true;
			}
			return false;
		});
	// Process the first join in the subquery
	mlir::Operation* left_def_op = joins_same_subq.front().left().getDefiningOp();
	mlir::Operation* right_def_op = joins_same_subq.front().right().getDefiningOp();
	auto left_def_join = llvm::dyn_cast_or_null<mlir::relalg::InnerJoinOp>(left_def_op);
	auto right_def_join = llvm::dyn_cast_or_null<mlir::relalg::InnerJoinOp>(right_def_op);
	assert((!left_def_join || !right_def_join) && "at most 1 join");
	auto def_join = left_def_join? left_def_join : right_def_join;
	std::string from_subq_res; 
	std::vector<std::string> new_col_names;
	mlir::relalg::ColumnManager& attr_manager = 
		builder.getContext()->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
	if (def_join) {
		int from_subq_id = def_join->getAttr("sub-query").dyn_cast<mlir::IntegerAttr>().getInt();
		from_subq_res = "res_"+std::to_string(from_subq_id);
		const auto from_subq_res_cols = subq_res_cols[from_subq_id];
		assert(from_subq_res_cols.size() > 0);
		const auto new_base_cols = mlir::relalg::CreateNamedColDef(
			builder, attr_manager, from_subq_res, from_subq_res_cols
		);
		
		builder.setInsertionPointToStart(builder.getInsertionBlock());
		old2new[def_join] = builder.create<mlir::relalg::BaseTableOp>(builder.getUnknownLoc(),
			mlir::relalg::TupleStreamType::get(builder.getContext()), from_subq_res, 
			mlir::relalg::TableMetaDataAttr::get(builder.getContext(), 
			std::make_shared<runtime::TableMetaData>()), builder.getDictionaryAttr(new_base_cols)
		);
		old2new[def_join]->setAttr("rows", def_join->getAttr("rows"));
		for (auto col : new_base_cols) {
			new_col_names.push_back(col.getName().str());
		}
	}
	// Process the last join in the subquery
	const std::string res_name = "res_" + std::to_string(subq_id);
	subq_res_cols[subq_id]	
		= mlir::relalg::ProcessEndJoin(old2new, res_name, builder, joins_same_subq.back(), true, col2res);

	// Update operands
	mlir::relalg::UpdateCreatedOperands(old2new);

	if (def_join) {
		// Update columns
		mlir::relalg::ColumnUpdater updater(from_subq_res, new_col_names, false);
		func.walk([&](mlir::Operation* op){
			updater.Update(op, builder, attr_manager);
		});
		bool final_mater = (subq_res_cols[subq_id].size() == 0);
		if (final_mater) {
			auto maters = func->getRegion(0).front().getOps<mlir::relalg::MaterializeOp>();
			for (auto mater_op : maters) {
				UpdateFinalMaterialize(builder, attr_manager, mater_op, from_subq_res, new_col_names);
			}
		}
	}

	builder.setInsertionPointAfter(module);
	return module;
}

class SplitSubqueryPass
	: public mlir::PassWrapper<SplitSubqueryPass, mlir::OperationPass<mlir::func::FuncOp>> {
	virtual llvm::StringRef getArgument() const override { return "split-subquery"; }

public:
	SplitSubqueryPass(llvm::SmallVector<mlir::ModuleOp, 8>& subq_modules)
		: subq_modules_(subq_modules) {}

private:
	llvm::SmallVector<mlir::ModuleOp, 8>& subq_modules_;
	// TODO: support complex graph case
	void runOnOperation() override {
		auto main_type = getOperation().getFunctionType();
		mlir::OpBuilder builder(&getContext());
		llvm::SmallVector<mlir::relalg::InnerJoinOp, 8> ending_joins;
		llvm::SmallVector<mlir::relalg::InnerJoinOp, 8> joins_same_subq;
		std::unordered_map<int, llvm::SmallVector<mlir::Attribute, 8>> subq_res_cols;
		std::unordered_map<std::string, std::string> col2res;
		auto joins = getOperation()->getRegion(0).front().getOps<mlir::relalg::InnerJoinOp>();
		int q_id = 0;
		for (auto join_op : joins) {
			join_op.getOperation()->setAttr("sub-query", builder.getI32IntegerAttr(q_id));
			joins_same_subq.push_back(join_op);
			if (isEndingJoin(join_op)) {
				ending_joins.push_back(join_op);
				subq_modules_.push_back(buildSubQueryModule(q_id++, builder, main_type, 
																										ending_joins, joins_same_subq,
																										subq_res_cols, col2res));
				joins_same_subq.clear();
			}
		}
		const size_t num_joins = std::distance(joins.begin(), joins.end());
		if (!joins_same_subq.empty() && joins_same_subq.size() < num_joins) {		// else no subquery is needed
			subq_modules_.push_back(buildSubQueryModule(q_id, builder, main_type, 
																									ending_joins, joins_same_subq,
																									subq_res_cols, col2res));
		}
	}
};

}



namespace mlir {
namespace relalg {

std::unique_ptr<Pass> createSplitSubqueryPass(llvm::SmallVector<mlir::ModuleOp, 8>& subq_modules) {
	return std::make_unique<SplitSubqueryPass>(subq_modules); 
}

}// end namespace relalg
}// end namespace mlir