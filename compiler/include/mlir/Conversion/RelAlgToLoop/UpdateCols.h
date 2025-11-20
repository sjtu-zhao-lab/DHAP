#ifndef MLIR_CONVERSION_RELALGTOLOOP_UPDATECOLS_H
#define MLIR_CONVERSION_RELALGTOLOOP_UPDATECOLS_H

#include "mlir/Conversion/RelAlgToLoop/RelAlgToLoopPass.h"
#include "mlir/Dialect/RelAlg/IR/util.h"

namespace mlir {
namespace relalg {

class ColumnUpdater
{
public:
ColumnUpdater(std::string new_table_name0,
							std::vector<std::string> new_col_names0,
							bool preprocess0)
	:	new_table_name(new_table_name0),
		new_col_names(new_col_names0),
		preprocess(preprocess0)
		{}

void Update(mlir::Operation* op, mlir::OpBuilder& builder, mlir::relalg::ColumnManager& attr_manager) {
	llvm::TypeSwitch<mlir::Operation*>(op)
		.Case<mlir::relalg::GetColumnOp>([&](auto x) { UpdateGetCol(builder, attr_manager, x); })
		.Case<mlir::relalg::AggregationOp>([&](auto x) { UpdateAggr(builder, attr_manager, x); })
		.Case<mlir::relalg::AggrFuncOp>([&](auto x) { UpdateAggrFunc(builder, attr_manager, x); })
		.Case<mlir::relalg::SortOp>([&](auto x) { UpdateSortOp(builder, attr_manager, x); })
		.Case<mlir::db::ConstantOp>([&](auto x) { 
			if (preprocess)
				UpdateConstOp(builder, attr_manager, x); 
		})
		.Case<mlir::db::SubOp, mlir::db::AddOp>([&](auto x) { 
			if (preprocess)
				UpdateBinaryOp(builder, attr_manager, x); 
		})
		.Case<mlir::db::CmpOp>([&](auto x) {
			if (preprocess)
				UpdateCmpOp(builder, attr_manager, x); 
		})
		.Case<mlir::relalg::MapOp>([&](auto x) { 
			if (preprocess)
				UpdateMapOp(builder, attr_manager, x); 
		})
		.Default([](auto x) {});
}

private:
	std::string new_table_name;
	std::vector<std::string> new_col_names;
	bool preprocess;

	void UpdateGetCol(mlir::OpBuilder& builder, mlir::relalg::ColumnManager& attr_manager, mlir::relalg::GetColumnOp getcol_op) {
		mlir::Operation* op = getcol_op.getOperation();
		mlir::relalg::Column* col = getcol_op.attr().getColumnPtr().get();
		auto [table, col_name] = attr_manager.getName(col);
		bool erased = false;
		for (auto new_col_name : new_col_names) {
			if (new_col_name == col_name) {   // Update old probe table with intermediates
				builder.setInsertionPoint(getcol_op);
				auto new_attr = attr_manager.createRef(new_table_name, col_name);
				mlir::Value new_res = builder.create<mlir::relalg::GetColumnOp>(builder.getUnknownLoc(),
					new_attr.getColumn().type, new_attr, op->getBlock()->getArgument(0)
				);
				getcol_op.res().replaceAllUsesWith(new_res);
				op->erase();
				erased = true;
				break;
			}
		}
		if (!erased) {
			builder.setInsertionPoint(getcol_op);
			mlir::Value new_res = builder.create<mlir::relalg::GetColumnOp>(builder.getUnknownLoc(),
				getcol_op.attr().getColumn().type, getcol_op.attr(), op->getBlock()->getArgument(0)
			);
			getcol_op.res().replaceAllUsesWith(new_res);
			op->erase();
		}
	}
	void UpdateAggr(mlir::OpBuilder& builder, mlir::relalg::ColumnManager& attr_manager, mlir::relalg::AggregationOp aggr_op) {
		mlir::ArrayAttr groupby_cols_attr = aggr_op.group_by_cols();
		llvm::SmallVector<mlir::Attribute, 4> new_groupby_cols_attr;
		for (auto const col_attr : groupby_cols_attr) {
			auto col = mlir::relalg::getColumnFromAttr(col_attr);
			auto [table, col_name] = attr_manager.getName(col);
			bool updated = false;
			for (auto new_col_name : new_col_names) {
				if (new_col_name == col_name) {
					updated = true;
					auto new_col_attr = attr_manager.createRef(new_table_name, col_name);
					new_groupby_cols_attr.push_back(new_col_attr);
					break;
				}
			}
			if (!updated) {
				new_groupby_cols_attr.push_back(col_attr);
			}
		}
		aggr_op.getOperation()->setAttr("group_by_cols", builder.getArrayAttr(new_groupby_cols_attr));
	}
	void UpdateAggrFunc(mlir::OpBuilder& builder, mlir::relalg::ColumnManager& attr_manager, mlir::relalg::AggrFuncOp aggr_func) {
		auto new_attr = aggr_func.attr();
		mlir::relalg::Column* col = aggr_func.attr().getColumnPtr().get();
		auto [table, col_name] = attr_manager.getName(col);
		for (auto new_col_name : new_col_names) {
			if (new_col_name == col_name) {
				new_attr = attr_manager.createRef(new_table_name, col_name);
				break;
			}
		}
		// Build a new aggrfunc although not a new table, to update the type
		builder.setInsertionPoint(aggr_func);
		auto new_aggr_func = builder.create<mlir::relalg::AggrFuncOp>(builder.getUnknownLoc(),
			new_attr.getColumn().type, aggr_func.fn(), aggr_func.rel(), new_attr
		);
		aggr_func.result().replaceAllUsesWith(new_aggr_func.result());
		aggr_func.getOperation()->erase();
		// Update the type of parent aggregation op
		mlir::Operation* parent_op = new_aggr_func.getOperation()->getParentOp();
		auto aggr_op = llvm::dyn_cast<mlir::relalg::AggregationOp>(parent_op);
		assert(aggr_op);
		mlir::ArrayAttr aggr_compute_cols = aggr_op.computed_cols();
		assert(aggr_compute_cols.size() == 1);
		aggr_compute_cols[0].dyn_cast<mlir::relalg::ColumnDefAttr>().getColumn().type = new_attr.getColumn().type;
	}
	void UpdateSortOp(mlir::OpBuilder& builder, mlir::relalg::ColumnManager& attr_manager, mlir::relalg::SortOp sort_op) {
		// Just ignore sort now
		if (preprocess) {
			sort_op.result().replaceAllUsesWith(sort_op.rel());
			sort_op.getOperation()->erase();
		}
		else {
			llvm::SmallVector<mlir::Attribute, 4> new_specs;
			for (auto sort_spec0 : sort_op.sortspecs()) {
				auto sort_spec = sort_spec0.dyn_cast_or_null<mlir::relalg::SortSpecificationAttr>();
				auto col_ref = sort_spec.getAttr();
				auto col_name = attr_manager.getColName(&col_ref.getColumn());
				bool updated = false;
				for (auto new_col_name : new_col_names) {
					if (new_col_name == col_name) {   // Update old probe table with intermediates
						updated = true;
						auto new_attr = attr_manager.createRef(new_table_name, col_name);
						new_specs.push_back(mlir::relalg::SortSpecificationAttr::get(
							builder.getContext(), new_attr, sort_spec.getSortSpec()
						));
						break;
					}
				}
				if (!updated) new_specs.push_back(sort_spec);
			}
			builder.setInsertionPoint(sort_op);
			sort_op->replaceAllUsesWith(
				builder.create<mlir::relalg::SortOp>(builder.getUnknownLoc(), sort_op.result().getType(),
																						sort_op.rel(), builder.getArrayAttr(new_specs))
			);
			sort_op->erase();
		}
	}
	void UpdateConstOp(mlir::OpBuilder& builder, mlir::relalg::ColumnManager& attr_manager, mlir::db::ConstantOp const_op) {
		mlir::Operation* op = const_op.getOperation();
		mlir::Attribute old_value = const_op.value();
		mlir::StringAttr old_value_str = old_value.dyn_cast_or_null<mlir::StringAttr>();
		if (old_value_str) {
			auto hash = std::hash<std::string>{};
			int int_value = hash(old_value_str.str());
			builder.setInsertionPoint(const_op);
			mlir::Value new_res = builder.create<mlir::db::ConstantOp>(builder.getUnknownLoc(),
				builder.getI32Type(), mlir::IntegerAttr::get(builder.getI32Type(), int_value)
			);
			const_op.result().replaceAllUsesWith(new_res);
			op->erase();
		}
		else {
			assert(old_value.isa<mlir::IntegerAttr>());
		}
	}
	void UpdateCmpOp(mlir::OpBuilder& builder, mlir::relalg::ColumnManager& attr_manager, mlir::db::CmpOp cmp_op) {
		cmp_op.res().setType(builder.getI1Type());
	}
	void UpdateBinaryOp(mlir::OpBuilder& builder, mlir::relalg::ColumnManager& attr_manager, mlir::Operation* bin_op) {
		assert(mlir::isa<mlir::db::SubOp>(bin_op));
		bin_op->getResult(0).setType(builder.getI32Type());
	}
	void UpdateMapOp(mlir::OpBuilder& builder, mlir::relalg::ColumnManager& attr_manager, mlir::relalg::MapOp map_op) {
		for (auto col_attr : map_op.computed_cols()) {
			auto col_def = col_attr.dyn_cast<mlir::relalg::ColumnDefAttr>();
			col_def.getColumn().type = builder.getI32Type();
		}
	}
};

}// end namespace relalg
}// end namespace mlir

#endif // MLIR_CONVERSION_RELALGTODB_UPDATECOLS_H