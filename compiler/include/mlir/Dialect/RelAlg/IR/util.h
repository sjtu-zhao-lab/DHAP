#ifndef MLIR_DIALECT_RELALG_IR_UTIL_H
#define MLIR_DIALECT_RELALG_IR_UTIL_H

#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"

namespace mlir {
namespace relalg {

inline mlir::relalg::Column* getColumnFromAttr(mlir::Attribute attr) {
	// return attr.dyn_cast_or_null<mlir::relalg::ColumnRefAttr>().getColumnPtr().get();
	if (auto col_ref = attr.dyn_cast_or_null<mlir::relalg::ColumnRefAttr>()) {
		return col_ref.getColumnPtr().get();
	}
	else if (auto col_def = attr.dyn_cast_or_null<mlir::relalg::ColumnDefAttr>()) {
		return col_def.getColumnPtr().get();
	}
	else {
		assert(false);
	}
}

std::vector<mlir::relalg::Column*> getOpAttrCols(mlir::Operation* op, std::string attr_name);

std::vector<std::string> getOpAttrColNames(mlir::Operation* op, std::string attr_name,
																					 mlir::relalg::ColumnManager& attr_manager);

std::vector<std::string> getOpAttrColTypes(mlir::Operation* op, std::string attr_name);

void CloneChildrenUntil(std::unordered_map<mlir::Operation*, mlir::Operation*>& old2new,
												mlir::OpBuilder& builder, Operator op,
												llvm::function_ref<bool(Operator)> is_end);

void UpdateCreatedOperands(std::unordered_map<mlir::Operation*, mlir::Operation*>& old2new);

llvm::SmallVector<mlir::Attribute, 8>
ProcessEndJoin(std::unordered_map<mlir::Operation*, mlir::Operation*>& old2new,
								const std::string res_name, mlir::OpBuilder& builder,
								mlir::relalg::InnerJoinOp& end_join, bool do_update,
								std::unordered_map<std::string, std::string>& col2res);

mlir::Type ToMLIRType(mlir::MLIRContext* mlir_ctxt, std::string type);

llvm::SmallVector<mlir::NamedAttribute, 8> 
CreateNamedColDef(mlir::OpBuilder& builder,
									mlir::relalg::ColumnManager& attr_manager, std::string tbl_name, 
									const llvm::SmallVector<mlir::Attribute, 8>& col_refs);

llvm::SmallVector<mlir::NamedAttribute, 8>
CreateNamedColDef(mlir::OpBuilder& builder,
									mlir::relalg::ColumnManager& attr_manager, std::string tbl_name, 
									const std::vector<std::string>& col_names);

void CloneGetCol(mlir::OpBuilder& builder, GetColumnOp getcol_op, mlir::BlockArgument tuple,
								 std::unordered_map<mlir::Operation*, mlir::Value>& old2new_res);

void CloneConstOp(mlir::OpBuilder& builder, mlir::db::ConstantOp getcol_op,
								 std::unordered_map<mlir::Operation*, mlir::Value>& old2new_res);

void CloneBinaryOp(mlir::OpBuilder& builder, mlir::Operation* op,
									 std::unordered_map<mlir::Operation*, mlir::Value>& old2new_res);

void PopulateUsedColumns(mlir::Operation* op, mlir::relalg::ColumnSet& used);
}// end namespace relalg
}// end namespace mlir

#endif