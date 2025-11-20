#include "mlir/Dialect/RelAlg/IR/util.h"

namespace mlir {
namespace relalg {

std::vector<mlir::relalg::Column*> getOpAttrCols(mlir::Operation* op, std::string attr_name) {
	std::vector<mlir::relalg::Column*> cols;
	if (op->hasAttr(attr_name)) {
		auto array_attr = op->getAttr(attr_name).dyn_cast_or_null<mlir::ArrayAttr>();
		for (auto const attr : array_attr) {
			mlir::relalg::Column* col = getColumnFromAttr(attr);
			cols.push_back(col);
		}
	}
	return cols;
}

std::vector<std::string> getOpAttrColNames(mlir::Operation* op, std::string attr_name,
																					 mlir::relalg::ColumnManager& attr_manager) {
	std::vector<std::string> names;
	if (op->hasAttr(attr_name)) {
		auto array_attr = op->getAttr(attr_name).dyn_cast_or_null<mlir::ArrayAttr>();
		for (auto const attr : array_attr) {
			mlir::relalg::Column* col = getColumnFromAttr(attr);
			auto [table, col_name] = attr_manager.getName(col);
			names.push_back(col_name);
		}
	}
	return names;
}

std::vector<std::string> getOpAttrColTypes(mlir::Operation* op, std::string attr_name) {
	std::vector<std::string> types;
	if (op->hasAttr(attr_name)) {
		auto array_attr = op->getAttr(attr_name).dyn_cast_or_null<mlir::ArrayAttr>();
		for (auto const attr : array_attr) {
			mlir::relalg::Column* col = getColumnFromAttr(attr);
			mlir::Type true_type;
			if (auto null_type = col->type.dyn_cast<mlir::db::NullableType>()) {
				true_type = null_type.getType();
			}
			else {
				true_type = col->type;
			}

			if (true_type.isa<mlir::IntegerType>()) {
				types.push_back("i32");
			}
			else if (true_type.isa<mlir::db::StringType>()) {
				types.push_back("string");
			}
			else if (true_type.isa<mlir::db::DecimalType>()) {
				types.push_back("decimal128");
			}
			else if (true_type.isa<mlir::db::CharType>()) {
				types.push_back("fixed_bin");
			}
			else if (true_type.isa<mlir::db::DateType>()) {
				types.push_back("date");
			}
			else {
				llvm::outs() << true_type << "\n";
				assert(false && "no other types are support on the runtime");
			}
		}
	}
	return types;
}

void CloneChildrenUntil(std::unordered_map<mlir::Operation*, mlir::Operation*>& old2new,
												mlir::OpBuilder& builder, Operator op,
												llvm::function_ref<bool(Operator)> is_end) {
	if (is_end(op)) {
		return;
	}
	old2new[op.getOperation()] = builder.clone(*op);
	builder.setInsertionPoint(old2new[op.getOperation()]);
	// builder.setInsertionPointToStart(builder.getInsertionBlock());
	for (auto child : op.getChildren()) {
		CloneChildrenUntil(old2new, builder, child, is_end);
	}
}

void UpdateCreatedOperands(std::unordered_map<mlir::Operation*, mlir::Operation*>& old2new) {
	for (auto old_new : old2new) {
		mlir::Operation* old_op = old_new.first;
		mlir::Operation* new_op = old_new.second;
		for (uint32_t i = 0; i < old_op->getNumOperands(); i++) {
			mlir::Value old_operand = old_op->getOperand(i);
			mlir::Operation* old_def_op = old_operand.getDefiningOp();
			if (old2new.find(old_def_op) != old2new.end()) {
				mlir::Operation* new_def_op = old2new[old_def_op];
				assert(new_def_op->getNumResults() == 1);
				new_op->setOperand(i, new_def_op->getResult(0));
			}
		}
	}
}

void PopulateUsedColumns(mlir::Operation* op, mlir::relalg::ColumnSet& used) {
	if (auto op0 = llvm::dyn_cast<Operator>(op)) {
		used.insert(op0.getUsedColumns());
		for (auto user : op->getUsers()) {
			PopulateUsedColumns(user, used);
		}
	}
}

llvm::SmallVector<mlir::Attribute, 8> 
CreateColRefs(mlir::relalg::ColumnManager& attr_manager,
							const std::unordered_map<std::string, std::string>& col2res,
							const llvm::SmallVector<std::string, 8>& col_names) {
	llvm::SmallVector<mlir::Attribute, 8> col_refs;
	for (auto col_name : col_names) {
		col_refs.push_back(attr_manager.createRef(col2res.at(col_name), col_name));
	}
	return col_refs;
}

llvm::SmallVector<mlir::Attribute, 8>
ProcessEndJoin(std::unordered_map<mlir::Operation*, mlir::Operation*>& old2new,
								const std::string res_name, mlir::OpBuilder& builder,
								mlir::relalg::InnerJoinOp& end_join, bool do_update,
								std::unordered_map<std::string, std::string>& col2res) {
	builder.setInsertionPointToEnd(builder.getInsertionBlock());
	auto last_join_res_users = end_join.result().getUsers();
	for (mlir::Operation* join_res_user : last_join_res_users) {
		// If it is the last subquery, clone the aggregation/materialization
		if (mlir::isa<mlir::relalg::MapOp>(join_res_user) || 
				mlir::isa<mlir::relalg::AggregationOp>(join_res_user) || 
				mlir::isa<mlir::relalg::MaterializeOp>(join_res_user) ) {
			while (true) {
				old2new[join_res_user] = builder.clone(*join_res_user);
				if (join_res_user->getNumResults() == 0) {
					break;
				}
				assert(join_res_user->getNumResults() == 1);
				auto res_user_res_users = join_res_user->getResult(0).getUsers();
				join_res_user = *res_user_res_users.begin();
			}
			return {};
		}
		// Otherwise, must be consumed by another join
		// Add a materialize op for intermediates 
		else if (mlir::isa<mlir::relalg::InnerJoinOp>(join_res_user)) {	
			llvm::SmallVector<std::string, 8> need_col_names;
			llvm::SmallVector<mlir::Attribute, 8> need_col_refs;
			llvm::SmallVector<mlir::Attribute, 8> need_col_names_attr;
			if (do_update) {
				mlir::relalg::ColumnManager& attr_manager = 
					builder.getContext()->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();

				Operator end_join_op = end_join;
				mlir::relalg::ColumnSet populated_used;
				PopulateUsedColumns(join_res_user, populated_used);
				// On old table identifiers
				auto need_cols = populated_used.intersect(end_join_op.getAvailableColumns());
				// need_cols.dump(builder.getContext());
				for (const auto c : need_cols) {
					const auto [old_tbl, col_name] = attr_manager.getName(c);
					need_col_names.push_back(col_name);
					need_col_names_attr.push_back(builder.getStringAttr(col_name));
					if (!col2res.contains(col_name)) { // first occur, assign to old table
						col2res[col_name] = old_tbl;
					}
				}
				need_col_refs = CreateColRefs(attr_manager, col2res, need_col_names);
			}

			auto new_last_join = llvm::dyn_cast<mlir::relalg::InnerJoinOp>(old2new[end_join]);
			auto mater_op = builder.create<mlir::relalg::MaterializeOp>(builder.getUnknownLoc(),
				mlir::dsa::TableType::get(builder.getContext()), new_last_join.result(),
				builder.getArrayAttr(need_col_refs), builder.getArrayAttr(need_col_names_attr)
			);
			if (do_update) {
				// Note the columns are now in new result
				for (auto col_name : need_col_names) {
					col2res[col_name] = res_name;
				}
			}
			mater_op.getOperation()->setAttr("table_identifier", builder.getStringAttr(res_name));
			builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), mater_op.result());
			return need_col_refs;
		}
		else {
			// assert(mlir::isa<mlir::loop::Shuffle>(join_res_user));
			continue;
		}
	}
	return {};
}

mlir::Type ToMLIRType(mlir::MLIRContext* mlir_ctxt, std::string type) {
	if (type == "i32") {
			return mlir::IntegerType::get(mlir_ctxt, 32);
	}
	else if (type == "string") {
			return mlir::db::StringType::get(mlir_ctxt);
	}
	else if (type == "decimal128") {
			return mlir::db::DecimalType::get(mlir_ctxt, 18, 2);
	}
	else if (type == "fixed_bin") {
			return mlir::db::CharType::get(mlir_ctxt, 7);
	}
	else {
			assert(false);
	}
}

llvm::SmallVector<mlir::NamedAttribute, 8> 
CreateNamedColDef(mlir::OpBuilder& builder,
									mlir::relalg::ColumnManager& attr_manager, std::string tbl_name,
									const llvm::SmallVector<mlir::Attribute, 8>& col_refs) {
	llvm::SmallVector<mlir::NamedAttribute, 8> cols;
	for (auto col_ref0 : col_refs) {
		auto col_ref = col_ref0.dyn_cast_or_null<mlir::relalg::ColumnRefAttr>();
		mlir::relalg::Column& col = col_ref.getColumn();
		auto col_name = attr_manager.getColName(&col);
		auto col_def = attr_manager.createDef(tbl_name, col_name);
		col_def.getColumn().type = col.type;
		cols.push_back(builder.getNamedAttr(col_name, col_def));
	}
	return cols;
}

llvm::SmallVector<mlir::NamedAttribute, 8>
CreateNamedColDef(mlir::OpBuilder& builder,
									mlir::relalg::ColumnManager& attr_manager, std::string tbl_name,
									const std::vector<std::string>& col_names) {
	llvm::SmallVector<mlir::NamedAttribute, 8> cols;
	for (auto col_name : col_names) {
		auto col_def = attr_manager.createDef(tbl_name, col_name);
		// attr_def.getColumn().type = ToMLIRType(mlir_ctxt, col_type);
		col_def.getColumn().type = builder.getI32Type();
		cols.push_back(builder.getNamedAttr(col_name, col_def));
	}
	return cols;
}

void CloneGetCol(mlir::OpBuilder& builder, GetColumnOp getcol_op, mlir::BlockArgument tuple,
								 std::unordered_map<mlir::Operation*, mlir::Value>& old2new_res) {
	auto col_ref_type = getcol_op.attr().getType();
	auto res_type = getcol_op.res().getType();
	mlir::Value new_res = builder.create<mlir::relalg::GetColumnOp>(
		getcol_op.getLoc(), col_ref_type, getcol_op.attr(), tuple
	);
	new_res.setType(res_type);
	old2new_res[getcol_op] = new_res;
}

void CloneConstOp(mlir::OpBuilder& builder, mlir::db::ConstantOp const_op,
								 std::unordered_map<mlir::Operation*, mlir::Value>& old2new_res) {
	mlir::Value new_res = builder.create<mlir::db::ConstantOp>(
		const_op.getLoc(), const_op.value().getType(), const_op.value()
	);
	new_res.setType(const_op.result().getType());
	old2new_res[const_op] = new_res;
}

void CloneBinaryOp(mlir::OpBuilder& builder, mlir::Operation* op,
									 std::unordered_map<mlir::Operation*, mlir::Value>& old2new_res) {
  // assert(mlir::isa<mlir::db::SubOp>(op));
	mlir::Operation* left_def_op = op->getOperand(0).getDefiningOp();
	mlir::Value new_left = old2new_res[left_def_op];
	mlir::Operation* right_def_op = op->getOperand(1).getDefiningOp();
	mlir::Value new_right = old2new_res[right_def_op];
	if (mlir::isa<mlir::db::SubOp>(op)) {
		old2new_res[op] = builder.create<mlir::db::SubOp>(
			op->getLoc(), new_left, new_right
		);
	}
	else if (mlir::isa<mlir::db::MulOp>(op)) {
		old2new_res[op] = builder.create<mlir::db::MulOp>(
			op->getLoc(), new_left, new_right
		);
	}
	else {
		assert(false);
	}
}


}// end namespace relalg
}// end namespace mlir