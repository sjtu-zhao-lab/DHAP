#include "mlir/Conversion/RelAlgToLoop/RelAlgToLoopPass.h"
#include "mlir/Conversion/RelAlgToLoop/UpdateCols.h"
#include "mlir/Dialect/RelAlg/IR/util.h"

class SplitForLLVM 
	: public mlir::PassWrapper<SplitForLLVM, mlir::OperationPass<mlir::func::FuncOp>> {
	virtual llvm::StringRef getArgument() const override { return "split llvm"; }

public:
   SplitForLLVM(std::unordered_map<std::string, mlir::ModuleOp>& named_module)
   : named_module_(named_module) {}

private:
	std::unordered_map<std::string, mlir::ModuleOp>& named_module_;

	void SplitModulesFromJoins(mlir::OpBuilder& builder, std::vector<mlir::relalg::InnerJoinOp> joins, int last_join_id) {
		Operator first_join_this_module = joins[0];
		Operator last_join_this_module = joins[joins.size()-1];
		std::unordered_map<mlir::Operation*, mlir::Operation*> old2new;
		// Copy joins to new module
		mlir::relalg::CloneChildrenUntil(old2new, builder, last_join_this_module, 
			[&](Operator op) {
				auto first_join_children = first_join_this_module.getChildren();
				for (auto& c : first_join_children) {
					if (op == c) return true;
				}
				return false;
			});
		// Process the first join
		mlir::Operation* left_def_op = joins[0].left().getDefiningOp();
		mlir::relalg::CloneChildrenUntil(old2new, builder, left_def_op, 
																		[&](Operator op){ return false; });
		mlir::Operation* right_def_op = joins[0].right().getDefiningOp();
		if (llvm::dyn_cast<mlir::relalg::BaseTableOp>(right_def_op)) {
			old2new[right_def_op] = builder.clone(*right_def_op);
		}
		else {	// Probe from intermediates
			// Get the root basetable of probe table
			mlir::Operation* right_def_op0 = right_def_op;
			while (!llvm::dyn_cast<mlir::relalg::BaseTableOp>(right_def_op)) {
				auto right_def_join = llvm::dyn_cast<mlir::relalg::InnerJoinOp>(right_def_op);
				assert(right_def_join && "join probe table is either from basetable or join");
				right_def_op = right_def_join.right().getDefiningOp();
			}
			auto probe_basetable = llvm::dyn_cast<mlir::relalg::BaseTableOp>(right_def_op);
			std::string probe_table_name = probe_basetable.table_identifier().str();
			// Build an empty basetable op and update it later
			builder.setInsertionPointToStart(builder.getInsertionBlock());
			std::vector<mlir::NamedAttribute> columns;
			old2new[right_def_op0] = builder.create<mlir::relalg::BaseTableOp>(builder.getUnknownLoc(),
				mlir::relalg::TupleStreamType::get(builder.getContext()), "join_"+std::to_string(last_join_id-joins.size()), 
				mlir::relalg::TableMetaDataAttr::get(builder.getContext(), std::make_shared<runtime::TableMetaData>()),
				builder.getDictionaryAttr(columns)
			);
			old2new[right_def_op0]->setAttr("old_table", builder.getStringAttr(probe_table_name));
		}

		// Process the last join in the stage
		const std::string res_name = "join_" + std::to_string(last_join_id);
		std::unordered_map<std::string, std::string> no;
		mlir::relalg::ProcessEndJoin(old2new, res_name, builder, joins.back(), false, no);
		// Update operands
		mlir::relalg::UpdateCreatedOperands(old2new);
	}
	
	void runOnOperation() override {
		auto func_type = getOperation().getFunctionType();
		int curr_stage = 0;
		std::vector<mlir::relalg::InnerJoinOp> joins_same_module;
		auto joins = getOperation()->getRegion(0).front().getOps<mlir::relalg::InnerJoinOp>();
		int join_id = 0;
		for (auto join_op : joins) {
			joins_same_module.push_back(join_op);
			for (auto join_op_user : join_op.result().getUsers()) {
				if (llvm::dyn_cast<mlir::loop::Shuffle>(join_op_user)) {
					curr_stage += 1;
					mlir::OpBuilder builder(&getContext());
					mlir::ModuleOp module_op = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
					std::string module_name = "stage"+std::to_string(curr_stage);
					named_module_[module_name] = module_op;
					builder.setInsertionPointToStart(module_op.getBody());
					mlir::func::FuncOp func_op = builder.create<mlir::func::FuncOp>(
						builder.getUnknownLoc(), "main", func_type
					);
					builder.createBlock(&func_op.getBody());
					SplitModulesFromJoins(builder, joins_same_module, join_id+1);
					joins_same_module.clear();
					break;
				}
			}
			join_id++;
		}
		if (joins_same_module.size()) {
			curr_stage += 1;
			mlir::OpBuilder builder(&getContext());
			mlir::ModuleOp module_op = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
			std::string module_name = "stage"+std::to_string(curr_stage);
			named_module_[module_name] = module_op;
			builder.setInsertionPointToStart(module_op.getBody());
			mlir::func::FuncOp func_op = builder.create<mlir::func::FuncOp>(
				builder.getUnknownLoc(), "main", func_type
			);
			mlir::Block* func_blk = builder.createBlock(&func_op.getBody());
			builder.setInsertionPointToStart(func_blk);
			SplitModulesFromJoins(builder, joins_same_module, join_id);
			joins_same_module.clear();
		}
  }
   
};


class UpdateProbeBaseTable
	: public mlir::PassWrapper<UpdateProbeBaseTable, mlir::OperationPass<mlir::func::FuncOp>> {
	virtual llvm::StringRef getArgument() const override { return "update probe table"; }

public:
  UpdateProbeBaseTable(nlohmann::json& table_schema) 
		: table_schema_(table_schema) {}

private:
	nlohmann::json& table_schema_;
	std::string new_table_name;
	std::vector<std::string> new_col_names;
	std::unordered_map<std::string, mlir::relalg::ColumnRefAttr> col_name2attr;
	
	void UpdateMaterialize(mlir::OpBuilder& builder, mlir::relalg::ColumnManager& attr_manager, mlir::relalg::MaterializeOp mater_op) {
		mlir::Operation* mater_op0 = mater_op.getOperation();
		mlir::ArrayAttr old_cols_attr = mater_op.cols();
		llvm::SmallVector<mlir::Attribute, 8> new_cols_attr;
		for (auto const col_attr : old_cols_attr) {
			auto col = mlir::relalg::getColumnFromAttr(col_attr);
			auto [table, col_name] = attr_manager.getName(col);
			bool updated = false;
			for (auto new_col_name : new_col_names) {
				if (new_col_name == col_name) {
					updated = true;
					auto new_col_attr = attr_manager.createRef(new_table_name, col_name);
					new_cols_attr.push_back(new_col_attr);
					break;
				}
			}
			if (!updated) {
				new_cols_attr.push_back(col_attr);
			}
		}
		if (new_cols_attr.size() != 0) {
			mater_op0->setAttr("cols", builder.getArrayAttr(new_cols_attr));
		}
		else {	// An empty materialize op that need update col attrs and names
			assert(mater_op0->hasAttr("table_identifier"));
			std::string to_mater_table 
				= mater_op0->getAttr("table_identifier").dyn_cast<mlir::StringAttr>().str();
			assert(table_schema_.find(to_mater_table) != table_schema_.end());
			std::vector<std::string> to_mater_cols = table_schema_[to_mater_table]["name"];
			llvm::SmallVector<mlir::Attribute, 8> col_names_attr;
			llvm::SmallVector<mlir::Attribute, 8> cols_attr;
			for (auto to_mater_col : to_mater_cols) {
				col_names_attr.push_back(builder.getStringAttr(to_mater_col));
				cols_attr.push_back(col_name2attr[to_mater_col]);
			}
			mater_op0->setAttr("columns", builder.getArrayAttr(col_names_attr));
			mater_op0->setAttr("cols", builder.getArrayAttr(cols_attr));
		}
	}

	void runOnOperation() override {
		col_name2attr.clear();		// is module/function local

		mlir::MLIRContext* mlir_ctxt = &getContext();
		mlir::relalg::ColumnManager& attr_manager = mlir_ctxt->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();;
		mlir::OpBuilder builder(mlir_ctxt);
		// Add column and type to empty basetable
		auto basetables = getOperation()->getRegion(0).front().getOps<mlir::relalg::BaseTableOp>();
		int num_empty_basetable = 0;
		for (auto basetable_op : basetables) {
			auto old_columns = basetable_op.columns();
			if (old_columns.size() == 0) {		// Empty basetable for intermediates
				assert(basetable_op.getOperation()->hasAttr("old_table"));
				new_table_name = basetable_op.table_identifier().str();
				assert(table_schema_.find(new_table_name) != table_schema_.end());
				std::vector<std::string> col_names = table_schema_[new_table_name]["name"];
				new_col_names = col_names;
				// std::vector<std::string> col_types = table_schema_[new_table_name]["type"];
				std::vector<mlir::NamedAttribute> new_columns;
				for (uint32_t c = 0; c < col_names.size(); c++) {
					std::string col_name = col_names[c];
					// std::string col_type = col_types[c];
					auto attr_def = attr_manager.createDef(new_table_name, col_name);
					// attr_def.getColumn().type = ToMLIRType(mlir_ctxt, col_type);
					attr_def.getColumn().type = builder.getI32Type();
					new_columns.push_back(builder.getNamedAttr(col_name, attr_def));
				}
				basetable_op.getOperation()->setAttr("columns", builder.getDictionaryAttr(new_columns));
				num_empty_basetable++;
			}
			else {
				std::string table_name = basetable_op.table_identifier().str();
				assert(table_schema_.find(table_name) != table_schema_.end());
				std::vector<std::string> col_names = table_schema_[table_name]["name"];
				std::vector<mlir::NamedAttribute> new_columns;
				for (uint32_t c = 0; c < col_names.size(); c++) {
					std::string col_name = col_names[c];
					auto attr_def = attr_manager.createDef(table_name, col_name);
					// set types of all columns to int32
					attr_def.getColumn().type = builder.getI32Type();
					new_columns.push_back(builder.getNamedAttr(col_name, attr_def));
				}
				basetable_op.getOperation()->setAttr("columns", builder.getDictionaryAttr(new_columns));
			}
			// Update the map from column name to ColumnRefAttr
			auto base_cols = basetable_op.columns();
			for (const mlir::NamedAttribute base_col_named_attr : base_cols) {
				std::string base_col_name = base_col_named_attr.getName().str();
				mlir::Attribute base_col_attr = base_col_named_attr.getValue();
				mlir::relalg::Column* base_col 
					= base_col_attr.dyn_cast_or_null<mlir::relalg::ColumnDefAttr>().getColumnPtr().get();
				assert(col_name2attr.find(base_col_name) == col_name2attr.end() &&
								"homonymous columns cannot exist in 1 splitted function");
				col_name2attr[base_col_name] = attr_manager.createRef(base_col);
			}
		}
		assert(num_empty_basetable <= 1);
		// Update ops that uses column attr in this function
		auto maters = getOperation()->getRegion(0).front().getOps<mlir::relalg::MaterializeOp>();
		for (auto mater_op : maters) {
			UpdateMaterialize(builder, attr_manager, mater_op);
		}
		mlir::relalg::ColumnUpdater updater(new_table_name, new_col_names, true);
		getOperation().walk([&](mlir::Operation* op){
			updater.Update(op, builder, attr_manager);
		});
	}
};


namespace mlir {
namespace relalg {

std::unique_ptr<Pass> createSplitForLLVMPass(std::unordered_map<std::string, mlir::ModuleOp>& named_module) {
   return std::make_unique<SplitForLLVM>(named_module);
}
std::unique_ptr<Pass> createUpdateProbeBaseTablePass(nlohmann::json& table_schema) {
   return std::make_unique<UpdateProbeBaseTable>(table_schema);
}

}
}
