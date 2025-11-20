#include "mlir/Conversion/RelAlgToLoop/RelAlgToLoopPass.h"
#include "mlir/Dialect/RelAlg/IR/util.h"
#include <fstream>

namespace {

class GeneratePlanPass 
	: public mlir::PassWrapper<GeneratePlanPass, mlir::OperationPass<mlir::func::FuncOp>> {
	virtual llvm::StringRef getArgument() const override { return "generate-plan"; }

public:
	GeneratePlanPass(std::string subq_id_str, nlohmann::json& plan) 
		: subq_id_str_(subq_id_str), plan_(plan) {}

private:
	std::string subq_id_str_;
	nlohmann::json& plan_;
	std::vector<nlohmann::json> stage_kernel_info_;
	std::vector<std::vector<std::string>> stage_request_table_;
	std::vector<std::vector<bool>> stage_request_table_from_shfl_;
	nlohmann::json shfl_worker_partition_col_;
	nlohmann::json shfl_target_stage_;
	nlohmann::json table_schema_;

	void runOnOperation() override {
		auto& attr_manager = getContext().getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();

		mlir::Block& func_blk = getOperation().getRegion().front();
		const int max_stage = 10;
		std::vector<mlir::Operation*> stage_probe_for(max_stage, NULL);
		std::unordered_map<mlir::relalg::Column*, mlir::Operation*> build_key_2_for;
		std::vector<std::vector<mlir::Operation*>> stage_build_fors(max_stage);
		int num_stages = 0;
		// First pass to sort stages
		for (auto for_op : func_blk.getOps<mlir::loop::ForOp>()) {
			mlir::Operation* op = for_op.getOperation();
			int stage = 0;
			if (op->hasAttr("stage")) {
				stage = op->getAttr("stage").dyn_cast_or_null<mlir::IntegerAttr>().getInt();
				if (stage > num_stages) {
					num_stages = stage;
				}
			}
			// int type = for_op.type().getInt();
			if (for_op.type() == 1) {
				stage_probe_for[stage] = op;
			}
			else if (for_op.type() == 0) {
				assert(op->hasAttr("build_keys"));
				auto build_keys = op->getAttr("build_keys").dyn_cast_or_null<mlir::ArrayAttr>();
				for (const auto attr : build_keys) {
					auto build_key = mlir::relalg::getColumnFromAttr(attr);
					build_key_2_for[build_key] = op;
				}
			}
		}
		std::string final_res_name = "final";
		for (int s = 0; s < num_stages + 1; s++) {
			nlohmann::json knl_info = nlohmann::json::object();
			std::vector<std::string> req_table;
			std::vector<bool> table_from_shfl;
			if (s == 0) {
				stage_kernel_info_.push_back(knl_info);
				stage_request_table_.push_back(req_table);
				stage_request_table_from_shfl_.push_back(table_from_shfl);
				continue;
			}
			mlir::Operation* op = stage_probe_for[s];
			if (op->hasAttr("computed_cols")) {
				auto array_attr = op->getAttr("computed_cols").dyn_cast_or_null<mlir::ArrayAttr>();
				std::unordered_map<std::string, std::vector<std::string>> computed_cols;
				for (auto const attr : array_attr) {
					mlir::relalg::Column* col 
						= attr.dyn_cast_or_null<mlir::relalg::ColumnDefAttr>().getColumnPtr().get();
					auto [table, col_name] = attr_manager.getName(col);
					computed_cols[table].push_back(col_name);
				}
				knl_info["computed_cols"] = computed_cols;
			}
			auto probe_for = llvm::dyn_cast<mlir::loop::ForOp>(op);
			// Add partition_col as attribute for kernel generation
			if (s < num_stages) {
				mlir::Operation* next_probe = stage_probe_for[s+1];
				auto next_probe_keys = next_probe->getAttr("probe_keys").dyn_cast_or_null<mlir::ArrayAttr>();
				assert(!next_probe_keys.empty());
				auto partition_col_attr = next_probe_keys.getValue()[0];
				op->setAttr("partition_col", partition_col_attr);
			}
			// Probe info
			knl_info["result_name"] = op->getAttr("name_res").dyn_cast<mlir::StringAttr>().str();
			for (mlir::Operation* probe_res_user : op->getUsers()) {
				// for the final result of current subquery
				if (probe_res_user->hasAttr("table_identifier")) {
					assert(mlir::isa<mlir::relalg::MaterializeOp>(probe_res_user));
					knl_info["result_name"] = probe_res_user->getAttr("table_identifier").dyn_cast<mlir::StringAttr>().str();
				}
			}
			if (s == num_stages) {
				if (knl_info["result_name"] != "") {
					final_res_name = knl_info["result_name"];
				}
				else {
					knl_info["result_name"] = final_res_name;
				}
			}
			knl_info["probe_knl"] = "subq"+subq_id_str_+"_probe" + std::to_string(s);
			auto probe_keys = mlir::relalg::getOpAttrColNames(op, "probe_keys", attr_manager);
			knl_info["probe_key"] = probe_keys;
			auto probe_payloads = mlir::relalg::getOpAttrColNames(op, "probe_payloads", attr_manager);
			auto probe_payloads_type0 = mlir::relalg::getOpAttrColTypes(op, "probe_payloads");
			knl_info["probe_payload"] = probe_payloads;
			std::string probe_inp_name = probe_for.name_inp().str();
			req_table.push_back(probe_inp_name);
			// If the probe table (input of probe for) is shuffled or not 
			bool probe_inp_shlfed = false;
			for (mlir::Operation* probe_inp_user : probe_for.table().getUsers()) {
				if (llvm::dyn_cast<mlir::loop::Shuffle>(probe_inp_user)) {
					probe_inp_shlfed = true;
					break;
				}
			}
			if (probe_inp_shlfed) {
				table_from_shfl.push_back(true);
				// Only the probe table of the first stage is shuffled by shuffle worker
				if (s == 1) {
					shfl_target_stage_[probe_inp_name] = 1;
					shfl_worker_partition_col_[probe_inp_name] = probe_keys[0];

					// Output the schema of the first probe table
					auto op_req_cols_name = mlir::relalg::getOpAttrColNames(op, "req_cols", attr_manager);
					std::vector<std::string> op_req_cols_type(op_req_cols_name.size(), "i32");
					auto op_req_cols_type0 = mlir::relalg::getOpAttrColTypes(op, "req_cols");
					table_schema_[probe_inp_name]["name"] = op_req_cols_name;
					table_schema_[probe_inp_name]["type"] = op_req_cols_type;
					table_schema_[probe_inp_name]["type0"] = op_req_cols_type0;
				}
			}
			else {
				table_from_shfl.push_back(false);
			}
			// Build info
			std::vector<std::string> build_knl, build_key, build_filter;
			std::vector<std::vector<std::string>> build_payload;
			std::vector<std::vector<std::string>> build_payload_type0;
			std::vector<mlir::relalg::Column*> build_keys_col = mlir::relalg::getOpAttrCols(op, "build_keys");
			for (auto col : build_keys_col) {
				auto [table, col_name] = attr_manager.getName(col);
				build_knl.push_back("subq"+subq_id_str_+"_build_"+table);
				build_key.push_back(col_name);
				mlir::Operation* build_for_op = build_key_2_for[col];
				build_payload.push_back(mlir::relalg::getOpAttrColNames(build_for_op, "build_payloads", attr_manager));
				build_payload_type0.push_back(mlir::relalg::getOpAttrColTypes(build_for_op, "build_payloads"));
				auto build_for = llvm::dyn_cast<mlir::loop::ForOp>(build_for_op);	
				if (!build_for_op->hasAttr("filter")) {
					build_filter.push_back("");
				}
				else {
					auto filter = mlir::relalg::getOpAttrColNames(build_for_op, "filter", attr_manager);
					assert(filter.size() == 1 && "each build loop has at most 1 filter");
					build_filter.push_back(filter[0]);
				}
				std::string build_inp_name = build_for.name_inp().str();
				req_table.push_back(build_inp_name);
				// If the build table (input of build for) is shuffled or not 
				bool build_tbl_shlfed = false;
				for (mlir::Operation* build_tbl_user : build_for.table().getUsers()) {
					if (llvm::dyn_cast<mlir::loop::Shuffle>(build_tbl_user)) {
						build_tbl_shlfed = true;	
						break;
					}
				}
				if (build_tbl_shlfed) {
					table_from_shfl.push_back(true);
					shfl_target_stage_[build_inp_name] = s;
					shfl_worker_partition_col_[build_inp_name] = col_name;
				}
				else {
					table_from_shfl.push_back(false);
				}
				auto build_col_names = mlir::relalg::getOpAttrColNames(build_for_op, "req_cols", attr_manager);
				std::vector<std::string> build_col_types(build_col_names.size(), "i32");
				auto build_col_types0 = mlir::relalg::getOpAttrColTypes(build_for_op, "req_cols");
				table_schema_[build_inp_name]["name"] = build_col_names;
				table_schema_[build_inp_name]["type"] = build_col_types;
				table_schema_[build_inp_name]["type0"] = build_col_types0;
			}
			knl_info["build_knl"] = build_knl;
			knl_info["build_key"] = build_key;
			knl_info["build_filter"] = build_filter;
			knl_info["build_payload"] = build_payload;
			// Output the table schema of probe result in the middle stages
			std::string probe_res_name = probe_for.name_res().str();
			if (probe_res_name != "" && s < num_stages) {
				std::vector<std::string> probe_res_col_names = probe_payloads;
				std::vector<std::string> probe_res_col_type0 = probe_payloads_type0;
				for (auto& build_plds_v : build_payload) {
					for (auto build_pld : build_plds_v) {
						probe_res_col_names.push_back(build_pld);
					}
				}
				for (auto& build_pld_type_v : build_payload_type0) {
					for (auto build_pld_type : build_pld_type_v) {
						probe_res_col_type0.push_back(build_pld_type);
					}
				}
				std::vector<std::string> probe_res_col_types(probe_res_col_names.size(), "i32");
				table_schema_[probe_res_name]["name"] = probe_res_col_names;
				table_schema_[probe_res_name]["type"] = probe_res_col_types;
				table_schema_[probe_res_name]["type0"] = probe_res_col_type0;
			}
			// Aggr Info
			std::vector<std::string> aggr_col, groupby_build, groupby_probe;
			if (op->hasAttr("aggr_col")) {
				assert(s == num_stages && "Aggregation can only be in the last stage");
				knl_info["aggr_col"] = mlir::relalg::getOpAttrColNames(op, "aggr_col", attr_manager);
				knl_info["groupby_build"] = mlir::relalg::getOpAttrColNames(op, "groupby_keys_build", attr_manager);
				knl_info["groupby_probe"] = mlir::relalg::getOpAttrColNames(op, "groupby_keys_probe", attr_manager);
				knl_info["groupby"] = mlir::relalg::getOpAttrColNames(op, "groupby_keys", attr_manager);
			}

			stage_kernel_info_.push_back(knl_info);
			stage_request_table_.push_back(req_table);
			stage_request_table_from_shfl_.push_back(table_from_shfl);
		}
		// Output the table schema of the final result (from aggregation or the last probe)
		for (auto mater_op : func_blk.getOps<mlir::relalg::MaterializeOp>()) {
			std::vector<std::string> res_col_names;
			for (auto attr : mater_op.columns()) {
				auto str_attr = attr.dyn_cast<mlir::StringAttr>();
				assert(str_attr);
				res_col_names.push_back(str_attr.getValue().str());
			}
			GetResAggrColNames(res_col_names);

			std::vector<std::string> res_col_types(res_col_names.size(), "i32");
			table_schema_[final_res_name]["name"] = res_col_names;
			table_schema_[final_res_name]["type"] = res_col_types;
			table_schema_[final_res_name]["type0"] = mlir::relalg::getOpAttrColTypes(
				mater_op.getOperation(), "cols"
			);

			ProcessSort(mater_op);
		}
		stage_request_table_.push_back({final_res_name});
		stage_request_table_from_shfl_.push_back({true});

		// Finalize the plan
		plan_["shfl_worker_partition_col"] = shfl_worker_partition_col_;
		plan_["shfl_target_stage"] = shfl_target_stage_;
		plan_["stage_request_table"] = stage_request_table_;
		plan_["request_table_from_shfl"] = stage_request_table_from_shfl_;
		plan_["stage_kernel_info"] = stage_kernel_info_;
		plan_["table_schema"] = table_schema_;
	}

	void GetResAggrColNames(const std::vector<std::string> res_col_names) {
		auto& last_knl = stage_kernel_info_[stage_kernel_info_.size()-1];
		if (last_knl.contains("groupby")) {
			std::vector<std::string> res_aggr_col;
			const std::vector<std::string> all_gb_keys = last_knl.at("groupby");
			// Extract the name of result aggregated column simply
			for (auto res_col : res_col_names) {
				if (std::find(all_gb_keys.begin(), all_gb_keys.end(), res_col) == all_gb_keys.end()) {
					res_aggr_col.push_back(res_col);
				}
			}
			last_knl["aggr_res_col"] = res_aggr_col;
		}
	}
	void ProcessSort(mlir::relalg::MaterializeOp mater_op) {
		if (auto sort_op = llvm::dyn_cast<mlir::relalg::SortOp>(
			mater_op.rel().getDefiningOp()
		)) {
			// To map mater column names
			std::unordered_map<mlir::relalg::Column*, std::string> mater_col2name;
			std::vector<mlir::relalg::Column*> mater_cols;
			for (const auto attr : mater_op.cols()) {
				auto col_ref = attr.dyn_cast_or_null<mlir::relalg::ColumnRefAttr>();
				mater_cols.push_back(&col_ref.getColumn());
			}
			for (int c = 0; c < mater_cols.size(); c++) {
				mater_col2name[mater_cols[c]] = mater_op.columns()[c].cast<mlir::StringAttr>().str();
			}
			auto& last_knl = stage_kernel_info_[stage_kernel_info_.size()-1];
			std::vector<std::string> sort_key_names;
			std::vector<int> sort_orders;
			for (const auto sort_attr : sort_op.sortspecs()) {
				auto sort_spec = sort_attr.cast<mlir::relalg::SortSpecificationAttr>();
				// auto [_, col_name] = attr_manager.getName(&sort_spec.getAttr().getColumn());
				sort_key_names.push_back(mater_col2name.at(&sort_spec.getAttr().getColumn()));
				if (sort_spec.getSortSpec() == mlir::relalg::SortSpec::desc) {
					sort_orders.push_back(0);
				}
				else {
					sort_orders.push_back(1);
				}
			}
			last_knl["sort"]["name"] = sort_key_names;
			last_knl["sort"]["order"] = sort_orders;
		}
	}
};

}

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createGeneratePlanPass(std::string subq_id_str, nlohmann::json& plan) {
	return std::make_unique<GeneratePlanPass>(subq_id_str, plan);
}
}
}