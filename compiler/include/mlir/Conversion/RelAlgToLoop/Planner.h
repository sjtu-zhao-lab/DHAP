#ifndef MLIR_CONVERSION_RELALGTOLOOP_PLANNER_H
#define MLIR_CONVERSION_RELALGTOLOOP_PLANNER_H

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h"
#include "mlir/Dialect/RelAlg/IR/util.h"

#include "runtime/Database.h"

#include <json.h>
#include <fstream>

namespace mlir {
namespace relalg {

class Planner
{
private:
  std::string plan_fname_;

	runtime::Database& db;
	std::vector<mlir::relalg::InnerJoinOp> joins_;
  std::unordered_map<mlir::Operation*, mlir::relalg::ColumnSet>& op_req_cols_;
	int num_joins_ = 0;
	std::vector<double> build_table_sizes;		// in MB
	std::vector<int> num_join_res_col;
	std::vector<double> num_join_res_row;
	// fisrt probe table
	std::string first_probe_name;
	double first_probe_rows;
	int first_probe_cols;
public:
	Planner(std::string plan_fname, runtime::Database& db,
					std::vector<mlir::relalg::InnerJoinOp> joins,
					std::unordered_map<mlir::Operation*, mlir::relalg::ColumnSet>& op_req_cols);
	~Planner() {};
	void act();
	void dump();
};

Planner::Planner(std::string plan_fname, runtime::Database& db,
								 std::vector<mlir::relalg::InnerJoinOp> joins,
								 std::unordered_map<mlir::Operation*, mlir::relalg::ColumnSet>& op_req_cols)
	: plan_fname_(plan_fname), db(db), joins_(joins), op_req_cols_(op_req_cols)
{
	num_joins_ = joins.size();
	// Get build table and result sizes
	build_table_sizes.resize(num_joins_);
	num_join_res_col.resize(num_joins_);
	num_join_res_row.resize(num_joins_);
	for (int i = 0; i < num_joins_; i++) {
		auto join_op = joins[i];
		if (i == 0) {
			mlir::Operation* probe_def_op = join_op.right().getDefiningOp();
			while (!llvm::dyn_cast<mlir::relalg::BaseTableOp>(probe_def_op)) {
				assert(probe_def_op->getNumOperands() == 1);
				probe_def_op = probe_def_op->getOperand(0).getDefiningOp();
			}
			auto probe_base = mlir::dyn_cast<mlir::relalg::BaseTableOp>(probe_def_op);
			first_probe_name = probe_base.table_identifier().str();
			first_probe_rows = probe_base->getAttr("rows").dyn_cast_or_null<FloatAttr>().getValueAsDouble();
			auto probe_base_users = probe_base.result().getUsers();
			assert(std::distance(probe_base_users.begin(), probe_base_users.end()) == 1);
			for (mlir::Operation* probe_base_user : probe_base_users) {
				mlir::relalg::ColumnSet populated_used;
				PopulateUsedColumns(probe_base_user, populated_used);
				auto need_cols = populated_used.intersect(probe_base.getAvailableColumns());
				first_probe_cols = need_cols.size();
			}
		}
		mlir::Operation* build_def_op = join_op.left().getDefiningOp();
		int num_build_cols = op_req_cols[build_def_op].size();
		while (!llvm::dyn_cast<mlir::relalg::BaseTableOp>(build_def_op)) {
			assert(build_def_op->getNumOperands() == 1);
			build_def_op = build_def_op->getOperand(0).getDefiningOp();
		}
		double num_build_rows = build_def_op->getAttr("rows").dyn_cast_or_null<FloatAttr>().getValueAsDouble();
		build_table_sizes[i] = num_build_cols * num_build_rows * sizeof(int32_t) / (1024*1024);
		// Get result size
		num_join_res_row[i] = join_op->getAttr("rows").dyn_cast_or_null<FloatAttr>().getValueAsDouble();
		auto join_res_users = join_op.result().getUsers();
		assert(std::distance(join_res_users.begin(), join_res_users.end()) == 1);
		for (mlir::Operation* join_res_user : join_res_users) {
			mlir::relalg::ColumnSet populated_used;
			PopulateUsedColumns(join_res_user, populated_used);
			auto need_cols = populated_used.intersect(join_op.getAvailableColumns());
			num_join_res_col[i] = need_cols.size();
		}
	}
}

void Planner::dump()
{
	nlohmann::json plan;
	plan["build_table_sizes_mb"] = build_table_sizes;
	plan["num_join_res_col"] = num_join_res_col;
	plan["num_join_res_row"] = num_join_res_row;
	plan["first_probe"]["name"] = first_probe_name;
	plan["first_probe"]["rows"] = first_probe_rows;
	plan["first_probe"]["cols"] = first_probe_cols;

  for (const auto& table : db.getAllTableNames()) {
		plan["num_batches"][table] = db.getTableMetaData(table)->getNumBatches();
  }
	std::ofstream plan_file(plan_fname_);
	plan_file << std::setw(2) << plan << std::endl;
	plan_file.close();
}

void Planner::act()
{
	const char* repo_cstr = std::getenv("DHAP_REPO");
	if (repo_cstr == nullptr) {
        assert(false && "DHAP_REPO environment variable must be set."); 
    }
	std::string repo(repo_cstr);
	if (!std::filesystem::is_directory(repo)) {
		assert(false && "DHAP_REPO not a valid directory.");
	}
	const std::string cmd 
		= "python " + repo + "/scripts/heuristic.py --plan_path " + plan_fname_;
	FILE* pipe = popen(cmd.c_str(), "r");
	if (!pipe) {
		assert(false && "Failed to open pipe");
	}
	char buffer[128];
	while (!feof(pipe)) {
		if (fgets(buffer, 128, pipe) != NULL)
			std::cout << buffer;
	}
	pclose(pipe);
}



} // end namespace relalg
} // end namespace mlir 

#endif
