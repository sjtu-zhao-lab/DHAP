
#include "mlir/Conversion/RelAlgToLoop/RelAlgToLoopPass.h"
#include "mlir/Conversion/RelAlgToLoop/UnrollRelAlg.h"
#include "mlir/Conversion/RelAlgToLoop/InsertShfl.h"
#include "mlir/Conversion/RelAlgToLoop/FuseLoop.h"
#include "mlir/Conversion/RelAlgToLoop/Planner.h"

namespace{

class PlanningPass
	: public mlir::PassWrapper<PlanningPass, mlir::OperationPass<mlir::func::FuncOp>> {
	virtual llvm::StringRef getArgument() const override { return "planning"; }

   std::string plan_fname_;
   runtime::Database& db;
public:
   PlanningPass(std::string plan_fname, runtime::Database& db) 
      : plan_fname_(plan_fname), db(db) {}
   void runOnOperation() override {
      std::unordered_map<mlir::Operation*, mlir::relalg::ColumnSet> op_req_cols;
      getOperation().walk([&](mlir::Operation* op) {
         if (llvm::dyn_cast<mlir::relalg::MaterializeOp>(op)) {
            mlir::relalg::GetRequiredColumns(op, nullptr, op_req_cols);
         }
      });
      std::vector<mlir::relalg::InnerJoinOp> join_vec;
      auto joins = getOperation()->getRegion(0).front().getOps<mlir::relalg::InnerJoinOp>();
      for (mlir::relalg::InnerJoinOp join_op : joins) {
         join_vec.push_back(join_op);
      }
      mlir::relalg::Planner planner(plan_fname_, db, join_vec, op_req_cols);
      planner.dump();
      planner.act();
   }
};

class InsertShflPass
	: public mlir::PassWrapper<InsertShflPass, mlir::OperationPass<mlir::func::FuncOp>> {
	virtual llvm::StringRef getArgument() const override { return "insert-shfl"; }

   nlohmann::json& plan_;
public:
   InsertShflPass(nlohmann::json& plan) : plan_(plan) {}
   void runOnOperation() override {
      std::vector<mlir::relalg::InnerJoinOp> join_vec;
      auto joins = getOperation()->getRegion(0).front().getOps<mlir::relalg::InnerJoinOp>();
      for (mlir::relalg::InnerJoinOp join_op : joins) {
         join_vec.push_back(join_op);
      }
      InsertShflBtwJoins(plan_, join_vec);
      // ReplaceJoinProbeTableIds(join_vec);
   }
};

class LowerToLoopPass 
	: public mlir::PassWrapper<LowerToLoopPass, mlir::OperationPass<mlir::func::FuncOp>> {
	virtual llvm::StringRef getArgument() const override { return "relalg-to-loop"; }

   bool isUnrollable(mlir::Operation* op) {
      return llvm::TypeSwitch<mlir::Operation*, bool>(op)
         .Case<mlir::relalg::SelectionOp>([&](mlir::Operation* op) {
            return true;
         })
         .Case<mlir::relalg::InnerJoinOp>([&](mlir::Operation* op) {
            return true;
         })
         .Case<mlir::relalg::AggregationOp>([&](mlir::Operation* op) {
            return true;
         })
         .Case<mlir::relalg::MapOp>([&](mlir::Operation* op) {
            return true;
         })
         .Default([&](auto op) {
            return false;
         });
   }

   void runOnOperation() override {
      // Back-propogate from `materialize` to get the required cols of each op
      std::unordered_map<mlir::Operation*, mlir::relalg::ColumnSet> op_req_cols;
      getOperation().walk([&](mlir::Operation* op) {
         if (llvm::dyn_cast<mlir::relalg::MaterializeOp>(op)) {
            mlir::relalg::GetRequiredColumns(op, nullptr, op_req_cols);
         }
      });
      getOperation().walk([&](mlir::Operation* op) {
         if (isUnrollable(op)) {
            mlir::OpBuilder builder(op);
            // builder.setInsertionPointAfter(op);
            // op->dump();
            mlir::relalg::Unroll(op, builder, op_req_cols);
         }
      });
   }
};

class FuseLoopPass 
	: public mlir::PassWrapper<FuseLoopPass, mlir::OperationPass<mlir::func::FuncOp>> {
	virtual llvm::StringRef getArgument() const override { return "fuse-loop"; }

   void runOnOperation() override {
      getOperation().walk([&](mlir::Operation* op) {
         if (llvm::dyn_cast<mlir::loop::ForOp>(op)) {
            mlir::OpBuilder builder(op);
            bool fused = mlir::relalg::TryFuse(op, builder);
            if (fused) {
               op->erase();
            }
         }
      });
   }
};

}

namespace mlir {
namespace relalg {
   
std::unique_ptr<Pass> createPlanningPass(std::string plan_fname, runtime::Database* db) { 
   return std::make_unique<PlanningPass>(plan_fname, *db);
}
std::unique_ptr<Pass> createInsertShflPass(nlohmann::json& plan) {
   return std::make_unique<InsertShflPass>(plan);
}

std::unique_ptr<Pass> createLowerToLoopPass() { return std::make_unique<LowerToLoopPass>(); }
std::unique_ptr<Pass> createFuseLoopPass() { return std::make_unique<FuseLoopPass>(); }

}
}
