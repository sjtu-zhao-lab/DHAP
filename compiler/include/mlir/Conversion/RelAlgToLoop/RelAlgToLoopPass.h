#ifndef MLIR_CONVERSION_RELALGTOLOOP_RELALGTOLOOPPASS_H
#define MLIR_CONVERSION_RELALGTOLOOP_RELALGTOLOOPPASS_H

// #include <memory>
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/Loop/IR/LoopOps.h"

#include "runtime/Database.h"

#include <unordered_map>
#include <json.h>

namespace mlir {
namespace relalg {

std::unique_ptr<Pass> createSplitSubqueryPass(llvm::SmallVector<mlir::ModuleOp, 8>& subq_modules);
std::unique_ptr<Pass> createPlanningPass(std::string plan_fname, runtime::Database* db);
std::unique_ptr<Pass> createInsertShflPass(nlohmann::json& plan);

std::unique_ptr<Pass> createSplitForLLVMPass(std::unordered_map<std::string, mlir::ModuleOp>& named_module);
std::unique_ptr<Pass> createUpdateProbeBaseTablePass(nlohmann::json& table_schema);

std::unique_ptr<Pass> createLowerToLoopPass();
std::unique_ptr<Pass> createFuseLoopPass();
std::unique_ptr<Pass> createGeneratePlanPass(std::string subq_id_str, nlohmann::json& plan);
// void registerRelAlgConversionPasses();
// void createLowerRelAlgPipeline(mlir::OpPassManager& pm);

}// end namespace relalg
}// end namespace mlir

#endif // MLIR_CONVERSION_RELALGTODB_RELALGTODBPASS_H