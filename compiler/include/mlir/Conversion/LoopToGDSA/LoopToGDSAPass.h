#ifndef MLIR_CONVERSION_LOOPTOGDSA_LOOPTOGDSAPASS_H
#define MLIR_CONVERSION_LOOPTOGDSA_LOOPTOGDSAPASS_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/Loop/IR/LoopOps.h"
#include "mlir/Dialect/GDSA/IR/GDSAOps.h"
// #include <memory>

namespace mlir {
namespace loop {

std::unique_ptr<Pass> createLowerToGDSAPass();
// void registerRelAlgConversionPasses();
// void createLowerRelAlgPipeline(mlir::OpPassManager& pm);

}// end namespace relalg
}// end namespace mlir


namespace mlir {
namespace gdsa {

std::unique_ptr<Pass> createCUDAGenPass(std::string subq_id);

}// end namespace relalg
}// end namespace mlir

#endif // MLIR_CONVERSION_RELALGTODB_RELALGTODBPASS_H