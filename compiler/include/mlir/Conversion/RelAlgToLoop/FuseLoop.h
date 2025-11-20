#ifndef MLIR_CONVERSION_RELALGTODB_FUSELOOP_H
#define MLIR_CONVERSION_RELALGTODB_FUSELOOP_H

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/Loop/IR/LoopOps.h"

namespace mlir {
namespace relalg {

bool TryFuse(mlir::Operation* op, mlir::OpBuilder& builder);

} // end namespace relalg
} // end namespace mlir 

#endif