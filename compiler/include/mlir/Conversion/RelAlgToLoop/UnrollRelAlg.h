#ifndef MLIR_CONVERSION_RELALGTOLOOP_UNROLLRELALG_H
#define MLIR_CONVERSION_RELALGTOLOOP_UNROLLRELALG_H

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/Loop/IR/LoopOps.h"

#include "mlir/Conversion/RelAlgToDB/HashJoinTranslator.h"

#include <json.h>

namespace mlir {
namespace relalg {

void GetRequiredColumns(mlir::Operation* op, mlir::Operation* consumer,
												std::unordered_map<mlir::Operation*, mlir::relalg::ColumnSet>& op_req_cols);

void Unroll(mlir::Operation* op, mlir::OpBuilder& builder,
						std::unordered_map<mlir::Operation*, mlir::relalg::ColumnSet>& op_req_cols);

} // end namespace relalg
} // end namespace mlir 

#endif
