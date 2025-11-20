#ifndef MLIR_CONVERSION_RELALGTODB_INSERTSHFL_H
#define MLIR_CONVERSION_RELALGTODB_INSERTSHFL_H

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/Loop/IR/LoopOps.h"

#include "mlir/Conversion/RelAlgToDB/HashJoinTranslator.h"

namespace mlir {
namespace relalg {

// void ReplaceJoinProbeTableIds(std::vector<mlir::relalg::InnerJoinOp> joins);
void InsertShflBtwJoins(nlohmann::json& plan, std::vector<mlir::relalg::InnerJoinOp> joins);
// void SplitModulesFromJoins(mlir::OpBuilder& builder, std::vector<mlir::relalg::InnerJoinOp> joins, int stage);

}
}

#endif