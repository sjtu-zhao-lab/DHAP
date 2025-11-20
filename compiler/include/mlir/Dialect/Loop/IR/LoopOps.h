#ifndef MLIR_DIALECT_LOOP_IR_LOOPOPS_H
#define MLIR_DIALECT_LOOP_IR_LOOPOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/Loop/IR/LoopTypes.h"
// #include "mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/Loop/IR/LoopOps.h.inc"

#endif