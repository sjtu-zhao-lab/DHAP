#ifndef MLIR_DIALECT_GDSA_IR_GDSAOPS_H
#define MLIR_DIALECT_GDSA_IR_GDSAOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/GDSA/IR/GDSATypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/GDSA/IR/GDSAOps.h.inc"

#endif