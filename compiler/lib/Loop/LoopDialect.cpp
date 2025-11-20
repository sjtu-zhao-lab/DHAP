#include "mlir/Dialect/Loop/IR/LoopDialect.h"
#include "mlir/Dialect/Loop/IR/LoopOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::loop;

void LoopDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Loop/IR/LoopOps.cpp.inc"
      >();
   // addInterfaces<LoopInlinerInterface>();
   registerTypes();
}
#include "mlir/Dialect/Loop/IR/LoopOpsDialect.cpp.inc"


