#include "mlir/Dialect/GDSA/IR/GDSADialect.h"
#include "mlir/Dialect/GDSA/IR/GDSAOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::gdsa;

void GDSADialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/GDSA/IR/GDSAOps.cpp.inc"
      >();
   // addInterfaces<GDSAInlinerInterface>();
   registerTypes();
}
#include "mlir/Dialect/GDSA/IR/GDSAOpsDialect.cpp.inc"


