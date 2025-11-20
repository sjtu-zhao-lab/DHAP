#include "mlir/Dialect/Loop/IR/LoopTypes.h"
#include "mlir/Dialect/Loop/IR/LoopDialect.h"
// #include "mlir/Dialect/Loop/IR/LoopOpsEnums.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include <llvm/ADT/TypeSwitch.h>

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Loop/IR/LoopOpsTypes.cpp.inc"
namespace mlir::loop {
void LoopDialect::registerTypes() {
   addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Loop/IR/LoopOpsTypes.cpp.inc"
      >();
}

} // namespace mlir::loop
