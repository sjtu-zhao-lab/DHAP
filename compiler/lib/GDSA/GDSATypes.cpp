#include "mlir/Dialect/GDSA/IR/GDSATypes.h"
#include "mlir/Dialect/GDSA/IR/GDSADialect.h"
// #include "mlir/Dialect/GDSA/IR/GDSAOpsEnums.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include <llvm/ADT/TypeSwitch.h>

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/GDSA/IR/GDSAOpsTypes.cpp.inc"
namespace mlir::gdsa {
void GDSADialect::registerTypes() {
   addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/GDSA/IR/GDSAOpsTypes.cpp.inc"
      >();
}

} // namespace mlir::GDSA
