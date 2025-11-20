#include "mlir/Conversion/LoopToGDSA/LoopToGDSAPass.h"

#include "mlir/Conversion/LoopToGDSA/TranslateLoop.h"

namespace {

class LowerToGDSAPass
	: public mlir::PassWrapper<LowerToGDSAPass, mlir::OperationPass<mlir::func::FuncOp>> {
	virtual llvm::StringRef getArgument() const override { return "to-gdsa"; }

   void runOnOperation() override {
		mlir::loop::LoopTranslator translator;
		getOperation().walk([&](mlir::Operation* op) {
				if (llvm::dyn_cast<mlir::loop::ForOp>(op)) {
					mlir::OpBuilder builder(op);
					// builder.setInsertionPointAfter(op);
					translator.LoopToWhile(op, builder);
					// op->erase();
				}
		});
   }
};

}


namespace mlir {
namespace loop {

std::unique_ptr<Pass> createLowerToGDSAPass() { return std::make_unique<LowerToGDSAPass>(); }

}
}