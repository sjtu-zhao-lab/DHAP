#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir-support/offloader.h"

class OffloadPass
	: public mlir::PassWrapper<OffloadPass, mlir::OperationPass<mlir::func::FuncOp>> {
	virtual llvm::StringRef getArgument() const override { return "offload"; }
public:
   OffloadPass(int subq_id) : subq_id_(subq_id) {}
private:
   int subq_id_;
   void runOnOperation() override {
      mlir::relalg::Offloader offloader(subq_id_);
      // auto offloader = std::make_unique<mlir::relalg::OffloaderIterface>();
      auto sels = getOperation()->getRegion(0).front().getOps<mlir::relalg::SelectionOp>();
      for (auto sel : sels) {
         auto sel_src = sel.rel().getDefiningOp();
         if (auto basetable = mlir::dyn_cast_or_null<mlir::relalg::BaseTableOp>(sel_src)) {
            offloader.OffloadSel(basetable, sel);
						sel.result().replaceAllUsesWith(basetable.result());
            sel.getOperation()->setAttr("offload", mlir::BoolAttr::get(&getContext(), true));
         }
      }
   }
};


namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createOffloadPass(int subq_id) {
   return std::make_unique<OffloadPass>(subq_id);
}
} // end namespace relalg
} // end namespace mlir