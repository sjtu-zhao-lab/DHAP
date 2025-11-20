#include "mlir/Conversion/LoopToGDSA/LoopToGDSAPass.h"

#include "mlir/Conversion/LoopToGDSA/CUDAGenerator.h"

namespace {

class CUDAGenPass
	: public mlir::PassWrapper<CUDAGenPass, mlir::OperationPass<mlir::func::FuncOp>> {
	virtual llvm::StringRef getArgument() const override { return "cuda-gen"; }
public:
	CUDAGenPass(std::string subq_id_str) : subq_id_str_(subq_id_str) {}
private:
	std::string subq_id_str_;
	void runOnOperation() override {
		mlir::Block& func_blk = getOperation().getRegion().front();

		int max_stage = 0;
		for (auto while_op : func_blk.getOps<mlir::gdsa::While>()) {
			mlir::Operation* while_op0 = while_op.getOperation();
			// assert(while_op0->hasAttr("stage"));
			if (while_op0->hasAttr("stage")) 
			{
				int stage_id = while_op0->getAttr("stage").dyn_cast<mlir::IntegerAttr>().getInt();
				if (stage_id > max_stage) {
					max_stage = stage_id;
				}
			}
		}
		mlir::gdsa::CUDAGenerator generator(subq_id_str_, getContext(), "kernel.cu", max_stage);
		for (auto while_op : func_blk.getOps<mlir::gdsa::While>()) {
			int while_type = while_op.getOperation()->getAttr("type").dyn_cast<mlir::IntegerAttr>().getValue().getSExtValue();
			if (while_type == 0) {
				generator.GenerateBuildKnl(while_op);
			}
			else if (while_type == 1) {
				generator.GenerateProbeKnl(while_op);
			}
		}
	}
};

}

namespace mlir {
namespace gdsa {

std::unique_ptr<Pass> createCUDAGenPass(std::string subq_id_str) {
	return std::make_unique<CUDAGenPass>(subq_id_str); 
}

}
}