#ifndef MLIR_CONVERSION_LOOPTOGDSA_TRANSLATELOOP_H
#define MLIR_CONVERSION_LOOPTOGDSA_TRANSLATELOOP_H

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
// #include "mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/Loop/IR/LoopOps.h"
#include "mlir/Dialect/GDSA/IR/GDSAOps.h"
#include "mlir/Dialect/GDSA/IR/GDSAOpsInterfaces.h"

// #include "mlir/Conversion/RelAlgToDB/HashJoinTranslator.h"

namespace mlir {
namespace loop {

class LoopTranslator
{
private:
	std::unordered_map<mlir::Operation*, mlir::Value> old2new_res_;
	// For the result of MapFind (index of build tables)
	std::unordered_map<mlir::relalg::Column*, mlir::Value> col2build_idx;
	// build payloads of hash tables probed in the `ForOp`
	std::unordered_map<mlir::Operation*, std::vector<mlir::relalg::ColumnRefAttr>> build_payloads_in_for;

	int curr_stage_ = 1;
	
	inline void copyAttrIfHas(mlir::Operation* src, mlir::Operation* tgt, std::vector<std::string> attr_list) {
		for (auto attr : attr_list) {
			if (src->hasAttr(attr)) {
				tgt->setAttr(attr, src->getAttr(attr));
			}
		}
	}

	void CmpToGDSA(mlir::db::CmpOp get_col_op, mlir::OpBuilder& builder);
	void UpdateJoinHTToGDSA(mlir::loop::UpdateHashTable update_op, mlir::OpBuilder& builder,
													mlir::BlockArgument& while_idx);
	void UpdateAggrHTToGDSA(mlir::loop::UpdateHashTable update_op, mlir::OpBuilder& builder,
													mlir::BlockArgument& while_idx);
	void ProbeHTToGDSA(mlir::loop::ProbeHashTable probe_op, mlir::OpBuilder& builder,
										 mlir::BlockArgument& while_idx);
	void YiledToGDSA(mlir::loop::YieldOp yield_op, mlir::OpBuilder& builder,
									 mlir::BlockArgument& while_idx);
	void NestedIfToGDSA(mlir::loop::IfOp if_op, mlir::OpBuilder& builder,
											mlir::BlockArgument& while_idx);
	void GetColToGDSA(mlir::relalg::GetColumnOp update_op, mlir::OpBuilder& builder, mlir::Value& idx);
	void BinaryToGDSA(mlir::Operation* bin_op, mlir::OpBuilder& builder);
	void UpdateToGDSA(mlir::loop::UpdateOp update_op, mlir::OpBuilder& builder,
										mlir::BlockArgument& while_idx);
	void ConstToGDSA(mlir::db::ConstantOp const_op, mlir::OpBuilder& builder);

public:
	// LoopTranslator(/* args */);
	// ~LoopTranslator();
	void LoopToWhile(mlir::Operation* op, mlir::OpBuilder& builder);
};


} // end namespace loop
} // end namespace mlir 

#endif
