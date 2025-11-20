#ifndef MLIR_SUPPORT_OFFLOAD_H
#define MLIR_SUPPORT_OFFLOAD_H

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/Loop/IR/LoopOps.h"

#include <arrow/flight/api.h>

namespace mlir {
namespace relalg {

class Offloader
{
private:
	std::unique_ptr<arrow::flight::FlightClient> client_;
public:
	Offloader(int subq_id);
	~Offloader();
	void OffloadSel(BaseTableOp basetable, SelectionOp sel);
};

}
}

#endif