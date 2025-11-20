#include "mlir-support/offloader.h"
#include "mlir-support/eval.h"
#include "mlir/Dialect/RelAlg/Transforms/queryopt/QueryGraph.h"

#include <arrow/api.h>
#include <arrow/compute/api.h>

using namespace mlir::relalg;

namespace mlir {
namespace relalg {

Offloader::Offloader(int subq_id)
{
	std::string server_ip(std::getenv("SR_IP"));
	auto location = arrow::flight::Location::ForGrpcTcp(server_ip, 36433).ValueOrDie();
	client_ = arrow::flight::FlightClient::Connect(location).ValueOrDie();
	if (subq_id == 0) {
		arrow::flight::Action reset_action{"reset"};
		auto r = client_->DoAction(reset_action).ValueOrDie();
	}
}

Offloader::~Offloader()
{
	assert(client_->Close().ok());
}

void Offloader::OffloadSel(BaseTableOp basetable, SelectionOp sel)
{
	if (std::getenv("DHAP_OFFLOAD_OFF")) {
		return;
	}
	std::unordered_map<const mlir::relalg::Column*, std::string> mapping;
	for (auto c : basetable.columns()) {
			mapping[&c.getValue().cast<mlir::relalg::ColumnDefAttr>().getColumn()] = c.getName().str();
	}
	std::string tbl_name = basetable.table_identifier().str();
	arrow::flight::Action set_action{"set_offload_tbl", arrow::Buffer::FromString(tbl_name)};
	auto r = client_->DoAction(set_action).ValueOrDie();

	auto v = mlir::cast<mlir::relalg::ReturnOp>(sel.getPredicateBlock().getTerminator()).results()[0];
	arrow::compute::Expression sel_expr = support::eval::unpack(buildEvalExpr(v, mapping));
	// llvm::outs() << sel_expr.ToString() << "\n";
	arrow::flight::Action off_action{"offload_sel", arrow::compute::Serialize(sel_expr).ValueOrDie()};
	auto r1 = client_->DoAction(off_action).ValueOrDie();
}

} // end namespace relalg
} // end namespace mlir 