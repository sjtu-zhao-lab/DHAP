#ifndef MLIR_CONVERSION_LOOPTOGDSA_CUDAGENERATOR_H
#define MLIR_CONVERSION_LOOPTOGDSA_CUDAGENERATOR_H

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/DenseMap.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
// #include "mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/Loop/IR/LoopOps.h"
#include "mlir/Dialect/GDSA/IR/GDSAOps.h"
#include "mlir/Dialect/GDSA/IR/GDSAOpsInterfaces.h"

#include <fstream>
#include <unordered_map>

namespace mlir {
namespace gdsa {

class CUDAGenerator
{
private:
	std::string subq_id_str_;
	mlir::MLIRContext& mlir_ctxt_;
	mlir::relalg::ColumnManager attr_manager;
	// std::ofstream knl_file_;
	std::string file_name_;
	std::string knl_content;
	
	const std::string knl_prefix = "extern \"C\"\n__global__ void ";
	const std::string knl_start = 
		"  constexpr uint32_t block_size = 128;\n"
		"	 constexpr uint32_t cg_size = 8;\n"
		"  auto curr_cg =  cg::tiled_partition<cg_size>(cg::this_thread_block());\n"
  	"  int64_t const loop_stride = gridDim.x * block_size / cg_size;\n"
  	"  const int tid = block_size * blockIdx.x + threadIdx.x;\n"
		"  uint32_t idx = tid / cg_size;\n\n";
	const std::string probe_knl_start = 
		"  uint32_t probing_cg_id = threadIdx.x / cg_size;\n"
		"  constexpr int num_probing_cgs = block_size / cg_size;\n"
  	"  constexpr int buffer_size = 1;\n\n";

	int space_indent = 0;
	std::hash<std::string> str_hash = std::hash<std::string>{};
	// Two maps are both kernel-local
	llvm::DenseMap<mlir::Value, std::string> value2name;
	std::unordered_map<mlir::relalg::Column*, mlir::relalg::Column*> probe2build_key;
	// Global for all build and probe kernels
	std::unordered_map<mlir::relalg::Column*, std::vector<mlir::relalg::Column*>> build_key2plds;

	// Position before while to insert shared memory declaration
	int before_while_pos = 0;

	// Max stage no., for knowing the last probe
	int max_stage_;

	inline std::string ListToCommaSep(const std::vector<std::string> list, std::string srd = "()") {
		std::string ret = srd.substr(0, 1);
		for (auto s : list) {
			ret += s + ", ";
		}
		return ret.substr(0, ret.size()-2) + srd.substr(1, 1);
	}

	void RegularName(std::string& name) {
		std::replace(name.begin(), name.end(), '[', '_');
		std::replace(name.begin(), name.end(), ']', '_');
	}

	void PrepareProbeParams(While while_op);

	void genGet(GetElement get_op);
	void genEq(Equal eq_op);
	void genIf(If if_op);
	void genInsertMap(InsertMap insert_op);
	void genMakePair(MakePair makep_op);
	void genMapCount(MapCount mapc_op);
	void genMapFind(MapFind mapf_op);
	void genIncrease(Increase inc_op);
	void genAnyThread(AnyThread any_op);
	void genSingleThread(SingleThread single_op);
	void genMakeAggrKey(MakeAggrKey make_aggr_op);
	void genGetSecond(GetSecond get_sec_op);
	void genNotEqual(NotEqual neq_op);
	void genAtomicAdd(AtomicAdd add_op);
	void genMod(Mod mod_op);
	void genAtomicAdd1(AtomicAdd1 add1_op);
	void genGetGlobalIndex(GetGlobalIndex get_gidx_op);
	void genMaterialize(Materialize mater_op);
	void genConst(Const const_op);
	void genBinary(mlir::Operation* bin_op);
	void genStore(StoreElement store_op);

public:
	CUDAGenerator(std::string subq_id_str, mlir::MLIRContext& mlir_ctxt, 
								std::string file_name, int max_stage);
	~CUDAGenerator();
	void Generate(mlir::Operation* op);
	void GenerateBuildKnl(mlir::Operation* op);
	void GenerateProbeKnl(mlir::Operation* op);
};

} // end namespace gdsa
} // end namespace mlir 

#endif