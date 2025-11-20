#include <csignal>
#include <filesystem>
#include <fstream>
#include <spawn.h>

#include "dlfcn.h"
#include "unistd.h"

#include "json.h"

#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Analysis/DataLayoutAnalysis.h"

#include "frontend/SQL/Parser.h"

#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/DBToStd/DBToStd.h"
#include "mlir/Conversion/DSAToStd/DSAToStd.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/RelAlgToDB/RelAlgToDBPass.h"
#include "mlir/Conversion/RelAlgToLoop/RelAlgToLoopPass.h"
#include "mlir/Conversion/LoopToGDSA/LoopToGDSAPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"

#include "mlir/Transforms/CustomPasses.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/Passes.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/Dialect/Loop/IR/LoopDialect.h"
#include "mlir/Dialect/GDSA/IR/GDSADialect.h"
// #include "mlir/Dialect/Loop/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"

#include "mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include <mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h>
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/util/UtilTypes.h>
#include <runner/runner.h>

#include <unordered_map>

#include <sched.h>

#include <iostream>
namespace {
struct ToLLVMLoweringPass
   : public mlir::PassWrapper<ToLLVMLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
   void getDependentDialects(mlir::DialectRegistry& registry) const override {
      registry.insert<mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect, mlir::arith::ArithmeticDialect>();
   }
   void runOnOperation() final;
};
struct ToLLVMExceptSCFLowering
   : public mlir::PassWrapper<ToLLVMExceptSCFLowering, mlir::OperationPass<mlir::ModuleOp>> {
   void getDependentDialects(mlir::DialectRegistry& registry) const override {
      registry.insert<mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect, mlir::arith::ArithmeticDialect>();
   }
   void runOnOperation() final;
};
struct InsertPerfAsmPass
   : public mlir::PassWrapper<InsertPerfAsmPass, mlir::OperationPass<mlir::ModuleOp>> {
   void getDependentDialects(mlir::DialectRegistry& registry) const override {
      registry.insert<mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect, mlir::arith::ArithmeticDialect>();
   }
   void runOnOperation() final;
};
struct EnforceCPPABIPass
   : public mlir::PassWrapper<EnforceCPPABIPass, mlir::OperationPass<mlir::LLVM::LLVMFuncOp>> {
   void getDependentDialects(mlir::DialectRegistry& registry) const override {
      registry.insert<mlir::LLVM::LLVMDialect>();
   }
   void runOnOperation() final;
};
} // end anonymous namespace

void EnforceCPPABIPass::runOnOperation() {
   auto funcOp = getOperation();
   if (funcOp.isPrivate()) {
      auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
      auto& dataLayoutAnalysis = getAnalysis<mlir::DataLayoutAnalysis>();
      size_t numRegs = 0;
      std::vector<size_t> passByMem;
      for (size_t i = 0; i < funcOp.getNumArguments(); i++) {
         auto dataLayout = dataLayoutAnalysis.getAbove(funcOp.getOperation());
         auto typeSize = dataLayout.getTypeSize(funcOp.getArgumentTypes()[i]);
         if (typeSize <= 16) {
            auto requiredRegs = typeSize <= 8 ? 1 : 2;
            if (numRegs + requiredRegs > 6) {
               passByMem.push_back(i);
            } else {
               numRegs += requiredRegs;
            }
         } else {
            passByMem.push_back(i);
         }
      }
      if (passByMem.empty()) return;
      std::vector<mlir::Type> paramTypes(funcOp.getArgumentTypes().begin(), funcOp.getArgumentTypes().end());
      for (size_t memId : passByMem) {
         paramTypes[memId] = mlir::LLVM::LLVMPointerType::get(paramTypes[memId]);
         funcOp.setArgAttr(memId, "llvm.byval", mlir::UnitAttr::get(&getContext()));
      }

      funcOp.setType(mlir::LLVM::LLVMFunctionType::get(funcOp.getFunctionType().getReturnType(), paramTypes));
      auto uses = mlir::SymbolTable::getSymbolUses(funcOp, moduleOp.getOperation());
      for (auto use : *uses) {
         auto callOp = mlir::cast<mlir::LLVM::CallOp>(use.getUser());
         for (size_t memId : passByMem) {
            auto userFunc = callOp->getParentOfType<mlir::LLVM::LLVMFuncOp>();
            mlir::OpBuilder builder(userFunc->getContext());
            builder.setInsertionPointToStart(&userFunc.getBody().front());
            auto const1 = builder.create<mlir::LLVM::ConstantOp>(callOp.getLoc(), builder.getI64Type(), builder.getI64IntegerAttr(1));
            mlir::Value allocatedElementPtr = builder.create<mlir::LLVM::AllocaOp>(callOp.getLoc(), paramTypes[memId], const1, 16);
            mlir::OpBuilder builder2(userFunc->getContext());
            builder2.setInsertionPoint(callOp);
            builder2.create<mlir::LLVM::StoreOp>(callOp->getLoc(), callOp.getOperand(memId), allocatedElementPtr);
            callOp.setOperand(memId, allocatedElementPtr);
         }
      }
   }
}
void ToLLVMLoweringPass::runOnOperation() {
   // The first thing to define is the conversion target. This will define the
   // final target for this lowering. For this lowering, we are only targeting
   // the LLVM dialect.
   const auto& dataLayoutAnalysis = getAnalysis<mlir::DataLayoutAnalysis>();

   mlir::LLVMConversionTarget target(getContext());
   target.addLegalOp<mlir::ModuleOp>();

   // During this lowering, we will also be lowering the MemRef types, that are
   // currently being operated on, to a representation in LLVM. To perform this
   // conversion we use a TypeConverter as part of the lowering. This converter
   // details how one type maps to another. This is necessary now that we will be
   // doing more complicated lowerings, involving loop region arguments.
   mlir::LowerToLLVMOptions options(&getContext(), dataLayoutAnalysis.getAtOrAbove(getOperation()));
   //options.emitCWrappers = true;
   mlir::LLVMTypeConverter typeConverter(&getContext(), options, &dataLayoutAnalysis);
   typeConverter.addSourceMaterialization([&](mlir::OpBuilder&, mlir::FunctionType type, mlir::ValueRange valueRange, mlir::Location loc) {
      return valueRange.front();
   });

   mlir::RewritePatternSet patterns(&getContext());
   populateAffineToStdConversionPatterns(patterns);
   mlir::populateSCFToControlFlowConversionPatterns(patterns);
   mlir::util::populateUtilToLLVMConversionPatterns(typeConverter, patterns);
   mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
   mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

   mlir::populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
   mlir::arith::populateArithmeticToLLVMConversionPatterns(typeConverter, patterns);
   // We want to completely lower to LLVM, so we use a `FullConversion`. This
   // ensures that only legal operations will remain after the conversion.
   auto module = getOperation();
   if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
}
void ToLLVMExceptSCFLowering::runOnOperation() {
   const auto& dataLayoutAnalysis = getAnalysis<mlir::DataLayoutAnalysis>();

   mlir::LLVMConversionTarget target(getContext());
   target.addLegalOp<mlir::ModuleOp>();
   target.addIllegalDialect<mlir::util::UtilDialect>();
   target.addIllegalDialect<mlir::arith::ArithmeticDialect>();

   mlir::LowerToLLVMOptions options(&getContext(), dataLayoutAnalysis.getAtOrAbove(getOperation()));
   mlir::LLVMTypeConverter typeConverter(&getContext(), options, &dataLayoutAnalysis);
   typeConverter.addSourceMaterialization([&](mlir::OpBuilder&, mlir::FunctionType type, mlir::ValueRange valueRange, mlir::Location loc) {
      return valueRange.front();
   });

   mlir::RewritePatternSet patterns(&getContext());
   populateAffineToStdConversionPatterns(patterns);
   // mlir::populateSCFToControlFlowConversionPatterns(patterns);
   mlir::util::populateUtilToLLVMConversionPatterns(typeConverter, patterns);
   mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
   // mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

   mlir::populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
   mlir::arith::populateArithmeticToLLVMConversionPatterns(typeConverter, patterns);

   auto module = getOperation();
   if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
}
mlir::Location dropNames(mlir::Location l) {
   if (auto namedLoc = l.dyn_cast<mlir::NameLoc>()) {
      return dropNames(namedLoc.getChildLoc());
   } else if (auto namedResultsLoc = l.dyn_cast<mlir::NamedResultsLoc>()) {
      return dropNames(namedResultsLoc.getChildLoc());
   }
   return l;
}
void InsertPerfAsmPass::runOnOperation() {
   getOperation()->walk([](mlir::LLVM::CallOp callOp) {
      size_t loc = 0xdeadbeef;
      if (auto fileLoc = dropNames(callOp.getLoc()).dyn_cast<mlir::FileLineColLoc>()) {
         loc = fileLoc.getLine();
      }
      mlir::OpBuilder b(callOp);
      const auto* asmTp = "mov r15,{0}";
      auto asmDialectAttr =
         mlir::LLVM::AsmDialectAttr::get(b.getContext(), mlir::LLVM::AsmDialect::AD_Intel);
      const auto* asmCstr =
         "";
      auto asmStr = llvm::formatv(asmTp, llvm::format_hex(loc, /*width=*/16)).str();
      b.create<mlir::LLVM::InlineAsmOp>(callOp->getLoc(), mlir::TypeRange(), mlir::ValueRange(), llvm::StringRef(asmStr), llvm::StringRef(asmCstr), true, false, asmDialectAttr, mlir::ArrayAttr());
   });
}

namespace runner {
std::unique_ptr<mlir::Pass> createLowerToLLVMPass() {
   return std::make_unique<ToLLVMLoweringPass>();
}
std::unique_ptr<mlir::Pass> createLowerToLLVMExceptSCFPass() {
   return std::make_unique<ToLLVMExceptSCFLowering>();
}
int loadMLIR(std::string inputFilename, mlir::MLIRContext& context, mlir::OwningOpRef<mlir::ModuleOp>& module) {
   llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
   if (std::error_code ec = fileOrErr.getError()) {
      llvm::errs() << "Could not open input file: " << ec.message() << "\n";
      return -1;
   }

   // Parse the input mlir.
   llvm::SourceMgr sourceMgr;
   sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
   module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
   if (!module) {
      llvm::errs() << "Error can't load file " << inputFilename << "\n";
      return 3;
   }
   return 0;
}
int loadMLIRFromString(const std::string& input, mlir::MLIRContext& context, mlir::OwningOpRef<mlir::ModuleOp>& module) {
   module = mlir::parseSourceString<mlir::ModuleOp>(input, &context);
   if (!module) {
      llvm::errs() << "Error can't load module\n";
      return 3;
   }
   return 0;
}
static std::unique_ptr<llvm::Module>
convertMLIRModule(mlir::ModuleOp module, llvm::LLVMContext& context, mlir::LLVM::detail::DebuggingLevel debugLevel) {
   auto startConv = std::chrono::high_resolution_clock::now();

   std::unique_ptr<llvm::Module> mainModule =
      translateModuleToLLVMIR(module, context, "LLVMDialectModule", debugLevel);
   auto endConv = std::chrono::high_resolution_clock::now();
   std::cout << "conversion: " << std::chrono::duration_cast<std::chrono::microseconds>(endConv - startConv).count() / 1000.0 << " ms" << std::endl;
   std::cout << "Lower to LLVM: " << std::endl;
   // mainModule->dump();
   return mainModule;
}

struct RunnerContext {
   mlir::MLIRContext context;
   mlir::OwningOpRef<mlir::ModuleOp> module;
   llvm::SmallVector<mlir::ModuleOp, 8> subq_modules;
   size_t numArgs;
   size_t numResults;
   std::unordered_map<std::string, mlir::ModuleOp> split_modules;
};
static mlir::Location tagLocHook(mlir::Location loc) {
   static size_t operationId = 0;
   auto idAsStr = std::to_string(operationId++);
   return mlir::NameLoc::get(mlir::StringAttr::get(loc.getContext(), idAsStr), loc);
}
RunMode Runner::getRunMode() {
   runner::RunMode runMode;
   if (RUN_QUERIES_WITH_PERF) {
      runMode = runner::RunMode::PERF;
   } else {
      runMode = runner::RunMode::DEFAULT;
   }
   if (const char* mode = std::getenv("DHAP_QC_DEBUG")) {
      if (std::string(mode) == "PERF") {
         runMode = runner::RunMode::PERF;
      } else if (std::string(mode) == "DEFAULT") {
         runMode = runner::RunMode::DEFAULT;
      } else if (std::string(mode) == "DEBUGGING" or std::string(mode) == "1") {
         runMode = runner::RunMode::DEBUGGING;
      } else if (std::string(mode) == "SPEED") {
         std::cout << "using speed mode" << std::endl;
         runMode = runner::RunMode::SPEED;
      }
   }
   return runMode;
}
Runner::Runner(RunMode mode) : context(nullptr), runMode(mode) {
   llvm::DebugFlag = false;
   LLVMInitializeX86AsmParser();
   if (mode == RunMode::DEBUGGING || mode == RunMode::PERF) {
      mlir::Operation::setTagLocationHook(tagLocHook);
   }
   RunnerContext* ctxt = new RunnerContext;
   ctxt->context.disableMultithreading();
   this->context = (void*) ctxt;
}
bool Runner::loadSQL(std::string sql, runtime::Database& database) {
   llvm::DebugFlag = false;
   RunnerContext* ctxt = (RunnerContext*) this->context;
   mlir::MLIRContext& context = ctxt->context;
   mlir::DialectRegistry registry;
   registry.insert<mlir::BuiltinDialect>();
   registry.insert<mlir::relalg::RelAlgDialect>();
   registry.insert<mlir::loop::LoopDialect>();
   registry.insert<mlir::gdsa::GDSADialect>();
   registry.insert<mlir::db::DBDialect>();
   registry.insert<mlir::func::FuncDialect>();
   registry.insert<mlir::arith::ArithmeticDialect>();

   registry.insert<mlir::memref::MemRefDialect>();
   registry.insert<mlir::util::UtilDialect>();
   registry.insert<mlir::scf::SCFDialect>();
   registry.insert<mlir::omp::OpenMPDialect>();
   registry.insert<mlir::LLVM::LLVMDialect>();
   context.appendDialectRegistry(registry);
   context.loadAllAvailableDialects();
   context.loadDialect<mlir::relalg::RelAlgDialect>();
   mlir::registerLLVMDialectTranslation(context);
   mlir::registerOpenMPDialectTranslation(context);

   mlir::OpBuilder builder(&context);

   mlir::ModuleOp moduleOp = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
   frontend::sql::Parser translator(sql, database, moduleOp);

   builder.setInsertionPointToStart(moduleOp.getBody());
   auto* queryBlock = new mlir::Block;
   std::vector<mlir::Type> returnTypes;
   {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(queryBlock);
      auto val = translator.translate(builder);
      if (val.has_value()) {
         builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), val.value());
         returnTypes.push_back(val.value().getType());
      } else {
         builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
      }
   }
   mlir::func::FuncOp funcOp = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "main", builder.getFunctionType({}, returnTypes));
   funcOp.getBody().push_back(queryBlock);
   ctxt->module = moduleOp;
   snapshot("sql-input.mlir");
   return true;
}
bool Runner::load(std::string file) {
   RunnerContext* ctxt = (RunnerContext*) this->context;

   mlir::DialectRegistry registry;
   registry.insert<mlir::relalg::RelAlgDialect>();
   registry.insert<mlir::db::DBDialect>();
   registry.insert<mlir::dsa::DSADialect>();
   registry.insert<mlir::func::FuncDialect>();
   registry.insert<mlir::arith::ArithmeticDialect>();
   registry.insert<mlir::cf::ControlFlowDialect>();

   registry.insert<mlir::scf::SCFDialect>();

   registry.insert<mlir::util::UtilDialect>();
   registry.insert<mlir::LLVM::LLVMDialect>();
   registry.insert<mlir::memref::MemRefDialect>();

   mlir::MLIRContext& context = ctxt->context;
   context.appendDialectRegistry(registry);
   mlir::registerLLVMDialectTranslation(context);

   llvm::SourceMgr sourceMgr;
   llvm::DebugFlag = false;
   mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
   if (loadMLIR(file, context, ctxt->module))
      return false;
   return true;
}
bool Runner::loadLLVM(std::string file) {
   RunnerContext* ctxt = (RunnerContext*) this->context;
   mlir::DialectRegistry registry;
   registry.insert<mlir::LLVM::LLVMDialect>();
   registry.insert<mlir::omp::OpenMPDialect>();

   mlir::MLIRContext& context = ctxt->context;
   context.appendDialectRegistry(registry);
   mlir::registerLLVMDialectTranslation(context);
   mlir::registerOpenMPDialectTranslation(context);

   llvm::SourceMgr sourceMgr;
   llvm::DebugFlag = false;
   mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
   if (loadMLIR(file, context, ctxt->module))
      return false;
   // The LLVM file will not be passed to lowering, check its numArg and numRes here
   mlir::ModuleOp moduleOp = ctxt->module.get();
   if (auto mainFunc = moduleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>("main")) {
      ctxt->numArgs = mainFunc.getNumArguments();
      ctxt->numResults = mainFunc.getNumResults();
   }
   return true;
}
bool Runner::loadString(std::string input) {
   RunnerContext* ctxt = (RunnerContext*) this->context;
   mlir::DialectRegistry registry;
   registry.insert<mlir::relalg::RelAlgDialect>();
   registry.insert<mlir::db::DBDialect>();
   registry.insert<mlir::dsa::DSADialect>();
   registry.insert<mlir::func::FuncDialect>();
   registry.insert<mlir::scf::SCFDialect>();
   registry.insert<mlir::cf::ControlFlowDialect>();

   registry.insert<mlir::util::UtilDialect>();
   registry.insert<mlir::LLVM::LLVMDialect>();
   registry.insert<mlir::memref::MemRefDialect>();

   mlir::MLIRContext& context = ctxt->context;
   context.appendDialectRegistry(registry);
   mlir::registerLLVMDialectTranslation(context);

   llvm::DebugFlag = false;
   if (loadMLIRFromString(input, context, ctxt->module))
      return false;
   return true;
}

bool dumpModules(std::string file_name,
                 llvm::SmallVector<mlir::ModuleOp, 8>& modules, RunMode runMode) {
   if (runMode != RunMode::DEBUGGING) {
      return true;
   }
   std::error_code error;
   llvm::raw_fd_ostream outfile(file_name, error, llvm::sys::fs::OF_Text);
   if (error) {
      return false;
   }
   for (auto& module_ref : modules) {
      module_ref->print(outfile);
   }
   outfile.flush();
   return true;
}

bool dumpModules(std::string file_name,
                 std::unordered_map<std::string, mlir::ModuleOp>& modules, RunMode runMode) {
   llvm::SmallVector<mlir::ModuleOp, 8> modules1;
   for (auto& nm : modules) {
      modules1.push_back(nm.second);
   }
   return dumpModules(file_name, modules1, runMode);
}

void snapshotModule(RunMode runMode, mlir::ModuleOp module, std::string fileName) {
   if (runMode == RunMode::DEBUGGING || runMode == RunMode::PERF) {
      fileName += ".mlir";
      mlir::PassManager pm(module->getContext());
      mlir::OpPrintingFlags flags;
      flags.enableDebugInfo(false);
      pm.addPass(mlir::createLocationSnapshotPass(flags, fileName));
      assert(pm.run(module).succeeded());
   }
}

bool Runner::dhap_compile(runtime::Database& db) {
   RunnerContext* ctxt = (RunnerContext*) this->context;
   mlir::PassManager pm(&ctxt->context);
   pm.enableVerifier(runMode != RunMode::SPEED);
   pm.addPass(mlir::createInlinerPass());
   pm.addPass(mlir::createSymbolDCEPass());
   mlir::relalg::createQueryOptPipeline(pm, &db);
   if (mlir::failed(pm.run(ctxt->module.get()))) {
      return false;
   }
   snapshot("sql-opt.mlir");

   mlir::PassManager pms(&ctxt->context);
   pms.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createSplitSubqueryPass(ctxt->subq_modules));
   if (mlir::failed(pms.run(ctxt->module.get()))) {
      return false;
   }
   if (ctxt->subq_modules.size() == 0) {		// if only 1 sub-query
      ctxt->subq_modules.push_back(ctxt->module.get());
   }
   if (!dumpModules("sub-queries.mlir", ctxt->subq_modules, runMode)) {
      return false;
   }
   
   const uint32_t num_subq = ctxt->subq_modules.size();
   std::ofstream num_subq_file("NUM_SUBQUERY");
   if (!num_subq_file.is_open()) {
      std::cerr << "Failed to open NUM_SUBQUERY." << std::endl;
      exit(1);
   }
   num_subq_file << num_subq;
   num_subq_file.close();
   for (uint32_t subq_id = 0; subq_id < num_subq; subq_id++) {
      const auto subq_id_str = std::to_string(subq_id);
      const auto plan_fname = "plan"+subq_id_str+".json";
   
      auto& subq_module = ctxt->subq_modules[subq_id];
      llvm::outs() << "Sub query " << subq_id << "\n";
      mlir::PassManager pmp(&ctxt->context);
      pmp.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createPlanningPass(plan_fname, &db));
      if (mlir::failed(pmp.run(subq_module))) {
         return false;
      }
      std::ifstream plan_file(plan_fname);
      if (!plan_file.is_open()) {
         std::cerr << "Failed to open plan file." << plan_fname << std::endl;
         exit(1);
      }
      nlohmann::json plan;
      plan_file >> plan;
      plan_file.close();

      mlir::PassManager pma(&ctxt->context);
      pma.enableVerifier();
      pma.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createOffloadPass(subq_id));
      pma.addPass(mlir::createCanonicalizerPass());
      if (mlir::failed(pma.run(subq_module))) {
         return false;
      }
      snapshotModule(runMode, subq_module, "sql-offload"+subq_id_str);

      mlir::PassManager pm0(&ctxt->context);
      pm0.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createInsertShflPass(plan));
      pm0.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createSplitForLLVMPass(ctxt->split_modules));
      if (mlir::failed(pm0.run(subq_module))) {
         return false;
      }
      snapshotModule(runMode, subq_module, "sql-shfl"+subq_id_str);

      mlir::PassManager pm1(&ctxt->context);
      pm1.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createLowerToLoopPass());
      if (mlir::failed(pm1.run(subq_module))) {
         return false;
      }
      snapshotModule(runMode, subq_module, "loop"+subq_id_str);

      mlir::PassManager pm2(&ctxt->context);
      pm2.addPass(mlir::createCanonicalizerPass());
      if (mlir::failed(pm2.run(subq_module))) {
         return false;
      }
      snapshotModule(runMode, subq_module, "loop-canonical"+subq_id_str);
      
      mlir::PassManager pm3(&ctxt->context);
      pm3.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createFuseLoopPass());
      if (mlir::failed(pm3.run(subq_module))) {
         return false;
      }
      snapshotModule(runMode, subq_module, "loop-fused"+subq_id_str);

      mlir::PassManager pm31(&ctxt->context);
      pm31.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createGeneratePlanPass(subq_id_str, plan));
      if (mlir::failed(pm31.run(subq_module))) {
         return false;
      }
		std::ofstream plan_ofile(plan_fname);
		plan_ofile << std::setw(2) << plan << std::endl;

      mlir::PassManager pm4(&ctxt->context);
      pm4.addNestedPass<mlir::func::FuncOp>(mlir::loop::createLowerToGDSAPass());
      // THE OPTTTTT
      if (!std::getenv("DHAP_GOPT_OFF")) {
         pm4.addPass(mlir::createCanonicalizerPass());
      }
      if (mlir::failed(pm4.run(subq_module))) {
         return false;
      }
      snapshotModule(runMode, subq_module, "gdsa"+subq_id_str);
      mlir::PassManager pm5(&ctxt->context);
      pm5.addNestedPass<mlir::func::FuncOp>(mlir::gdsa::createCUDAGenPass(subq_id_str));
      if (mlir::failed(pm5.run(subq_module))) {
         return false;
      }
      
      // Process LLVM modules
      mlir::PassManager pm6(&ctxt->context);
      pm6.addNestedPass<mlir::func::FuncOp>(
         mlir::relalg::createUpdateProbeBaseTablePass(plan["table_schema"])
      );
      for (auto& nm : ctxt->split_modules) {
         mlir::ModuleOp module_op = nm.second;
         if (mlir::failed(pm6.run(module_op))) {
            return false;
         }
      }
      std::error_code error;
      if (runMode == RunMode::DEBUGGING) {
         const auto split_mod_fname = "split-relalg"+subq_id_str+".mlir";
         llvm::raw_fd_ostream split_mod_out(split_mod_fname, error, llvm::sys::fs::OF_Text);
         if (error) {
            return false;
         }
         for (auto& nm : ctxt->split_modules) {
            mlir::ModuleOp module_op = nm.second;
            module_op.print(split_mod_out);
         }
         split_mod_out.flush();
      }

      if (!lowerSplitModules(subq_id_str)) {
         std::cerr << "cannot lower split modules to std" << std::endl;
      }
      if (!lowerSplitModulesToLLVM()) {
         std::cerr << "cannot lower split modules to llvm" << std::endl;
      }
      for (auto& nm : ctxt->split_modules) {
         const auto mod_llvm_fname = "subq"+subq_id_str+nm.first+".mlir";
         llvm::raw_fd_ostream module_llvm_out(mod_llvm_fname, error, llvm::sys::fs::OF_Text);
         if (error) {
            return false;
         }
         mlir::ModuleOp module_op = nm.second;
         module_op.print(module_llvm_out);
         module_llvm_out.flush();
      }
      // clear modules for next sub-query
      ctxt->split_modules.clear();
   }

   return true;
}

bool Runner::optimize(runtime::Database& db) {
   auto start = std::chrono::high_resolution_clock::now();
   RunnerContext* ctxt = (RunnerContext*) this->context;
   mlir::PassManager pm(&ctxt->context);
   pm.enableVerifier(runMode != RunMode::SPEED);
   pm.addPass(mlir::createInlinerPass());
   pm.addPass(mlir::createSymbolDCEPass());
   mlir::relalg::createQueryOptPipeline(pm, &db);
   if (mlir::failed(pm.run(ctxt->module.get()))) {
      return false;
   }
   snapshot("sql-opt.mlir");
   auto end = std::chrono::high_resolution_clock::now();
   
   std::cout << "optimization took: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << " ms" << std::endl;
   {
      auto start = std::chrono::high_resolution_clock::now();

      mlir::PassManager pm2(&ctxt->context);
      pm2.enableVerifier(runMode != RunMode::SPEED);
      mlir::relalg::createLowerRelAlgPipeline(pm2);
      if (mlir::failed(pm2.run(ctxt->module.get()))) {
         return false;
      }
      auto end = std::chrono::high_resolution_clock::now();
      std::cout << "lowering to db took: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << " ms" << std::endl;
   }
   snapshot("dbdsa.mlir");
   return true;
}
bool Runner::lower() {
   auto start = std::chrono::high_resolution_clock::now();
   RunnerContext* ctxt = (RunnerContext*) this->context;
   mlir::PassManager pm(&ctxt->context);
   pm.enableVerifier(runMode != RunMode::SPEED);
   mlir::db::createLowerDBPipeline(pm);
   if (mlir::failed(pm.run(ctxt->module.get()))) {
      return false;
   }
   snapshot("lower-db.mlir");
   mlir::PassManager pm2(&ctxt->context);

   pm2.addPass(mlir::dsa::createLowerToStdPass());
   pm2.addPass(mlir::createCanonicalizerPass());
   if (mlir::failed(pm2.run(ctxt->module.get()))) {
      return false;
   }
   snapshot("std0.mlir");
   mlir::PassManager pmFunc(&ctxt->context, mlir::func::FuncOp::getOperationName());
   pmFunc.enableVerifier(runMode != RunMode::SPEED);
   pmFunc.addPass(mlir::createLoopInvariantCodeMotionPass());
   pmFunc.addPass(mlir::createSinkOpPass());
   pmFunc.addPass(mlir::createCSEPass());

   ctxt->module.get().walk([&](mlir::func::FuncOp f) {
      if (!f->hasAttr("passthrough")) {
         if (mlir::failed(pmFunc.run(f))) {
            return; //todo:fixed
         }
      }
   });

   auto end = std::chrono::high_resolution_clock::now();
   std::cout << "lowering to std took: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << " ms" << std::endl;
   snapshot("std.mlir");
   return true;
}
bool Runner::lowerSplitModules(std::string subq_id_str) {
   RunnerContext* ctxt = (RunnerContext*) this->context;
   
   std::unique_ptr<llvm::raw_fd_ostream> split_mod_out;
   if (runMode == RunMode::DEBUGGING) {
      std::error_code error;
      const auto split_mod_fname = "split-dsa"+subq_id_str+".mlir";
      split_mod_out = std::make_unique<llvm::raw_fd_ostream>(split_mod_fname, error, 
                                                             llvm::sys::fs::OF_Text);
      if (error) {
         return false;
      }
   }
   for (auto& nm : ctxt->split_modules) {
      mlir::ModuleOp moduleOp = nm.second;
      mlir::MLIRContext* mlir_ctxt = moduleOp.getContext();
      
      mlir::PassManager pm0(mlir_ctxt);
      pm0.enableVerifier(runMode != RunMode::SPEED);
      mlir::relalg::createLowerRelAlgPipeline(pm0);
      if (mlir::failed(pm0.run(moduleOp))) {
         return false;
      }
      if (runMode == RunMode::DEBUGGING) {
         moduleOp.print(*split_mod_out);
      }

      mlir::PassManager pm(mlir_ctxt);
      pm.enableVerifier(runMode != RunMode::SPEED);
      mlir::db::createLowerDBPipeline(pm);
      if (mlir::failed(pm.run(moduleOp))) {
         return false;
      }
      
      mlir::PassManager pm2(mlir_ctxt);
      pm2.enableVerifier(runMode != RunMode::SPEED);
      pm2.addPass(mlir::dsa::createLowerToStdPass());
      pm2.addPass(mlir::createCanonicalizerPass());
      if (mlir::failed(pm2.run(moduleOp))) {
         return false;
      }
      mlir::PassManager pmFunc(mlir_ctxt, mlir::func::FuncOp::getOperationName());
      pmFunc.enableVerifier(runMode != RunMode::SPEED);
      pmFunc.addPass(mlir::createLoopInvariantCodeMotionPass());
      pmFunc.addPass(mlir::createSinkOpPass());
      pmFunc.addPass(mlir::createCSEPass());

      moduleOp.walk([&](mlir::func::FuncOp f) {
         if (!f->hasAttr("passthrough")) {
            if (mlir::failed(pmFunc.run(f))) {
               return; //todo:fixed
            }
         }
      });
   }

   return true;
}
bool Runner::lowerToLLVMExceptSCF() {
   RunnerContext* ctxt = (RunnerContext*) this->context;
   mlir::ModuleOp moduleOp = ctxt->module.get();
   if (auto mainFunc = moduleOp.lookupSymbol<mlir::func::FuncOp>("main")) {
      ctxt->numArgs = mainFunc.getNumArguments();
      ctxt->numResults = mainFunc.getNumResults();
   }
   mlir::PassManager pm3(&ctxt->context);
   pm3.addPass(createLowerToLLVMExceptSCFPass());
   if (mlir::failed(pm3.run(ctxt->module.get()))) {
      return false;
   }
   snapshot("llvm_except_scf.mlir");
   return true;
}
bool Runner::lowerSCFToLLVMInnerPar() {
   auto start = std::chrono::high_resolution_clock::now();
   RunnerContext* ctxt = (RunnerContext*) this->context;
   mlir::ModuleOp moduleOp = ctxt->module.get();
   mlir::PassManager pm2(&ctxt->context);
   pm2.enableVerifier(runMode != RunMode::SPEED);
   
   pm2.addPass(mlir::createConvertSCFToOpenMPPass());
   pm2.addPass(mlir::createConvertOpenMPToLLVMPass());
   pm2.addPass(mlir::createConvertSCFToCFPass());
   pm2.addPass(mlir::createConvertOpenMPToLLVMPass());
   pm2.addPass(mlir::createReconcileUnrealizedCastsPass());
   
   pm2.addNestedPass<mlir::LLVM::LLVMFuncOp>(std::make_unique<EnforceCPPABIPass>());
   pm2.addPass(mlir::createCSEPass());
   if (mlir::failed(pm2.run(ctxt->module.get()))) {
      return false;
   }
   mlir::OpBuilder builder(moduleOp->getContext());
   builder.setInsertionPointToStart(moduleOp.getBody());
   auto pointerType = mlir::LLVM::LLVMPointerType::get(builder.getI8Type());
   auto globalOp = builder.create<mlir::LLVM::GlobalOp>(builder.getUnknownLoc(), builder.getI64Type(), false, mlir::LLVM::Linkage::Private, "execution_context", builder.getI64IntegerAttr(0));
   auto setExecContextFn = builder.create<mlir::LLVM::LLVMFuncOp>(moduleOp.getLoc(), "rt_set_execution_context", mlir::LLVM::LLVMFunctionType::get(mlir::LLVM::LLVMVoidType::get(builder.getContext()), builder.getI64Type()), mlir::LLVM::Linkage::External);
   {
      mlir::OpBuilder::InsertionGuard guard(builder);
      auto *block = setExecContextFn.addEntryBlock();
      auto execContext = block->getArgument(0);
      builder.setInsertionPointToStart(block);
      auto ptr = builder.create<mlir::LLVM::AddressOfOp>(builder.getUnknownLoc(), globalOp);
      builder.create<mlir::LLVM::StoreOp>(builder.getUnknownLoc(), execContext, ptr);
      builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{});
   }
   if (auto getExecContextFn = mlir::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(moduleOp.lookupSymbol("rt_get_execution_context"))) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      auto *block = getExecContextFn.addEntryBlock();
      builder.setInsertionPointToStart(block);
      auto ptr = builder.create<mlir::LLVM::AddressOfOp>(builder.getUnknownLoc(), globalOp);
      auto execContext = builder.create<mlir::LLVM::LoadOp>(builder.getUnknownLoc(), ptr);
      auto execContextAsPtr = builder.create<mlir::LLVM::IntToPtrOp>(builder.getUnknownLoc(), pointerType, execContext);
      builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{execContextAsPtr});
   }
   auto end = std::chrono::high_resolution_clock::now();
   std::cout << "lowering scf to llvm (omp) took: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << " ms" << std::endl;
   snapshot("llvm_omp.mlir");
   return true;
}
bool Runner::lowerSplitModulesToLLVM() {
   RunnerContext* ctxt = (RunnerContext*) this->context;
   for (auto& nm : ctxt->split_modules) {
      mlir::ModuleOp moduleOp = nm.second;
      if (auto mainFunc = moduleOp.lookupSymbol<mlir::func::FuncOp>(nm.first)) {
         llvm::outs() << "Process func " << nm.first << "\n";
         ctxt->numArgs = mainFunc.getNumArguments();
         ctxt->numResults = mainFunc.getNumResults();
      }
      mlir::PassManager pm2(&ctxt->context);
      pm2.enableVerifier(runMode != RunMode::SPEED);
      pm2.addPass(mlir::createConvertSCFToCFPass());
      pm2.addPass(createLowerToLLVMPass());
      pm2.addNestedPass<mlir::LLVM::LLVMFuncOp>(std::make_unique<EnforceCPPABIPass>());
      pm2.addPass(mlir::createCSEPass());
      if (mlir::failed(pm2.run(moduleOp))) {
         return false;
      }
      mlir::OpBuilder builder(moduleOp->getContext());
      builder.setInsertionPointToStart(moduleOp.getBody());
      auto pointerType = mlir::LLVM::LLVMPointerType::get(builder.getI8Type());
      auto globalOp = builder.create<mlir::LLVM::GlobalOp>(builder.getUnknownLoc(), builder.getI64Type(), false, mlir::LLVM::Linkage::Private, "execution_context", builder.getI64IntegerAttr(0));
      auto setExecContextFn = builder.create<mlir::LLVM::LLVMFuncOp>(moduleOp.getLoc(), "rt_set_execution_context", mlir::LLVM::LLVMFunctionType::get(mlir::LLVM::LLVMVoidType::get(builder.getContext()), builder.getI64Type()), mlir::LLVM::Linkage::External);
      {
         mlir::OpBuilder::InsertionGuard guard(builder);
         auto *block = setExecContextFn.addEntryBlock();
         auto execContext = block->getArgument(0);
         builder.setInsertionPointToStart(block);
         auto ptr = builder.create<mlir::LLVM::AddressOfOp>(builder.getUnknownLoc(), globalOp);
         builder.create<mlir::LLVM::StoreOp>(builder.getUnknownLoc(), execContext, ptr);
         builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{});
      }
      if (auto getExecContextFn = mlir::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(moduleOp.lookupSymbol("rt_get_execution_context"))) {
         mlir::OpBuilder::InsertionGuard guard(builder);
         auto *block = getExecContextFn.addEntryBlock();
         builder.setInsertionPointToStart(block);
         auto ptr = builder.create<mlir::LLVM::AddressOfOp>(builder.getUnknownLoc(), globalOp);
         auto execContext = builder.create<mlir::LLVM::LoadOp>(builder.getUnknownLoc(), ptr);
         auto execContextAsPtr = builder.create<mlir::LLVM::IntToPtrOp>(builder.getUnknownLoc(), pointerType, execContext);
         builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{execContextAsPtr});
      }
   }
   return true;
}
bool Runner::lowerToLLVM() {
   auto start = std::chrono::high_resolution_clock::now();
   RunnerContext* ctxt = (RunnerContext*) this->context;
   mlir::ModuleOp moduleOp = ctxt->module.get();
   if (auto mainFunc = moduleOp.lookupSymbol<mlir::func::FuncOp>("main")) {
      ctxt->numArgs = mainFunc.getNumArguments();
      ctxt->numResults = mainFunc.getNumResults();
   }
   mlir::PassManager pm2(&ctxt->context);
   pm2.enableVerifier(runMode != RunMode::SPEED);
   pm2.addPass(mlir::createConvertSCFToCFPass());
   pm2.addPass(createLowerToLLVMPass());
   pm2.addNestedPass<mlir::LLVM::LLVMFuncOp>(std::make_unique<EnforceCPPABIPass>());
   pm2.addPass(mlir::createCSEPass());
   if (mlir::failed(pm2.run(ctxt->module.get()))) {
      return false;
   }
   mlir::OpBuilder builder(moduleOp->getContext());
   builder.setInsertionPointToStart(moduleOp.getBody());
   auto pointerType = mlir::LLVM::LLVMPointerType::get(builder.getI8Type());
   auto globalOp = builder.create<mlir::LLVM::GlobalOp>(builder.getUnknownLoc(), builder.getI64Type(), false, mlir::LLVM::Linkage::Private, "execution_context", builder.getI64IntegerAttr(0));
   auto setExecContextFn = builder.create<mlir::LLVM::LLVMFuncOp>(moduleOp.getLoc(), "rt_set_execution_context", mlir::LLVM::LLVMFunctionType::get(mlir::LLVM::LLVMVoidType::get(builder.getContext()), builder.getI64Type()), mlir::LLVM::Linkage::External);
   {
      mlir::OpBuilder::InsertionGuard guard(builder);
      auto *block = setExecContextFn.addEntryBlock();
      auto execContext = block->getArgument(0);
      builder.setInsertionPointToStart(block);
      auto ptr = builder.create<mlir::LLVM::AddressOfOp>(builder.getUnknownLoc(), globalOp);
      builder.create<mlir::LLVM::StoreOp>(builder.getUnknownLoc(), execContext, ptr);
      builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{});
   }
   if (auto getExecContextFn = mlir::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(moduleOp.lookupSymbol("rt_get_execution_context"))) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      auto *block = getExecContextFn.addEntryBlock();
      builder.setInsertionPointToStart(block);
      auto ptr = builder.create<mlir::LLVM::AddressOfOp>(builder.getUnknownLoc(), globalOp);
      auto execContext = builder.create<mlir::LLVM::LoadOp>(builder.getUnknownLoc(), ptr);
      auto execContextAsPtr = builder.create<mlir::LLVM::IntToPtrOp>(builder.getUnknownLoc(), pointerType, execContext);
      builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{execContextAsPtr});
   }
   auto end = std::chrono::high_resolution_clock::now();
   std::cout << "lowering to llvm took: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << " ms" << std::endl;
   snapshot("llvm.mlir");
   return true;
}
void Runner::dump() {
   RunnerContext* ctxt = (RunnerContext*) this->context;
   mlir::OpPrintingFlags flags;
   ctxt->module->print(llvm::dbgs(), flags);
}

void Runner::snapshot(std::string fileName) {
   if (runMode == RunMode::DEBUGGING || runMode == RunMode::PERF) {
      static size_t cntr = 0;
      RunnerContext* ctxt = (RunnerContext*) this->context;
      mlir::PassManager pm(&ctxt->context);
      pm.enableVerifier(runMode == RunMode::DEBUGGING);
      mlir::OpPrintingFlags flags;
      flags.enableDebugInfo(false);
      if (fileName.empty()) {
         fileName = "snapshot-" + std::to_string(cntr++) + ".mlir";
      }
      pm.addPass(mlir::createLocationSnapshotPass(flags, fileName));
      assert(pm.run(*ctxt->module).succeeded());
   }
}

static llvm::Error optimizeModule(llvm::Module* module) {
   llvm::legacy::FunctionPassManager funcPM(module);
   funcPM.add(llvm::createInstructionCombiningPass());
   funcPM.add(llvm::createReassociatePass());
   funcPM.add(llvm::createGVNPass());
   funcPM.add(llvm::createCFGSimplificationPass());

   funcPM.doInitialization();
   for (auto& func : *module) {
      if (!func.hasOptNone()) {
         funcPM.run(func);
      }
   }
   funcPM.doFinalization();
   return llvm::Error::success();
}
cpu_set_t mask;

inline void assignToThisCore(int coreId) {
   CPU_ZERO(&mask);
   CPU_SET(coreId, &mask);
   sched_setaffinity(0, sizeof(mask), &mask);
}

static pid_t runPerfRecord() {
   assignToThisCore(0);
   pid_t childPid = 0;
   auto parentPid = std::to_string(getpid());
   const char* argV[] = {"perf", "record", "-R", "-e", "ibs_op//p", "-c", "5000", "--intr-regs=r15", "-C", "0", nullptr};
   auto status = posix_spawn(&childPid, "/usr/bin/perf", nullptr, nullptr, const_cast<char**>(argV), environ);
   sleep(5);
   assignToThisCore(0);
   if (status != 0)
      std::cerr << "Launching application Failed: " << status << std::endl;
   return childPid;
}

class WrappedExecutionEngine {
   std::unique_ptr<mlir::ExecutionEngine> engine;
   size_t jitTime;
   void* mainFuncPtr;
   void* setContextPtr;

   public:
   WrappedExecutionEngine(mlir::ModuleOp module, RunMode runMode) : mainFuncPtr(nullptr), setContextPtr(nullptr) {
      auto start = std::chrono::high_resolution_clock::now();
      auto jitCodeGenLevel = runMode == RunMode::DEBUGGING ? llvm::CodeGenOpt::Level::None : llvm::CodeGenOpt::Level::Default;
      auto debuggingLevel = runMode == RunMode::DEBUGGING ? mlir::LLVM::detail::DebuggingLevel::VARIABLES : (runMode == RunMode::PERF ? mlir::LLVM::detail::DebuggingLevel::LINES : mlir::LLVM::detail::DebuggingLevel::OFF);
      auto convertFn = [&](mlir::ModuleOp module, llvm::LLVMContext& context) { return convertMLIRModule(module, context, debuggingLevel); };
      auto optimizeFn = [&](llvm::Module* module) -> llvm::Error {if (runMode==RunMode::DEBUGGING){return llvm::Error::success();}else{return optimizeModule(module);} };
      auto maybeEngine = mlir::ExecutionEngine::create(module, {.llvmModuleBuilder = convertFn, .transformer = optimizeFn, .jitCodeGenOptLevel = jitCodeGenLevel, .sharedLibPaths = {"/build/llvm/lib/libomp.so", "/repo/resources/mlir/test/libprint.so"}, .enableObjectCache = true});
      assert(maybeEngine && "failed to construct an execution engine");
      engine = std::move(maybeEngine.get());

      auto lookupResult = engine->lookup("main");
      if (!lookupResult) {
         llvm::errs() << "JIT invocation failed\n";
      }
      mainFuncPtr = lookupResult.get();
      auto lookupResult2 = engine->lookup("rt_set_execution_context");
      if (!lookupResult2) {
         llvm::errs() << "JIT invocation failed\n";
      }
      setContextPtr = lookupResult2.get();
      auto end = std::chrono::high_resolution_clock::now();
      jitTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
   }
   bool succeeded() {
      return mainFuncPtr != nullptr && setContextPtr != nullptr;
   }
   bool linkStatic() {
      auto currPath = std::filesystem::current_path();

      engine->dumpToObjectFile("llvm-jit-static.o");
      std::string cmd = "g++ -shared -fPIC -o llvm-jit-static.so llvm-jit-static.o";
      auto* pPipe = ::popen(cmd.c_str(), "r");
      if (pPipe == nullptr) {
         return false;
      }
      std::array<char, 256> buffer;
      std::string result;
      while (not std::feof(pPipe)) {
         auto bytes = std::fread(buffer.data(), 1, buffer.size(), pPipe);
         result.append(buffer.data(), bytes);
      }
      auto rc = ::pclose(pPipe);
      if (WEXITSTATUS(rc)) {
         return false;
      }

      void* handle = dlopen(std::string(currPath.string() + "/llvm-jit-static.so").c_str(), RTLD_LAZY);
      const char* dlsymError = dlerror();
      if (dlsymError) {
         std::cout << "QueryCompiler: Cannot load symbol: " << std::string(dlsymError) << std::endl;
      }
      mainFuncPtr = dlsym(handle, "main");
      dlsymError = dlerror();
      if (dlsymError) {
         dlclose(handle);
         std::cout << "QueryCompiler: Cannot load symbol: " << std::string(dlsymError) << std::endl;
         return false;
      }
      setContextPtr = dlsym(handle, "rt_set_execution_context");
      dlsymError = dlerror();
      if (dlsymError) {
         dlclose(handle);
         std::cout << "QueryCompiler: Cannot load symbol: " << std::string(dlsymError) << std::endl;
         return false;
      }
      return true;
   }
   size_t getJitTime() {
      return jitTime;
   }
   void* getMainFuncPtr() const {
      return mainFuncPtr;
   }
   void* getSetContextPtr() const {
      return setContextPtr;
   }
};
bool Runner::runJit(runtime::ExecutionContext* context, size_t repeats, std::function<void(uint8_t*)> callback) {
   if (runMode == RunMode::PERF) {
      repeats = 1;
      reserveLastRegister = true;
   }
   RunnerContext* ctxt = (RunnerContext*) this->context;
   // Initialize LLVM targets.
   llvm::InitializeNativeTarget();
   llvm::InitializeNativeTargetAsmPrinter();
   auto targetTriple = llvm::sys::getDefaultTargetTriple();
   std::string errorMessage;
   const auto* target = llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
   if (!target) {
      assert(false && "could not get target");
      return false;
   }

   // Initialize LLVM targets.
   llvm::InitializeNativeTarget();
   llvm::InitializeNativeTargetAsmPrinter();

   // An optimization pipeline to use within the execution engine.
   if (runMode == RunMode::PERF) {
      mlir::PassManager pm(&ctxt->context);
      pm.enableVerifier(false);
      pm.addPass(std::make_unique<InsertPerfAsmPass>());
      if (mlir::failed(pm.run(ctxt->module.get()))) {
         return false;
      }
   }
   WrappedExecutionEngine engine(ctxt->module.get(), runMode);
   if (!engine.succeeded()) return false;
   if ((runMode == RunMode::PERF || runMode == RunMode::DEBUGGING) && !engine.linkStatic()) return false;
   typedef uint8_t* (*myfunc)(void*);
   auto fn = (myfunc) engine.getSetContextPtr();
   fn(context);
   uint8_t* res;
   std::cout << "jit: " << engine.getJitTime() / 1000.0 << " ms" << std::endl;
   pid_t pid;
   if (runMode == RunMode::PERF) {
      pid = runPerfRecord();
      uint64_t r15DefaultValue = 0xbadeaffe;
      __asm__ __volatile__("mov %0, %%r15\n\t"
                           : /* no output */
                           : "a"(r15DefaultValue)
                           : "%r15");
   }
   std::vector<size_t> measuredTimes;
   for (size_t i = 0; i < repeats; i++) {
      auto executionStart = std::chrono::high_resolution_clock::now();
      if (ctxt->numResults == 1) {
         typedef uint8_t* (*myfunc)();
         auto fn = (myfunc) engine.getMainFuncPtr();
         res = fn();
      } else {
         typedef void (*myfunc)();
         auto fn = (myfunc) engine.getMainFuncPtr();
         fn();
      }
      auto executionEnd = std::chrono::high_resolution_clock::now();
      measuredTimes.push_back(std::chrono::duration_cast<std::chrono::microseconds>(executionEnd - executionStart).count());
   }
   if (runMode == RunMode::PERF) {
      reserveLastRegister = false;
      kill(pid, SIGINT);
      sleep(2);
   }
   std::cout << "runtime: " << (measuredTimes.size() > 1 ? *std::min_element(measuredTimes.begin() + 1, measuredTimes.end()) : measuredTimes[0]) / 1000.0 << " ms" << std::endl;

   if (ctxt->numResults == 1) {
      callback(res);
   }

   return true;
}
Runner::~Runner() {
   if (this->context) {
      delete (RunnerContext*) this->context;
   }
}
} // namespace runner
