#include "mlir/ExecutionEngine/ExecutionEngine.h"

#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Host.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Parser/Parser.h"

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>

#include <llvm/Support/SourceMgr.h>

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>

#include "runtime/ExecutionContext.h"

enum class RunMode {
   SPEED = 0, //Aim for maximum speed (no verification of generated MLIR
   DEFAULT = 1, //Execute without introducing extra steps for debugging/profiling, but verify generated MLIR
   PERF = 2, //Profiling
   DEBUGGING = 3 //Make generated code debuggable
};

static std::unique_ptr<llvm::Module>
convertMLIRModule(mlir::ModuleOp module, llvm::LLVMContext& context, mlir::LLVM::detail::DebuggingLevel debugLevel) {
   auto startConv = std::chrono::high_resolution_clock::now();

   std::unique_ptr<llvm::Module> mainModule =
      translateModuleToLLVMIR(module, context, "LLVMDialectModule", debugLevel);
   auto endConv = std::chrono::high_resolution_clock::now();
   // mainModule->dump();
   return mainModule;
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

class WrappedExecutionEngine {
   std::unique_ptr<mlir::ExecutionEngine> engine;
   uint32_t jitTime;
   void* mainFuncPtr;
   void* setContextPtr;

   public:
   WrappedExecutionEngine(mlir::ModuleOp module, RunMode runMode) : mainFuncPtr(nullptr), setContextPtr(nullptr) {
      auto start = std::chrono::high_resolution_clock::now();
      auto jitCodeGenLevel = runMode == RunMode::DEBUGGING ? llvm::CodeGenOpt::Level::None : llvm::CodeGenOpt::Level::Default;
      auto debuggingLevel = runMode == RunMode::DEBUGGING ? mlir::LLVM::detail::DebuggingLevel::VARIABLES : (runMode == RunMode::PERF ? mlir::LLVM::detail::DebuggingLevel::LINES : mlir::LLVM::detail::DebuggingLevel::OFF);
      auto convertFn = [&](mlir::ModuleOp module, llvm::LLVMContext& context) { return convertMLIRModule(module, context, debuggingLevel); };
      auto optimizeFn = [&](llvm::Module* module) -> llvm::Error {if (runMode==RunMode::DEBUGGING){return llvm::Error::success();}else{return optimizeModule(module);} };
      // auto maybeEngine = mlir::ExecutionEngine::create(module, {.llvmModuleBuilder = convertFn, .transformer = optimizeFn, .jitCodeGenOptLevel = jitCodeGenLevel, .sharedLibPaths = {"/repo/resources/mlir/test/libprint.so"}, .enableObjectCache = true});
      auto maybeEngine = mlir::ExecutionEngine::create(module, {.llvmModuleBuilder = convertFn, .transformer = optimizeFn, .jitCodeGenOptLevel = jitCodeGenLevel, .enableObjectCache = true});
      assert(maybeEngine && "failed to construct an execution engine");
      engine = std::move(maybeEngine.get());

      auto lookupResult = engine->lookup("main");
      if (!lookupResult) {
         llvm::errs() << "JIT invocation failed (no main)\n";
      }
      mainFuncPtr = lookupResult.get();
      auto lookupResult2 = engine->lookup("rt_set_execution_context");
      if (!lookupResult2) {
         llvm::errs() << "JIT invocation failed (no rt_set_execution_context)\n";
      }
      setContextPtr = lookupResult2.get();
      auto end = std::chrono::high_resolution_clock::now();
      jitTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
   }
   bool succeeded() {
      return mainFuncPtr != nullptr && setContextPtr != nullptr;
   }
  //  bool linkStatic() {
  //     auto currPath = std::filesystem::current_path();

  //     engine->dumpToObjectFile("llvm-jit-static.o");
  //     std::string cmd = "g++ -shared -fPIC -o llvm-jit-static.so llvm-jit-static.o";
  //     auto* pPipe = ::popen(cmd.c_str(), "r");
  //     if (pPipe == nullptr) {
  //        return false;
  //     }
  //     std::array<char, 256> buffer;
  //     std::string result;
  //     while (not std::feof(pPipe)) {
  //        auto bytes = std::fread(buffer.data(), 1, buffer.size(), pPipe);
  //        result.append(buffer.data(), bytes);
  //     }
  //     auto rc = ::pclose(pPipe);
  //     if (WEXITSTATUS(rc)) {
  //        return false;
  //     }

  //     void* handle = dlopen(std::string(currPath.string() + "/llvm-jit-static.so").c_str(), RTLD_LAZY);
  //     const char* dlsymError = dlerror();
  //     if (dlsymError) {
  //        std::cout << "QueryCompiler: Cannot load symbol: " << std::string(dlsymError) << std::endl;
  //     }
  //     mainFuncPtr = dlsym(handle, "main");
  //     dlsymError = dlerror();
  //     if (dlsymError) {
  //        dlclose(handle);
  //        std::cout << "QueryCompiler: Cannot load symbol: " << std::string(dlsymError) << std::endl;
  //        return false;
  //     }
  //     setContextPtr = dlsym(handle, "rt_set_execution_context");
  //     dlsymError = dlerror();
  //     if (dlsymError) {
  //        dlclose(handle);
  //        std::cout << "QueryCompiler: Cannot load symbol: " << std::string(dlsymError) << std::endl;
  //        return false;
  //     }
  //     return true;
  //  }
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

extern "C" void printFinish() {printf("LLVM Finished\n");}