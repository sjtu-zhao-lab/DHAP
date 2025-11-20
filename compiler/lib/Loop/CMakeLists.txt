
add_mlir_dialect_library(MLIRLoop
      LoopDialect.cpp
      LoopOps.cpp
      # LoopOpsEnums.cpp
      LoopTypes.cpp
      ADDITIONAL_HEADER_DIRS
      ../../include/mlir/Dialect/Loop

      DEPENDS
      MLIRLoopOpsIncGen
      LINK_LIBS PUBLIC
      MLIRIR
      MLIRFuncTransforms
      MLIRUtil
      )