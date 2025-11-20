
add_mlir_dialect_library(MLIRGDSA
      GDSADialect.cpp
      GDSAOps.cpp
      # GDSAOpsEnums.cpp
      GDSATypes.cpp
      ADDITIONAL_HEADER_DIRS
      ../../include/mlir/Dialect/GDSA

      DEPENDS
      MLIRGDSAOpsIncGen
      LINK_LIBS PUBLIC
      MLIRIR
      MLIRFuncTransforms
      MLIRUtil
      )