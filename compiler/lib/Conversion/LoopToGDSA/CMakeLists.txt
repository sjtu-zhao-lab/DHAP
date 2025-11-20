add_mlir_conversion_library(MLIRLoopToGDSA
   LowerToGDSA.cpp
	TranslateLoop.cpp
   GenerateCUDA.cpp
   CUDAGenerator.cpp

   ADDITIONAL_HEADER_DIRS
   ../../../mlir/Conversion/LoopToGDSA

   DEPENDS

   LINK_COMPONENTS
   Core

   LINK_LIBS mlir-support PUBLIC
   MLIRIR
   MLIRRelAlg
   MLIRLoop
	MLIRGDSA
   )