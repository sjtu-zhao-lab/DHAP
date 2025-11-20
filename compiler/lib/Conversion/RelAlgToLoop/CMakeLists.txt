add_mlir_conversion_library(MLIRRelAlgToLoop
   LowerToLoop.cpp
   UnrollRelAlg.cpp
   InsertShfl.cpp
   FuseLoop.cpp
   GeneratePlan.cpp
   SplitRelAlg.cpp
   Subquery.cpp
   
   ADDITIONAL_HEADER_DIRS
   ../../../mlir/Conversion/RelAlgToLoop

   DEPENDS

   LINK_COMPONENTS
   Core

   LINK_LIBS mlir-support PUBLIC
   MLIRIR
   MLIRRelAlg
   MLIRLoop
   )