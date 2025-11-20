add_library(mlir-support OBJECT parsing.cpp eval.cpp offloader.cpp)
target_link_libraries(mlir-support PRIVATE arrow_shared arrow_flight_shared)
