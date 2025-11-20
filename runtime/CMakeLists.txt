cmake_minimum_required(VERSION 3.10)

project(dhap-rt LANGUAGES CXX CUDA)

# set(CMAKE_PREFIX_PATH "/grpc_flight_install" "/build/llvm/install" ${CMAKE_PREFIX_PATH})

if(NOT DEFINED CMAKE_CXX_STANDARD)
set(CMAKE_CXX_STANDARD 20)
endif()
message(STATUS ${CMAKE_CXX_STANDARD})

# We require a C++17 compliant compiler
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(include)
include_directories(/usr/local/cuda-11.8/include)
include_directories("${CUCO_DIR}/include")
include_directories("${CXXOPT_DIR}/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../compiler/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../compiler/vendored")

find_package(MLIR REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

option(ARROW_LINK_SHARED "Link to the Arrow shared library" ON)
find_package(Arrow REQUIRED)
find_package(ArrowFlight REQUIRED)
find_package(ArrowAcero REQUIRED)
# find_package(ArrowCUDA REQUIRED)
message(STATUS "Arrow version: ${ARROW_VERSION}")
message(STATUS "Arrow SO version: ${ARROW_FULL_SO_VERSION}")

find_package(CUDA REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda --std=c++17")

find_package(CUDAToolkit REQUIRED)

find_package(MPI REQUIRED)

set(LINGO_RT_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../compiler/lib/runtime")
add_library(lingo_runtime OBJECT
        "${LINGO_RT_SRC}/TableBuilder.cpp"
        "${LINGO_RT_SRC}/DumpRuntime.cpp"
        "${LINGO_RT_SRC}/DataSourceIteration.cpp"
        "${LINGO_RT_SRC}/Vector.cpp"
        "${LINGO_RT_SRC}/LazyJoinHashtable.cpp"
        "${LINGO_RT_SRC}/Hashtable.cpp"
        "${LINGO_RT_SRC}/StringRuntime.cpp"
        "${LINGO_RT_SRC}/Hash.cpp"
        "${LINGO_RT_SRC}/DateRuntime.cpp"
        "${LINGO_RT_SRC}/ExecutionContext.cpp"
        "${LINGO_RT_SRC}/Database.cpp"
				"${LINGO_RT_SRC}/MetaData.cpp"
				"${LINGO_RT_SRC}/ArrowDirDatabase.cpp"
				"${LINGO_RT_SRC}/ExternalArrowDatabase.cpp"
				"${LINGO_RT_SRC}/MetaDataOnlyDatabase.cpp")
target_link_libraries(lingo_runtime PUBLIC Arrow::arrow_shared)

add_subdirectory(src)
