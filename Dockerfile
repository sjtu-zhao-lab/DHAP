FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS base
RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata vim tmux curl wget git ninja-build
RUN apt-get -y install python3-pip python3-venv

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip3 install --upgrade pip
RUN pip3 install cmake
RUN pip3 install numpy
RUN pip3 install Cython
RUN pip3 install pybind11
RUN pip3 install scipy

RUN apt-get -y install autoconf flex bison libjemalloc-dev libboost-dev libboost-filesystem-dev libboost-system-dev libboost-regex-dev python3.8-dev libssl-dev

FROM base AS buildllvm
COPY deps/llvm-project/ /workspace/llvm-project
WORKDIR /workspace/llvm-project
RUN mkdir build
RUN cmake -G Ninja llvm/  -B /build/llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang;clang-tools-extra" \
    -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi"\
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="X86;" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DPython3_FIND_VIRTUALENV=ONLY \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON
RUN cmake --build /build/llvm -j32
RUN cmake --install /build/llvm --prefix /build/llvm/install

FROM base AS local-deps
COPY --from=buildllvm /build/llvm/install /build/llvm/install
COPY deps/cxxopts /cxxopts
COPY deps/cuCollections /cuCollections
# To build gRPC:
# cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=$MY_INSTALL_DIR
# make -j16 && make install
# To build Arrow:
# cmake .. -DCMAKE_INSTALL_PREFIX=$MY_INSTALL_DIR -DARROW_FLIGHT=ON -DARROW_FILESYSTEM=ON -DARROW_CSV=ON -DARROW_COMPUTE=ON -DARROW_ACERO=ON -DARROW_PARQUET=ON
# cmake --build . && cmake --install .

# use pre-built grpc_flight instead
COPY deps/grpc_flight_install /grpc_flight_install

WORKDIR /workspace
RUN wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz
RUN tar xf openmpi-4.1.5.tar.gz && rm openmpi-4.1.5.tar.gz
WORKDIR /workspace/openmpi-4.1.5
RUN ./configure --with-cuda=/usr/local/cuda-11.8
RUN make -j32 && make install && ldconfig

FROM local-deps AS build-dhap
COPY . /workspace/dhap
RUN rm -rf /workspace/dhap/deps
WORKDIR /workspace/dhap
RUN mkdir -p build
RUN cmake -G Ninja . -B build/ -DCUCO_DIR=/cuCollections -DCXXOPT_DIR=/cxxopts \
	-DCMAKE_PREFIX_PATH="/grpc_flight_install/;/build/llvm/install/" \
	-DMLIR_DIR=/build/llvm/install/lib/cmake/mlir/ -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
RUN cmake --build build/ -j$(nproc)
ENV PATH=/workspace/dhap/build/:$PATH