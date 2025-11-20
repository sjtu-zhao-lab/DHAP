# DHAP

## Dependencies

- LLVM
- Apache Arrow (12.0)
	- gRPC and Arrow Flight RPC
	-	has been installed in `deps/grpc_flight_install`
- cuCollections (commit 8b07be33)
- OpenMPI 4.1.5
- UCX (for the multi-node RDMA setting)
- cxxopts

For single-node local testing, you can directly use our pre-built docker image (without UCX)
```
docker pull bingp/dhap:local
docker run --privileged --gpus=all -it -v ~/dhap:/workspace/dhap \
		--hostname dhap --name dhap-container bingp/dhap:local
```

Alternatively, you can build it from the `Dockerfile`
```
docker build -t bingp/dhap:local -f Dockerfile .
```

## Build

To build the whole project
```
cd /workspace/dhap
mkdir -p build
cmake -G Ninja . -B build/ -DCUCO_DIR=/cuCollections -DCXXOPT_DIR=/cxxopts \
	-DCMAKE_PREFIX_PATH="/grpc_flight_install/;/build/llvm/install/" \
	-DMLIR_DIR=/build/llvm/install/lib/cmake/mlir/ -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/ -j$(nproc)
```

> The paths indicated above is from our pre-built docker image

And export the binary paths for later use
```
export PATH=/workspace/dhap/build/:$PATH
```

## Data Preparation

DHAP require data in Apache Arrow format, we provide a tool to convert csv data into Arrow format. Take `resources/data/csv/exp-test/` as example
```
cd resources/data/ && mkdir -p arrow/exp-test
cp csv/exp-test/metadata.json arrow/exp-test/ && csv_convert csv/exp-test/ arrow/exp-test/
```

For the SSB dataset
```
mkdir -p arrow/ssb-test
cp csv/ssb-test/metadata.json arrow/ssb-test/ && csv_convert csv/ssb-test/ arrow/ssb-test/
```

Then the Arrow data will be prepared in `resources/data/arrow/exp-test/`

## Running

First, run the storage server the with specific data directory
```
cd /workspace/dhap
mem_server --data_dir resources/data/arrow/exp-test/ &> /workspace/mem.log &
```

For the SSB dataset
```
pkill mem_server	# kill the previous storage server
cd /workspace/dhap
mem_server --data_dir resources/data/arrow/ssb-test/ &> /workspace/mem.log &
```

Next, we create and switch to a test user (since OpenMPI cannot be executed by root user), and create a output directory for the test user
```
useradd -s /bin/bash test
mkdir /workspace/test && chown test:test /workspace/test
su test
source /opt/venv/bin/activate && export PATH=/usr/local/cuda/bin/:$PATH
```

Finally, you can run the DHAP by
```
cd dhap
bash run.sh
```

The `run.sh` support the following configuration arguments
- `--sql-file`: Absolute path of the SQL file to be executed
- `--data-dir`: Absolute path of the data needed (same as the one for `mem_server`)
- `--plan-dir`: Absolute path for the outputs, where the execution plan will be located here
- `--server-ip`: IP address of running storage server
- `--num_cpu_worker`: Number of CPU workers
- `--num_gpu_worker`: Number of GPU workers
- `--l_cpu`: Memory limitation of CPU in MB
- `--l_gpu`: Memory limitation of GPU in MB
- `--max_gpu_stage`: Max. number of stages that are allowed to be executed by GPU

The default config (when running `bash run.sh`) is
```
bash run.sh --sql_file /workspace/dhap/resources/sql/join3.sql \
		--data_dir /workspace/dhap/resources/data/arrow/exp-test \
		--plan_dir /workspace/test --server_ip localhost \
		--num_cpu_worker 2  --num_gpu_worker 2 \
		--l_cpu 1  --l_gpu 0.1 --max_gpu_stage 2
```

For the SSB dataset 
```
bash run.sh --sql_file /workspace/versa-cube/resources/sql/ssb21a.sql --data_dir /workspace/versa-cube/resources/data/arrow/ssb-test
```
