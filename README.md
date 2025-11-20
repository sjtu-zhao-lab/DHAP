# DHAP

# Setup
## Prepare docker image and container
Download the docker image from https://zenodo.org/records/15239348/files/dhap-sc25-local-test.tar.gz, and
```
docker load < dhap-sc25-local-test.tar.gz
```
Or just use 
```
docker pull bingp/dhap:local-test
```

Launch the container using
```
docker run --privileged --gpus=all -it \
--network host --hostname dhap --name dhap-local \
bingp/dhap:local-test /bin/bash
```

## Re-build the project (optional)
```
cd /workspace/dhap

cmake -G Ninja . -B build/ \
-DCUCO_DIR=/cuCollections \
-DCXXOPT_DIR=/cxxopts \
-DCMAKE_PREFIX_PATH="/grpc_flight_install/;/build/llvm/install/" \
-DMLIR_DIR=/build/llvm/install/lib/cmake/mlir/ \
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    
cmake --build build/ -j$(nproc)
```

## Convert the csv data to Arrow
```
cd /workspace/dhap/resources/data
csv_convert csv/ssb_1/ arrow/ssb_1/
csv_convert csv/tpcds_1/ arrow/tpcds_1/
csv_convert csv/tpch_1/ arrow/tpch_1/
```

## Launch the memory server
```
cd /workspace/dhap
mem_server --data_dir resources/data/arrow/ssb_1 \
    &> /workspace/mem_server.log &
```

### Launch the memory server for other datasets (please do not execute now!)
```
cd /workspace/dhap & pkill mem_server
mem_server --data_dir resources/data/arrow/tpcds_1 \
    &> /workspace/mem_server.log & 
# or
mem_server --data_dir resources/data/arrow/tpch_1 \ 
    &> /workspace/mem_server.log & 
```

## Create test user
```
useradd -s /bin/bash test
mkdir /workspace/test && chown test:test /workspace/test
su test
source /opt/venv/bin/activate
export PATH=/usr/local/cuda/bin/:$PATH
```

# Execution
## Execute Q21 from SSB
```
bash run.sh \
--sql_file /workspace/dhap/resources/sql/ssb/21.sql \
--data_dir /workspace/dhap/resources/data/arrow/ssb_1
```

## Execute Q21 from SSB with printing result and debugging
```
bash run.sh \
--sql_file /workspace/dhap/resources/sql/ssb/21.sql \
--data_dir /workspace/dhap/resources/data/arrow/ssb_1 \
--print_res --debug
```

## Execute all SSB queries
```
for qid in 21 22 23 31 32 33 34 41 42 43; do
bash run.sh \
--sql_file /workspace/dhap/resources/sql/ssb/${qid}.sql \
--data_dir /workspace/dhap/resources/data/arrow/ssb_1;
done
```

# Analysis
## Execute Q43 from SSB with debugging for analysis
```
rm -r /workspace/test/*
bash run.sh \
--sql_file /workspace/dhap/resources/sql/ssb/43.sql \
--data_dir /workspace/dhap/resources/data/arrow/ssb_1 \
--debug
```

## List intermediate files
```
ls -lrt /workspace/test/
```

## Effect of GPU optimization
### enabled
```
rm -r /workspace/test/*
bash run.sh \
--sql_file /workspace/dhap/resources/sql/ssb/41.sql \
--data_dir /workspace/dhap/resources/data/arrow/ssb_1

grep -o "pair_retrieve" /workspace/test/kernel.cu | wc -l
```

### disabled
```
rm -r /workspace/test/*
DHAP_GOPT_OFF=1 bash run.sh \
--sql_file /workspace/dhap/resources/sql/ssb/41.sql \
--data_dir /workspace/dhap/resources/data/arrow/ssb_1

grep -o "pair_retrieve" /workspace/test/kernel.cu | wc -l
```

# To run queries for other datasets
## TPC-DS
```
exit # switch back to root
cd /workspace/dhap && pkill mem_server
mem_server --data_dir resources/data/arrow/tpcds_1 \
    &> /workspace/mem_server.log & 
su test
source /opt/venv/bin/activate
export PATH=/usr/local/cuda/bin/:$PATH
for qid in 3 42; do \
bash run.sh \
--sql_file /workspace/dhap/resources/sql/tpcds/${qid}.sql \
--data_dir /workspace/dhap/resources/data/arrow/tpcds_1/; \
done
```

## TPC-H
```
exit # switch back to root
cd /workspace/dhap && pkill mem_server
mem_server --data_dir resources/data/arrow/tpch_1 \
    &> /workspace/mem_server.log & 
su test
source /opt/venv/bin/activate
export PATH=/usr/local/cuda/bin/:$PATH
for qid in 3 5; do \
bash run.sh \
--sql_file /workspace/dhap/resources/sql/tpch/${qid}.sql \
--data_dir /workspace/dhap/resources/data/arrow/tpch_1/; \
```