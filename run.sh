#!/bin/bash

set -e

export DHAP_REPO=$(pwd)
SQL_PATH=${DHAP_REPO}/resources/sql/join3.sql
DATA_DIR=${DHAP_REPO}/resources/data/arrow/exp-test
PLAN_DIR=/workspace/test
export DHAP_LOG_DIR=/workspace/test/logs
mkdir -p $DHAP_LOG_DIR

CUCO_PATH=/cuCollections
export PATH=${DHAP_REPO}/build/:$PATH

export SR_IP=localhost
export L_CPU=8			# in GB
export L_GPU=8		    # in GB

MAX_NUMB=0
MAX_SHFLW=0
P0=8

DIST=0
DIST_DBG=0
PRINT_RES=0

TIMEFORMAT=%R
# Parse named arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sql_file)
            SQL_PATH="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --plan_dir)
            PLAN_DIR="$2"
            shift 2
            ;;
        --server_ip)
            export SR_IP="$2"
            shift 2
            ;;
        --l_cpu)
			export L_CPU="$2"
            shift 2
            ;;
        --l_gpu)
			export L_GPU="$2"
            shift 2
            ;;
        --ro)
            RUN_ONLY=1
            shift 1
            ;;
        --p0)
			P0="$2"
            shift 2
            ;;
        --max_shflw)
			MAX_SHFLW="$2"
            shift 2
            ;;
        --max_numb)
			MAX_NUMB="$2"
            shift 2
            ;;
        --dist)
            DIST=1
            shift 1
            ;;
        --debug)
            export DHAP_QC_DEBUG=DEBUGGING
            shift 1
            ;;
        --dist_debug)
            export DHAP_QC_DEBUG=DEBUGGING
            DIST_DBG=1
            shift 1
            ;;
        --print_res)
            PRINT_RES=1
            shift 1
            ;;
        *)
            echo "Invalid option: $1" >&2
            exit 1
            ;;
    esac
done
if [[ "$SQL_PATH" == *"tpch"* ]]; then
    export USE_TPCH=1
fi

mkdir -p $PLAN_DIR
echo "========================================================================================="
cat $SQL_PATH
echo ""
echo "========================================================================================="
pushd $PLAN_DIR > /dev/null
if [ -n "${RUN_ONLY}" ]; then
    echo "No code generation"
else
    OLD_CUDA_MD5=$(md5sum <(cat kernel.cu))
    echo "Starting DHAP code generation ..."
    time dhap-compiler $SQL_PATH $DATA_DIR
    echo "DHAP code generation finshed, starting cubin compilation ..."
    NEW_CUDA_MD5=$(md5sum <(cat kernel.cu))
    if [[ "$OLD_CUDA_MD5" != "$NEW_CUDA_MD5" && $DIST_DBG -eq 0 ]]; then
        nvcc --cubin -o kernel.cubin kernel.cu -arch=sm_80 --expt-relaxed-constexpr \
            --extended-lambda -I${CUCO_PATH}/include/ -I${DHAP_REPO}/runtime/include
    fi
    echo "DHAP compilation finished, starting execution ..."
fi
NUM_SUBQUERY=$(cat NUM_SUBQUERY)
echo "# sub-queries: ${NUM_SUBQUERY}"
popd > /dev/null
commands=()
for ((SUB_QUERY=0; SUB_QUERY<$NUM_SUBQUERY; SUB_QUERY++)); do
    commands[${#commands[@]}]=$(python3 scripts/regular.py --plan_dir $PLAN_DIR \
        --num_subq $NUM_SUBQUERY --sub_query $SUB_QUERY --p0 $P0 \
        --max_numb $MAX_NUMB --max_shflw $MAX_SHFLW \
        --dist $DIST --dist_debug $DIST_DBG --print_res $PRINT_RES)
done
if [ $DIST -eq 1 ] && [ $DIST_DBG -eq 0 ]; then
    ${HOME}/plan_sync.sh $PLAN_DIR/
fi
total_time=0
# temp_file=$(mktemp)
rm -f ${DHAP_LOG_DIR}/stage0.log
temp_file=${DHAP_LOG_DIR}/dhap.err
for ((SUB_QUERY=0; SUB_QUERY<$NUM_SUBQUERY; SUB_QUERY++)); do
    echo "Running sub-query ${SUB_QUERY}"
    command=${commands[SUB_QUERY]}
    echo $command
    if [ $DIST_DBG -eq 0 ]; then
        eval $command 2> $temp_file
        echo "Subquery Finish"
        subq_time=$(tail -1 $temp_file)
        echo $subq_time
        total_time=$(awk "BEGIN {print $total_time + $subq_time}")
    fi
done
# rm -f $temp_file
echo $total_time >&1
echo $total_time >&2
echo $total_time >> $temp_file

# Usage
# [DHAP_PLAN=3/g4-c16./c16] [DHAP_NAIVE=1] [DHAP_UCX_RNDV=1] 
# [NROWS_1TIME=xx] [DHAP_GOPT_OFF=1] [DHAP_OFFLOAD_OFF=1]
# [DHAP_SCALE="x1"]
# bash run.sh --server_ip r6 --plan_dir /home/test/test1 --data_dir /workspace/data/arrow/ssb_1000i/ --sql_file /workspace/dhap/resources/sql/ssb/41.sql --dist --max_numb 152 --max_shflw 20 