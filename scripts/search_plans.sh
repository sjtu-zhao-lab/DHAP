#!/bin/bash
set -e

export DHAP_REPO=$(pwd)
export TIMEFORMAT=%R
LOG_PATH=$1
QID=$2
RESTORE=$3
NUMB=$4
if [ -z "$RESTORE" ]; then
  RESTORE="no"
fi
echo $RESTORE

if [ "$RESTORE" == "naive" ]; then
  TEMP_ERR=$(mktemp)
  export DHAP_NAIVE=1
  for i in {1..2}; do
    DHAP_PLAN=/c16./c16./c16./c16 bash run.sh --server_ip r6 --plan_dir /home/test/test1 --data_dir /workspace/data/arrow/ssb_1000i/ --sql_file "${DHAP_REPO}/resources/sql/ssb/${QID}.sql" --no_print_res --dist --max_numb $NUMB --max_shflw 10 2> $TEMP_ERR
    tail -n 1 $TEMP_ERR >> $LOG_PATH
  done
  rm -f $TEMP_ERR
  exit
fi

if [ "$RESTORE" == "best" ]; then
  unset DHAP_PLAN
  TEMP_ERR=$(mktemp)
  for i in {1..2}; do
    bash run.sh --server_ip r6 --plan_dir /home/test/test1 --data_dir /workspace/data/arrow/ssb_1000i/ --sql_file "${DHAP_REPO}/resources/sql/ssb/${QID}.sql" --no_print_res --dist --max_numb $NUMB --max_shflw 10 2> $TEMP_ERR
    tail -n 1 $TEMP_ERR >> $LOG_PATH
    rm -f $TEMP_ERR
  done
  exit
fi

if [ "$RESTORE" == "pipeline" ]; then
  case "$QID" in
    21|22|23|32|33|34)
      PLANS=("2/g4-c16" "12/c14-c1-c1" "12/g2-g1-g1" "12/c8-c4-c4" "12/g4-c15-c1")
      ;;
    31)
      PLANS=("2/g4-c16" "12/c12-c3-c1" "12/g2-g1-g1" "12/c8-c4-c4" "12/g3-g1-c16")
      ;;
    41|42)
      PLANS=("3/g4-c16" "123/c12-c2-c1-c1" "123/g1-g1-g1-g1" "123/c4-c4-c4-c4" "123/g3-g1-c15-c1")
      ;;
    43)
      PLANS=("2/g4-c16./g4" "12/c14-c1-c1./c16" "12/g2-g1-g1./g4" "12/c8-c4-c4./c16" "12/g4-c15-c1./g4")
      ;;
    esac
  TEMP_ERR=$(mktemp)
  for PLAN in "${PLANS[@]}"; do
    export DHAP_PLAN=$PLAN
    echo $DHAP_PLAN >> $LOG_PATH
    for i in {1..2}; do
      bash run.sh --server_ip r6 --plan_dir /home/test/test1 --data_dir /workspace/data/arrow/ssb_1000i/ --sql_file "${DHAP_REPO}/resources/sql/ssb/${QID}.sql" --no_print_res --dist --max_numb $NUMB --max_shflw 10 2> $TEMP_ERR
      tail -n 1 $TEMP_ERR >> $LOG_PATH
    done
  done
  rm -f $TEMP_ERR
  exit
fi

output=$(python scripts/enumerate_plans.py --qid $QID --restore $RESTORE)
# Convert the Python list output to a Bash array
eval "ALL_PLANS=($(echo $output | sed -e 's/\[//g' -e 's/\]//g' -e 's/,//g' -e 's/\"//g'))"

for PLAN in ${ALL_PLANS[@]}; do
	echo $PLAN >> $LOG_PATH
	for i in {1..2}; do
    TEMP_ERR=$(mktemp)
    DHAP_PLAN=$PLAN bash run.sh --server_ip r6 --plan_dir /home/test/test1 --data_dir /workspace/data/arrow/ssb_1000i/ --sql_file "${DHAP_REPO}/resources/sql/ssb/${QID}.sql" --no_print_res --dist --max_numb $NUMB --max_shflw 10 2> $TEMP_ERR
    tail -n 1 $TEMP_ERR >> $LOG_PATH
    rm -f $TEMP_ERR
  done
done