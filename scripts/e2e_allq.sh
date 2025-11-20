#!/bin/bash

DHAP_REPO=$(pwd)

SHFLW=8
NUMB=$1

if [ -v DHAP_NAIVE ]; then
	echo "Using Naive"
	export DHAP_NAIVE=1
	export NOT_MERGE_RES=1
	LOG_DIR=logs/e2e_naive/ssb_shflw${SHFLW}_b${NUMB}_naive_$(date +%m%d%H%M)
else
	if [ -v DHAP_CPU_ONLY ]; then
		export DHAP_CPU_ONLY=1
		LOG_DIR=logs/e2e_cpu/ssb_shflw${SHFLW}_b${NUMB}_naive_$(date +%m%d%H%M)
	else
		LOG_DIR=logs/e2e/ssb_shflw${SHFLW}_b${NUMB}_$(date +%m%d%H%M)
	fi
fi
mkdir $LOG_DIR

ALL_Q=("21" "22" "23" "31" "32" "33" "34" "41" "42" "43")
ALL_Q=("31" "41" "42")
ALL_Q=("21" "22" "23" "32" "33" "34" "43")
for qid in ${ALL_Q[@]};
	do
	echo S$qid;
	bash run.sh --server_ip r6 --plan_dir /home/test/test1 \
		--data_dir /workspace/data/arrow/ssb_1000i/ \
		--sql_file ${DHAP_REPO}/resources/sql/ssb/$qid.sql \
		--dist --max_numb ${NUMB} --max_shflw ${SHFLW} \
		> ${LOG_DIR}/${qid}.log
done

