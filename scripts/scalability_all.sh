#!/bin/bash

DHAP_REPO=$(pwd)

export QID=$1
export SHFLW=32

LOG_DIR=logs/scale_s${QID}_shflw${SHFLW}_$(date +%m%d%H%M)
mkdir $LOG_DIR

ALL_SCALE=("x1" "c2g1" "c3g1" "c4g1" \
			"c1g2" "x2" "c3g2" "c4g2" \
			"c1g4" "c2g4" "c3g4" "x4" )
for scale in ${ALL_SCALE[@]}; do
	echo $scale
	export DHAP_SCALE=$scale 
	bash run.sh --server_ip r6 --plan_dir /home/test/test1 \
		--data_dir /workspace/data/arrow/ssb_1000i/ \
		--sql_file ${DHAP_REPO}/resources/sql/ssb/$QID.sql \
		--dist --max_numb 304 --max_shflw ${SHFLW} \
		> ${LOG_DIR}/dhap_${DHAP_SCALE}.log 
	mv logs/stage0.log ${LOG_DIR}/stage0_${DHAP_SCALE}.log
done

ALL_SCALE=("c1g3" "c2g3" "x3" "c4g3" )
for scale in ${ALL_SCALE[@]}; do
	echo $scale
	export DHAP_SCALE=$scale SHFLW=32
	bash run.sh --server_ip r6 --plan_dir /home/test/test1 \
		--data_dir /workspace/data/arrow/ssb_1000i/ \
		--sql_file ${DHAP_REPO}/resources/sql/ssb/$QID.sql \
		--dist --max_numb 304 --max_shflw ${SHFLW} --p0 6 \
		> ${LOG_DIR}/dhap_${DHAP_SCALE}.log 
	mv logs/stage0.log ${LOG_DIR}/stage0_${DHAP_SCALE}.log
done
