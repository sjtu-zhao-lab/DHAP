
#!/bin/bash
set -e

DHAP_REPO=$(pwd)
export TIMEFORMAT=%R

QID=$1
RESTORE=$2
RESTORE_PATH=$3

NUMB=20
SHFLW=8

if [ -z $RESTORE ]; then
	output=$(python scripts/enumerate_plans.py --qid $QID)
else
	echo Restoring from $RESTORE
	output=$(python scripts/enumerate_plans.py --qid $QID --restore $RESTORE)
fi
# Convert the Python list output to a Bash array
eval "ALL_PLANS=($(echo $output | sed -e 's/\[//g' -e 's/\]//g' -e 's/,//g' -e 's/\"//g'))"

if [ -z $RESTORE ]; then
	LOG_PATH=logs/all_plans/s${QID}_b${NUMB}_shflw${SHFLW}_$(date +%m%d%H%M).log
	echo Plan Config. ID,latency >> $LOG_PATH
else
	LOG_PATH=$RESTORE_PATH
fi
for PLAN in ${ALL_PLANS[@]}; do
	for i in {1..1}; do
		RETRIES=3
		ATTEMPT=0
		SUCCESS=0

		while [[ $ATTEMPT -lt $RETRIES ]]; do
			ATTEMPT=$((ATTEMPT + 1))
			echo "Attempt $ATTEMPT of $RETRIES for plan: $PLAN"
			
			if DHAP_PLAN=$PLAN bash run.sh --server_ip r6 --plan_dir /home/test/test1 \
            --data_dir /workspace/data/arrow/ssb_1000i/ \
            --sql_file "${DHAP_REPO}/resources/sql/ssb/${QID}.sql" \
            --dist --max_numb $NUMB --max_shflw $SHFLW; then
                SUCCESS=1
                break
            else
                echo "Command failed, retrying..."
            fi
		done
		# Exit if all retries failed
		if [[ $SUCCESS -eq 0 ]]; then
			echo "All attempts failed for plan: $PLAN"
			exit 1
		fi

		lat=$(tail -n 1 logs/dhap.err)
		echo $PLAN,$lat >> $LOG_PATH
	done
done