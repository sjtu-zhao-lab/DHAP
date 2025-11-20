#!/usr/bin/env bash

set -x

DS=$1
SF=$2
STO_ROOT=$3
STO_DIR="${STO_ROOT}/${DS}_${SF}"

if [ ! -d $STO_DIR ]; then
	mkdir $STO_DIR
fi

cp -r "${DS}_dbgen" $STO_DIR

if [ $DS == "ssb" ]; then
	pushd "${STO_DIR}/${DS}_dbgen"
	./dbgen -qf -T a -s "$SF" 
fi
if [ $DS == "tpch" ]; then
	pushd "${STO_DIR}/${DS}_dbgen"
	./dbgen -qf -s "$SF" 
fi

if [ $DS == "tpcds" ]; then
	pushd "${STO_DIR}/${DS}_dbgen/tools"
	./dsdgen -FORCE -SCALE "$SF" 
	for file in *.dat; do
		mv "$file" "${file%.dat}.tbl"
	done
fi

chmod 666 ./*.tbl
for table in ./*.tbl; do  sed -i 's/|$//' "$table"; done
mv ./*.tbl $STO_DIR

popd
rm -r "${STO_DIR}/${DS}_dbgen"

# Usage: `bash ssb_dbgen.sh [ssb,tpch,tpcds] 1 /workspace/data/csv/` 
# or `bash ssb_dbgen.sh [ssb,tpch,tpcds] 1 /workspace/dhap/resources/data/csv/`
# Then have a metadata.json under the database dir (by sql initialize.sql or copy)
# Then use disagg_plat/build/csv_convert to transform into .arrow format