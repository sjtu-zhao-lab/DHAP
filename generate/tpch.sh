#!/usr/bin/env bash
set -x
set -euo pipefail
# TMPDIR=`mktemp --directory`
TMPDIR=/repo/tools/generate/tpch_dbgen
echo $TMPDIR
if [ ! -d $TMPDIR ]; then
	mkdir $TMPDIR
	cp  resources/sql/tpch/initialize.sql $TMPDIR/initialize.sql
	pushd $TMPDIR
	wget -q https://github.com/electrum/tpch-dbgen/archive/32f1c1b92d1664dba542e927d23d86ffa57aa253.zip -O tpch-dbgen.zip
	unzip -q tpch-dbgen.zip
	mv tpch-dbgen-32f1c1b92d1664dba542e927d23d86ffa57aa253/* .
	rm tpch-dbgen.zip
	make
fi
pushd $TMPDIR
./dbgen -f -s $3
ls -la .
for table in ./*.tbl; do  sed -i 's/|$//' "$table"; done

"$1/sql" $2 < initialize.sql
popd

# Usage: bash ./tools/generate/tpch.sh /build/lingodb/ /repo/resources/data/tpch_1 1