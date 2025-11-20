#!/usr/bin/env bash
set -x
set -euo pipefail
# TMPDIR=`mktemp --directory`
TMPDIR=/repo/tools/generate/ssb_dbgen
echo $TMPDIR
if [ ! -d $TMPDIR ]; then
	mkdir $TMPDIR
	cp  resources/sql/ssb/initialize.sql $TMPDIR/initialize.sql
	pushd $TMPDIR
	echo 'd37618c646a6918be8ccc4bc79704061  dbgen.zip' | md5sum --check --status 2>/dev/null || curl -OL https://db.in.tum.de/~fent/dbgen/ssb/dbgen.zip
	echo 'd37618c646a6918be8ccc4bc79704061  dbgen.zip' | md5sum --check --status
	unzip -u dbgen.zip
	mv dbgen/* .
	rm -rf dbgen
	rm dbgen.zip
	rm -rf ./*.tbl
	sed -i 's/#define  MAXAGG_LEN    10/#define  MAXAGG_LEN    20/' shared.h
	make dbgen
fi
pushd $TMPDIR
SF=$3
./dbgen -f -T c -s "$SF"
./dbgen -qf -T d -s "$SF"
./dbgen -qf -T p -s "$SF"
./dbgen -qf -T s -s "$SF"
./dbgen -q -T l -s "$SF"
for table in ./*.tbl; do  sed -i 's/|$//' "$table"; done

"$1/sql" $2 < initialize.sql
popd
echo "Generation over"

# Usage: bash ./tools/generate/ssb.sh /build/lingodb/ /repo/resources/data/ssb_1 1
# Make sure that resources/data/ssb_1 has been created