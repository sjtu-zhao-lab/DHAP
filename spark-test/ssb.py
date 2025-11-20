from pyspark.sql import SQLContext
from pyspark import SparkConf
from pyspark import SparkContext
import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd
import time
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, BooleanType, DateType, TimestampType, DecimalType

conf = SparkConf()
sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

RETRY = 2
SF = 1000
NUMB = 152

qid = [11, 12, 13]
qid = [21, 22, 23, 31, 32, 33, 34, 41, 42, 43]
NUMR = -1

table_names = ['lineorder', 'customer', 'part', 'supplier', 'date']
schema = {
'lineorder': StructType([
	StructField("lo_orderkey", IntegerType(), True),
	StructField("lo_linenumber", IntegerType(), True),
	StructField("lo_custkey", IntegerType(), True),
	StructField("lo_partkey", IntegerType(), True),
	StructField("lo_suppkey", IntegerType(), True),
	StructField("lo_orderdate", IntegerType(), True),
	StructField("lo_orderpriority", StringType(), True),
	StructField("lo_shippriority", StringType(), True),
	StructField("lo_quantity", IntegerType(), True),
	StructField("lo_extendedprice", DecimalType(18, 2), True),
	StructField("lo_ordtotalprice", DecimalType(18, 2), True),
	StructField("lo_discount", IntegerType(), True),
	StructField("lo_revenue", DecimalType(18, 2), True),
	StructField("lo_supplycost", DecimalType(18, 2), True),
	StructField("lo_tax", IntegerType(), True),
	StructField("lo_commitdate", IntegerType(), True),
	StructField("lo_shipmode", StringType(), True)
]),
'part': StructType([
    StructField("p_partkey", IntegerType(), True),
    StructField("p_name", StringType(), True),
    StructField("p_mfgr", StringType(), True),
    StructField("p_category", StringType(), True),
    StructField("p_brand1", StringType(), True),
    StructField("p_color", StringType(), True),
    StructField("p_type", StringType(), True),
    StructField("p_size", IntegerType(), True),
    StructField("p_container", StringType(), True)
]),
'customer': StructType([
    StructField("c_custkey", IntegerType(), True),
    StructField("c_name", StringType(), True),
    StructField("c_address", StringType(), True),
    StructField("c_city", StringType(), True),
    StructField("c_nation", StringType(), True),
    StructField("c_region", StringType(), True),
    StructField("c_phone", StringType(), True),
    StructField("c_mktsegment", StringType(), True)
]),
'supplier': StructType([
    StructField("s_suppkey", IntegerType(), True),
    StructField("s_name", StringType(), True),
    StructField("s_address", StringType(), True),
    StructField("s_city", StringType(), True),
    StructField("s_nation", StringType(), True),
    StructField("s_region", StringType(), True),
    StructField("s_phone", StringType(), True)
]),
'date': StructType([
    StructField("d_datekey", IntegerType(), True),
    StructField("d_date", StringType(), True),
    StructField("d_dayofweek", StringType(), True),
    StructField("d_month", StringType(), True),
    StructField("d_year", IntegerType(), True),
    StructField("d_yearmonthnum", IntegerType(), True),
    StructField("d_yearmonth", StringType(), True),
    StructField("d_daynuminweek", IntegerType(), True),
    StructField("d_daynuminmonth", IntegerType(), True),
    StructField("d_daynuminyear", IntegerType(), True),
    StructField("d_monthnuminyear", IntegerType(), True),
    StructField("d_weeknuminyear", IntegerType(), True),
    StructField("d_sellingseason", StringType(), True),
    StructField("d_lastdayinweekfl", IntegerType(), True),
    StructField("d_lastdayinmonthfl", IntegerType(), True),
    StructField("d_holidayfl", IntegerType(), True),
    StructField("d_weekdayfl", IntegerType(), True)
])
}

read_csv = True
spark_dfs = {}
for table_name in table_names:
	if read_csv:
		spark_dfs[table_name] = sqlContext.read.format("csv") \
																					.option("sep", "|") \
																					.option("header", "false") \
    																			.schema(schema[table_name]) \
																					.load(f'data/csv/ssb_{SF}/{table_name}.tbl')
		if NUMR > 0:
			spark_dfs[table_name].limit(NUMR)
	else:
		read_opt = ipc.IpcReadOptions()
		if table_name == 'lineorder':
			if qid[0] == 11:
				read_opt.included_fields = [5, 8, 9, 11]
			else:
				read_opt.included_fields = [2, 3, 4, 5, 12, 13]
		# with open(f'data/arrow/ssb_{SF}i/{table_name}.arrow', 'r') as arrow_file:
		with ipc.open_file(f'data/arrow/ssb_{SF}i/{table_name}.arrow', options=read_opt) as reader:
			# reader = ipc.RecordBatchFileReader(source, options=read_opt)
			num_batches = reader.num_record_batches
			num_read_batches = min(num_batches, NUMB)
			batches = []
			for b in range(num_read_batches):
				print(f'Reading batch {b} / {num_read_batches} / {num_batches}')
				batches.append(reader.get_batch(b))
			# arrow_table = reader.read_all()
			arrow_table = pa.Table.from_batches(batches)
			print("Table combination finished")
			# pandas_df = arrow_table.to_pandas(split_blocks=True, self_destruct=True)
			pandas_df = arrow_table.to_pandas()
			print(pandas_df)
			spark_dfs[table_name] = sqlContext.createDataFrame(pandas_df, verifySchema=False)
	spark_dfs[table_name].createOrReplaceTempView(table_name)

time_file = f'test/ssb{SF}_r{NUMR}_time.log' if read_csv else f'test/ssb{SF}_b{NUMB}_time.log'
with open(time_file, 'a') as time_record:
	for id in qid:
		with open(f'ssb/{id}.sql', 'r') as qfile:
			query = qfile.read()
		test_time = []
		for t in range(RETRY):
			st = time.time()
			result = sqlContext.sql(query)
			result.collect()
			ed = time.time()
			test_time.append(ed-st)
			# if t == RETRY-1:
			# 	result.show()
		avg_time = sum(test_time)	/ len(test_time)
		for t in test_time:
			time_record.write(f'Q{id}: {t:.2f}s\n')
		time_record.write(f'Q{id}: {avg_time:.2f}s (avg)\n')
		time_record.flush()
	