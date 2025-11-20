import argparse

def sum_shflw_send_bw(log_file):
	total_mpi_send_bw = 0
	with open(log_file, 'r') as file:
		for line in file:
			if "MPISend BW" in line:
				bw_value = float(line.split(":")[1].strip().split(" ")[0])
				if bw_value > 50:
					total_mpi_send_bw += bw_value
	return total_mpi_send_bw

def extract_field(log_file, worker_type, field_name):
	field_values = []
	with open(log_file, 'r') as file:
		for line in file:
			if f"[{worker_type} Worker" in line and field_name in line:
				value = line.strip().split(":")[1].strip()
				value = float(value.split(" ")[0])
				field_values.append(value)
	return field_values

def read_last_line(log_file):
    with open(log_file, 'r') as file:
        lines = file.readlines()
        last_line = lines[-1].strip()
        try:
            value = float(last_line)
            return value
        except ValueError:
            print("Error: The last line of the log file is not a valid float value.")
            return None

def parse_scale(config):
	log_file = f"{log_dir}/dhap_{config}.log"
	shflw_log = f"{log_dir}/stage0_{config}.log"

	total_time = read_last_line(log_file)
	
	sync_size_time = extract_field(log_file, "GPU", "ShflRecv-Sync size")
	# print(sync_size_time)
	if len(sync_size_time) == 0:
		sync_size_time = extract_field(log_file, "CPU", "ShflRecv-Sync size")
	avg_sync_time = sum(sync_size_time) / 1000.0 / len(sync_size_time)
	# avg_sync_time = max(sync_size_time) / 1000.0
	est_time = total_time - avg_sync_time

	mpi_waittime = extract_field(log_file, "GPU", "ShflRecv-WaitAll")
	if len(mpi_waittime) == 0:
		mpi_waittime = extract_field(log_file, "CPU", "ShflRecv-WaitAll")
	avg_mpi_wait = sum(mpi_waittime) / 1000.0 / len(mpi_waittime)

	total_mpi_send_bw = sum_shflw_send_bw(shflw_log)

	if args.verb:
		print("Config: ", config)
		print("Total time", total_time)
		print("Avg. sync time: ", avg_sync_time)
		print("Est. time", est_time)
		print("Avg. mpi wait", avg_mpi_wait)
		print("Total MPISend BW:", total_mpi_send_bw)
	
	return est_time

# Example usage:
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--log_dir', type=str, default="logs")
	parser.add_argument('--config', type=str, default=None)
	parser.add_argument('--verb', action='store_true')
	args = parser.parse_args()
	config = args.config
	log_dir = args.log_dir

	if config is not None:
		parse_scale(config)
	else:
		all_est_time = {}
		for config in ["x1", "c2g1", "c3g1", "c4g1", \
						"c1g2", "x2", "c3g2", "c4g2", \
						# "c1g3", "c2g3", "x3", "c4g3", \
						"c1g4", "c2g4", "c3g4", "x4",]:
			est_time = parse_scale(config)
			all_est_time[config] = round(est_time, 2)
		print(all_est_time)
# export DHAP_SCALE=x1 SHFLW=32; bash run.sh --server_ip r6 --plan_dir /home/test/test1 --data_dir /workspace/data/arrow/ssb_1000i/ --sql_file /workspace/dhap/resources/sql/ssb/41.sql --dist --max_numb 304 --max_shflw ${SHFLW} > logs/dhap_s${SHFLW}${DHAP_SCALE}.log && mv logs/stage0.log logs/stage0_s${SHFLW}${DHAP_SCALE}.log