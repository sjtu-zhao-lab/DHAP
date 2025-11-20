import json, sys, os, argparse, sys

# MAX_SLOTS = {"cpu": 16, "gpu": 4}
# MAX_SLOTS = {"cpu": 4, "gpu": 1}
# MAX_SLOTS = {"cpu": 4, "gpu": 1}
MAX_SLOTS = {"cpu": 8, "gpu": 2}
# MAX_SLOTS = {"cpu": 4, "gpu": 2}
# MAX_SLOTS = {"cpu": 1, "gpu": 2}

def set_scale():
	# ALL_MACHINES = {"cpu": ["r1d", "r7d"], "gpu": ["r4d", "r5d"]}
	ALL_MACHINES = {"cpu": ["r1c", "r7c", "r5c", "r4c"], 
					"gpu": ["r1d", "r7d", "r5d", "r4d"]}
	# ALL_MACHINES = {"cpu": ["r1d", "r7d", "r5d", "r4d"], 
	# 				"gpu": ["r1d", "r7d", "r5d", "r4d"]}
	scale = os.getenv("DHAP_SCALE")
	if os.getenv("DHAP_NAIVE") is not None or os.getenv("DHAP_CPU_ONLY") is not None:
		MACHINES = {"cpu": ALL_MACHINES["cpu"], "gpu": []}
	elif scale is None:
		MACHINES = ALL_MACHINES
	elif scale[0] == 'x':
		# x1, x2, x3, x4
		if len(scale) == 2:
			num = int(scale[1])
			MACHINES = {"cpu": ALL_MACHINES["cpu"][:num], "gpu": ALL_MACHINES["gpu"][:num]}
	elif scale[0] == 'c':
		assert len(scale) == 4 and scale[2] == 'g'
		cnum = int(scale[1])
		gnum = int(scale[3])
		MACHINES = {"cpu": ALL_MACHINES["cpu"][:cnum], "gpu": ALL_MACHINES["gpu"][:gnum]}
	else:
		assert False
	return MACHINES

MACHINES = set_scale()

def update_num_workers(plan, sub_q, max_num_batches, max_shfl_worker, num_subq):
	# Num. of shuffle workers (equal to #batches now)
	plan["num_shfl_workers"] = {}
	shfl_tables = plan["shfl_target_stage"].keys()
	total_shfl_workers = 0
	for shfl_table in shfl_tables:
		assert shfl_table in plan["num_batches"]
		if max_num_batches != 0 and sub_q == 0:
			plan["num_batches"][shfl_table] = min(max_num_batches, plan["num_batches"][shfl_table])
		if max_shfl_worker == 0:
			plan["num_shfl_workers"][shfl_table] = plan["num_batches"][shfl_table] 
		else:
			if os.getenv("DHAP_NAIVE") and sub_q == 0:		# limit the cpu used by naive
				max_shfl_worker = max_shfl_worker // num_subq + 1
			plan["num_shfl_workers"][shfl_table] = min(max_shfl_worker, plan["num_batches"][shfl_table])
		total_shfl_workers += plan["num_shfl_workers"][shfl_table]
	# Update all
	plan["num_stage_workers"] = [total_shfl_workers] + plan["worker_num"][1:] + [1]


def update_num_partitions(plan, p0):
	num_workers = plan["num_stage_workers"]
	while p0 % num_workers[1] != 0:
		p0 += 1
	num_stages = len(num_workers)
	num_out_partitions = [0] * num_stages
	num_out_partitions[0] = p0
	for s in range(1, num_stages-2) :
		# ps = num_workers[s+1]
		# to ensure the partition size and number keep stable
		# assert (p0 * num_workers[s]) % num_out_partitions[s-1] == 0
		ps = p0 * num_workers[s] // num_out_partitions[s-1]
		# ps = 2
		while ps % num_workers[s+1] != 0 or ps == 0:
			ps += 1
		num_out_partitions[s] = ps

	num_out_partitions[num_stages-2] = 1		# to collect the results
	num_out_partitions[num_stages-1] = 0		# the final collect worker
	plan["stage_out_partitions"] = num_out_partitions

def update_res_num_batches(plan, sub_q, num_subq, plan_dir):
	last_knl = plan["stage_kernel_info"][-1]
	assert "result_name" in last_knl
	res_name = last_knl["result_name"]
	first_knl = plan["stage_kernel_info"][1]
	probe_key = first_knl["probe_key"][0]
	probe_tbl = ""
	for k in plan["shfl_worker_partition_col"]:
		if plan["shfl_worker_partition_col"][k] == probe_key:
			probe_tbl = k
	assert probe_tbl != ""
	if os.getenv("NOT_MERGE_RES"):
		# Note that the storage server needs to be restarted with it !!!
		res_num_batch = (plan["num_batches"][probe_tbl]-1) // plan["num_shfl_workers"][probe_tbl] + 1
		num_partitions = plan["stage_out_partitions"]
		num_workers = plan["num_stage_workers"]
		for s in range(len(num_partitions)-2):
			res_num_batch = res_num_batch * num_partitions[s] // num_workers[s+1]
		res_num_batch *= num_workers[-2]
	else:
		res_num_batch = 1
	plan["num_batches"][res_name] = res_num_batch
	if sub_q + 1 < num_subq:
		next_plan_path = os.path.join(plan_dir, 'plan'+str(sub_q+1)+'.json')
		with open(next_plan_path, "r") as next_plan_file:
			next_plan = json.load(next_plan_file)
		next_plan["num_batches"][res_name] = res_num_batch
		with open(next_plan_path, "w") as next_plan_file:
			json.dump(next_plan, next_plan_file, indent=2)

def prepare_env(plan_dir, dist):
	with open(os.path.join(plan_dir, "env.tune"), "w") as env_file:
		if dist:
			env_file.write("-mca pml ucx -x UCX_LOG_LEVEL=ERROR\n")
			if os.getenv("DHAP_UCX_RNDV"):
				env_file.write("-x UCX_RNDV_SCHEME=get_zcopy\n")
		export_env_list = ["SR_IP", "L_CPU", "L_GPU", "NROWS_1TIME", "DHAP_LOG_DIR"]
		if os.getenv("DHAP_NAIVE"):
			env_file.write("-x DHAP_NAIVE=1\n")
		for env_name in export_env_list:
			env = os.getenv(env_name)
			if env is not None:
				env_file.write(f"-x {env_name}={env}\n")
		env_file.write(f"-x PLAN_DIR={plan_dir}\n")
		env_file.write(f"-x SUB_QUERY=0\n")

def append_cmd_workers(dist: int, host: str, sub_q: int, num: int, type: str, worker_args: str = '') -> str:
	host2mlx = {
		"r6": 1, "r1": 1, "r4": 1, "r5": 0, "r7": 1
	}
	assert type == 'cpu' or type == 'gpu'
	cmd = ''
	if sub_q > 0:
		cmd += f"-x SUB_QUERY={sub_q} "
	if dist:
		cmd += f"-host {host} "
		repo = os.getenv("DHAP_REPO")
		assert repo is not None
		if not args.dist_debug:
			cmd += f"-path {repo}/build "
			# Slice "r1" from "r1d" and "r1c"
			cmd += f"-x UCX_NET_DEVICES=mlx5_{host2mlx[host[:2]]}:1 "
			if os.getenv("DHAP_UCX_RNDV"):
				if type == 'cpu':
					cmd += "-x UCX_RNDV_SCHEME=put_zcopy "
				elif type == 'gpu':
					cmd += "-x UCX_RNDV_SCHEME=put_ppln "
	cmd += '-n ' + str(num) + ' ' + type + '_worker ' + worker_args + ' : '
	return cmd

def allocate_workers_balanced(n, machines, max_slot, cur_slots, host_list, t):
	num_machines = len(machines)
	# Step 1: Calculate even distribution
	workers_per_machine = [n // num_machines] * num_machines
	remainder_workers = n % num_machines

	# Distribute remainder evenly
	for i in range(remainder_workers):
		workers_per_machine[i] += 1
	# Step 2: Adjust allocation to respect max_slot on each machine
	for i in range(num_machines):
		machine = machines[i]
		if cur_slots[machine] + workers_per_machine[i] > max_slot:
			# If the current machine cannot take all workers assigned, adjust
			excess_workers = cur_slots[machine] + workers_per_machine[i] - max_slot
			workers_per_machine[i] -= excess_workers  # Adjust current machine's allocation
			remainder_workers = excess_workers  # Handle excess workers
			# Try to redistribute excess workers to other machines
			for j in range(i+1, num_machines):
				machine_j = machines[j]
				available_slots = max_slot - cur_slots[machine_j]
				if available_slots >= remainder_workers:
					workers_per_machine[j] += remainder_workers
					remainder_workers = 0
					break
				else:
					workers_per_machine[j] += available_slots
					remainder_workers -= available_slots
	# Step 3: Allocate workers to each machine and update `cur_slots`
	cmd = ""
	for i in range(num_machines):
		machine = machines[i]
		n_workers = workers_per_machine[i]
		assert cur_slots[machine] + n_workers <= max_slot
		if n_workers > 0:
			host = host_list[i]
			cmd += append_cmd_workers(1, host, args.sub_query, n_workers, t)
			cur_slots[machine] += n_workers

	return cmd

def arrange_dist_workers(nums, types):
	cur_slots = {m: 0 for m in MACHINES["cpu"] + MACHINES["gpu"]}
	machine_idx = {"cpu": 0, "gpu": 0}

	cmd = ""
	for (n, t) in zip(nums, types):
		# print("num", n, "type", t, file=sys.stderr)
		num_machines = len(MACHINES[t])
		max_slot = MAX_SLOTS[t]

		machines = [MACHINES[t][(machine_idx[t] + i) % num_machines] for i in range(num_machines)]
		hosts = [f"{machine}:{max_slot}" for machine in machines]
		# print("machines", machines, file=sys.stderr)
		# print("cur_slots", cur_slots[machine], cur_slots[machine1], cur_slots[machine2], file=sys.stderr)
		# worker allocation balance
		cmd += allocate_workers_balanced(n, machines, max_slot, cur_slots, hosts, t)
	return cmd

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--plan_dir', required=True)
	parser.add_argument('--p0', type=int, default=8)
	parser.add_argument('--sub_query', type=int, required=True)
	parser.add_argument('--num_subq', type=int, required=True)
	parser.add_argument('--max_numb', type=int, default=0)
	parser.add_argument('--max_shflw', type=int, default=0)
	parser.add_argument('--dist', type=int, default=0)
	parser.add_argument('--dist_debug', type=int, default=0)
	parser.add_argument('--print_res', type=int, default=0)
	args = parser.parse_args()
	
	plan_path = os.path.join(args.plan_dir, 'plan'+str(args.sub_query)+'.json')
	with open(plan_path, "r") as plan_file:
		plan = json.load(plan_file)
	
	update_num_workers(plan, args.sub_query, args.max_numb, args.max_shflw, args.num_subq)
	update_num_partitions(plan, args.p0)
	update_res_num_batches(plan, args.sub_query, args.num_subq, args.plan_dir)

	with open(plan_path, "w") as plan_file:
		json.dump(plan, plan_file, indent=2)
	
	if args.sub_query == 0:
		prepare_env(args.plan_dir, args.dist)
	# generate commands
	num_workers = plan["num_stage_workers"]
	worker_type = plan["worker_type"]
	tune_path = os.path.join(args.plan_dir, "env.tune")
	command = f"time mpirun --tune {tune_path} "
	if not args.dist:
		command += "--host localhost:64 "

	master_node = "r6d:50" if args.dist else ""

	command += append_cmd_workers(args.dist, master_node, args.sub_query, 
															 	num_workers[0], 'cpu', '--shfl')
	if args.dist:
		command += arrange_dist_workers(num_workers[1:len(num_workers)], worker_type[1:len(num_workers)])
	else:
		for s in range(1, len(num_workers) - 1):
			command += append_cmd_workers(args.dist, "", args.sub_query, num_workers[s], worker_type[s])
	command += append_cmd_workers(args.dist, master_node, args.sub_query, 1, 'cpu',
															 '' if args.print_res else '--no_print_res')
	print(command)