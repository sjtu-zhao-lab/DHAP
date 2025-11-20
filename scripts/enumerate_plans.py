import argparse, sys
from regular import MACHINES, MAX_SLOTS
from itertools import combinations, product

NUM_WOKRERS = {t[0]: len(MACHINES[t]) * MAX_SLOTS[t] for t in MAX_SLOTS}

def get_all_nums(stage_type):
	all_range = [range(1, NUM_WOKRERS[st]+1) for st in stage_type]
	all_nums = list(product(*all_range))
	feasible = []
	for num in all_nums:
		used_num = {'c': 0, 'g': 0}
		for s in range(len(stage_type)):
			st = stage_type[s]
			used_num[st] += num[s]
		full = [used_num[t] == NUM_WOKRERS[t] for t in used_num]
		not_use = [used_num[t] == 0 for t in used_num]
		if (full[0] and full[1]) or (full[0] and not_use[1]) or (full[1] and not_use[0]):
			feasible.append(num)
	return feasible
	
def build_plan(shfl, stage_type, num):
	plan = ''
	for p in shfl:
		plan += str(p)
	plan += '/'
	for (t, n) in zip(stage_type, num):
		plan += t+str(n)+'-'
	return plan[:-1]

def get_all_plans(num_joins: int, num_subq: int):
	all_plans = []
	for num_shfl in range(0, num_joins):
		num_stages = num_shfl + 1
		all_shfl_points = [i for i in range(1, num_joins)]
		all_shfl_combs = list(combinations(all_shfl_points, num_shfl))
		stage_type_combs = list(product(['c', 'g'], repeat=num_stages))
		for shfl in all_shfl_combs:
			for stage_type in stage_type_combs:
				all_nums = get_all_nums(stage_type)
				for num in all_nums:
					plan = build_plan(shfl, stage_type, num)
					if num_subq > 1:
						assert num_subq == 2
						all_plans.append(plan+f"./c{NUM_WOKRERS['c']}")
						all_plans.append(plan+f"./g{NUM_WOKRERS['g']}")
					else:
						all_plans.append(plan)
	return all_plans

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--qid', required=True)
	parser.add_argument('--downsample', type=int, default=None, help="Sample 1 every x plans")
	parser.add_argument('--restore', default=None, help="Restore from a indicated plan")
	args = parser.parse_args()
	qid = args.qid
	num_joins = 0
	num_subq = 1
	if qid[0] == '2' or qid[0] == '3':
		num_joins = 3
	elif qid == '43':
		num_joins = 3
		num_subq = 2
	elif qid[0] == '4':
		num_joins = 4
	else:
		assert False
	all_plans = get_all_plans(num_joins, num_subq)
	downsample = args.downsample
	if downsample is not None:
		all_plans_d = []
		for p in range(len(all_plans)):
			if p % downsample == 0:
				all_plans_d.append(all_plans[p])
		all_plans = all_plans_d

	i = 0
	if args.restore is not None:
		for i in range(len(all_plans)+1):
			if i == len(all_plans) or all_plans[i] == args.restore:
				i += 1
				break
		assert i > 0 and i < len(all_plans)
	print(all_plans[i:])
	print(len(all_plans[i:]), file=sys.stderr)