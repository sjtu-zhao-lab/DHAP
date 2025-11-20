import argparse, os, json
from typing import Dict, List, Tuple
import numpy as np
from itertools import combinations
from bw_predict import cpu_r1, cpu_r2, cpu_r3, gpu_r1, gpu_r2, gpu_r3, calc_reversed_bw
from regular import MAX_SLOTS, set_scale
MACHINES = set_scale()
from enum import Enum
class DevT(Enum):
	udf = 0
	cpu = 1
	gpu = 2

class Planner:
	def __init__(self, plan_path: str) -> None:
		self.plan_path = plan_path
		# system configuration
		self.total_num_cpu = len(MACHINES["cpu"]) * MAX_SLOTS["cpu"]
		self.total_num_gpu = len(MACHINES["gpu"]) * MAX_SLOTS["gpu"]
		# ========= tmp fix for tpch 3&5 ===========
		subq_id = int(plan_path[-6])
		if os.getenv("USE_TPCH") and subq_id == 1:
			self.total_num_gpu = 0
		# ========= tmp fix for tpch 3&5 ===========

		# GB to MB
		self.L_CPU = float(os.environ["L_CPU"]*1024)
		self.L_GPU = float(os.environ["L_GPU"]*1024)
		# profiled relative performance
		self.prof_perf = {DevT.cpu: 1, DevT.gpu: 6}

		with open(plan_path, 'r') as plan_file:
			plan = json.load(plan_file)
			self.num_joins = len(plan["build_table_sizes_mb"])
			self.build_size = [0] + plan["build_table_sizes_mb"]
			first_probe = plan["first_probe"]
			self.res_ncols = [first_probe["cols"]] + plan["num_join_res_col"]
			self.res_size = [first_probe["rows"]] + plan["num_join_res_row"]
			self.sel = [0] + [self.res_size[i]/self.res_size[i-1] for i in range(1, self.num_joins+1)]
			print("Result #cols: ", self.res_ncols)
			print("Result #rows: ", self.res_size)
			print("Selectivity: ", self.sel)

		self.worker_type: Dict[Tuple[int], List[DevT]] = {}		# map from shfl_points
		self.worker_num: Dict[Tuple[int], List[int]] = {}
		self.all_stage_workloads: Dict[Tuple[int], List[int]] = {}
		self.candidate: List[Tuple[int]] = []
		# shfl_point, worker_type, worker_num, max_stage_time
		self.plan_max_stage_time: List[Tuple[Tuple[int], Tuple[DevT], Tuple[int], float]] = []

	def valid_shfl(self, shfl_points: Tuple[int]) -> bool:
		shfl_points_l = list(shfl_points) + [self.num_joins]
		last_shfl_p = 0
		cur_stage = 1
		for shfl_p in shfl_points_l:
			all_build_use = 0
			for s in range(last_shfl_p+2, shfl_p+1):
				all_build_use += self.build_size[s]
			if all_build_use > self.L_CPU and all_build_use > self.L_GPU:
				return False
			elif all_build_use < self.L_CPU and all_build_use > self.L_GPU:
				self.worker_type[shfl_points][cur_stage] = DevT.cpu
			elif all_build_use > self.L_CPU and all_build_use < self.L_GPU:
				self.worker_type[shfl_points][cur_stage] = DevT.gpu
			last_shfl_p = shfl_p
			cur_stage += 1

		return True
	
	def get_shfl_candidate(self):
		all_shfl_points = [i for i in range(1, self.num_joins)]
		# get feasible plan candidates, with type of some stagess determined
		for num_shfls in range(0, self.num_joins):
			all_shfl_combs = list(combinations(all_shfl_points, num_shfls))
			for shfl_points in all_shfl_combs:
				self.worker_type[shfl_points] = [DevT.udf] * (len(shfl_points)+2)
				self.worker_num[shfl_points] = [0] * (len(shfl_points)+2)
				valid = self.valid_shfl(shfl_points)		# to process the last stage
				# print(shfl_points, valid, self.worker_type[shfl_points])
				if (valid):
					self.candidate.append(shfl_points)
		print("Candidate shuffle: ", self.candidate)
	
	def get_stage_workload(self, shfl_points_l: List[int]) -> List[int]:
		stage_workloads = [0]
		for s in (shfl_points_l + [self.num_joins]):
			if s != 0:
				stage_workloads.append(self.res_size[s]*self.res_ncols[s])
		return stage_workloads

	def update_TN_(self):
		for shfl_points in self.candidate:
			shfl_points_l = [0] + list(shfl_points)
			num_stages = len(shfl_points_l)
			stage_bw = {DevT.cpu: np.zeros(num_stages), DevT.gpu: np.zeros(num_stages)}
			stage_inp = np.zeros(num_stages)
			for s in range(num_stages):
				shfl_st = shfl_points_l[s]
				shfl_ed = shfl_points_l[s+1] if s < num_stages-1 else shfl_st+1
				stage_sel = self.sel[shfl_st+1: shfl_ed+1]
				stage_col = self.res_ncols[shfl_st: shfl_ed+1]
				if s == num_stages-1:
					stage_bw[DevT.cpu][s] = 1/calc_reversed_bw(stage_sel, stage_col, cpu_r1, cpu_r2, cpu_r3, True)
					stage_bw[DevT.gpu][s] = 1/calc_reversed_bw(stage_sel, stage_col, gpu_r1, gpu_r2, gpu_r3, True)
				else:
					stage_bw[DevT.cpu][s] = 1/calc_reversed_bw(stage_sel, stage_col, cpu_r1, cpu_r2)
					stage_bw[DevT.gpu][s] = 1/calc_reversed_bw(stage_sel, stage_col, gpu_r1, gpu_r2)
				stage_inp[s] = self.res_size[shfl_st]
			stage_cpu_time = stage_inp/stage_bw[DevT.cpu]
			stage_gpu_time = stage_inp/stage_bw[DevT.gpu]
			for k in range(num_stages+1):
				stage_time = np.zeros(num_stages)
				heavy_k_stages = np.argsort(-stage_gpu_time)[:k]
				heavy_k_stages_gpu_time = stage_gpu_time[heavy_k_stages]
				heavy_k_stages_gpu_num = np.round(self.total_num_gpu * heavy_k_stages_gpu_time/np.sum(heavy_k_stages_gpu_time))
				other_stages = np.delete(np.arange(0, num_stages), heavy_k_stages)
				# print("stage indices: ", heavy_k_stages, other_stages)
				other_stages_cpu_time = np.delete(stage_cpu_time, heavy_k_stages)
				other_stages_cpu_num = np.round(self.total_num_cpu * other_stages_cpu_time/np.sum(other_stages_cpu_time))
				if 0 in heavy_k_stages_gpu_num or 0 in other_stages_cpu_num:	# illegal plan
					continue
				w_type = [DevT.cpu]*num_stages
				w_num = [0]*num_stages
				for ki, i in zip(heavy_k_stages, range(k)):
					w_type[ki] = DevT.gpu
					w_num[ki] = int(heavy_k_stages_gpu_num[i])
				for oi, i in zip(other_stages, range(num_stages-k)):
					w_type[oi] = DevT.cpu
					w_num[oi] = int(other_stages_cpu_num[i])
				# print("stage nums: ", heavy_k_stages_gpu_num, other_stages_cpu_num)
				amort_stage_gpu_time = heavy_k_stages_gpu_time / heavy_k_stages_gpu_num
				amort_stage_cpu_time = other_stages_cpu_time / other_stages_cpu_num
				# print("amort time: ", amort_stage_gpu_time, amort_stage_cpu_time)
				max_time = np.concatenate(( amort_stage_cpu_time, amort_stage_gpu_time )).max()
				self.plan_max_stage_time.append((shfl_points, tuple(w_type), tuple(w_num), max_time))
		print("#Candidate plans: ", len(self.plan_max_stage_time))

	def get_best_plan(self) -> Tuple[Tuple[int], List[DevT], List[int]]:
		est_time: Dict[List[int], float] = {}			# map from shfl_points
		for shfl_points in self.candidate:
			w_type = self.worker_type[shfl_points]
			w_num = self.worker_num[shfl_points]
			t = 0
			for s in range(1, len(w_type)):
				t += self.all_stage_workloads[shfl_points][s] / (w_num[s] * self.prof_perf[w_type[s]])
			est_time[shfl_points] = t
		
		best_shfl_points = min(est_time, key=est_time.get)
		return best_shfl_points, self.worker_type[best_shfl_points], self.worker_num[best_shfl_points]
	
	def get_best_plan_v2(self) -> Tuple[Tuple[int], List[DevT], List[int]]:
		best_plan = min(self.plan_max_stage_time, key=lambda x: x[3])
		return best_plan[0], best_plan[1], best_plan[2]

	def update_and_dump(self, shfl_points, worker_type, worker_num):
		with open(self.plan_path, 'r') as plan_file:
			plan = json.load(plan_file)
			plan["shfl_points"] = shfl_points
			plan["worker_type"] = worker_type
			plan["worker_num"] = worker_num
		with open(self.plan_path, 'w') as plan_file:
			json.dump(plan, plan_file, indent=2)

	def heuristic_plan(self):
		self.get_shfl_candidate()
		self.update_TN_()
		shfl_points, w_type, w_num = self.get_best_plan_v2()
		best_plan = ''
		for shfl_p in shfl_points:
			best_plan += str(shfl_p)
		best_plan += '/'
		for t, n in zip(w_type, w_num):
			best_plan += t.name[0]+str(n)+'-'
		print("Using heuristic plan: ", best_plan[:-1])
		self.update_and_dump([0]+list(shfl_points), ["udf"]+[t.name for t in w_type], [0]+list(w_num))

	def parse_manual_plan(self, plan: str):
		[shfl_p, tn] = plan.split('/')
		manual_sp = [0] + [int(p) for p in shfl_p]
		manual_type = ["udf"]
		manual_num = [0]
		for tn0 in tn.split('-'):
			manual_type.append("gpu" if tn0[0] == 'g' else "cpu")
			manual_num.append(int(tn0[1:]))
		assert len(manual_sp)+1 == len(manual_type)
		self.update_and_dump(manual_sp, manual_type, manual_num)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--plan_path', required=True)
	args = parser.parse_args()
	planner = Planner(args.plan_path)
	manual_plan = os.getenv("DHAP_PLAN")
	if (not manual_plan):
		planner.heuristic_plan()
	else:
		# plan0.json, only 1-digit subq_id 
		subq_id = int(args.plan_path[-6])
		manual_plan = manual_plan.split('.')[subq_id]
		print("Using manual plan: ", manual_plan)
		planner.parse_manual_plan(manual_plan)
