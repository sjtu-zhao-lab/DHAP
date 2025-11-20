import numpy as np
from scipy.optimize import minimize

aggr = [
	False, False, False, False, False, 
	False, False, False, False, False, 
	# False, False, False,
	# True, True, True, 
	# True,
	# True, True, True, 
	True,
	# True, True, 
	# True, True, 
	# True, True, True, True
]
# million rows/s
cpu_bw = np.array([
	5.2, 3.7, 3.1, 6.9, 7.5,
	4.4, 3.7, 4.5, 6.8, 7.9,
	# 5.9, 6.2, 11,
	# 39, 14, 4.7, 
	# 9.4,
	# 8.4, 3.5, 5.5,
	9.6,
	# 30, 27, 
	# 12, 16, 
	# 15, 40, 77, 77
])		
gpu_bw = np.array([
	110, 105, 83, 180, 190,
	160, 105, 83, 140, 130,
	# 240, 130, 220,
	# 0.06, 0.19, 1.1, 
	# 5.9,
	# 1.6, 6.9, 46, 
	170,
	# 0.4, 2, 
	# 0.6, 17, 
	# 12, 260, 300, 300
])
sel = [
	[0.173], [0.214], [0.389], [0.173, 0.214], [0.173, 0.214, 0.389],
	[0.173], [0.214], [0.289], [0.173, 0.214], [0.173, 0.214, 0.289],
	# [0.183], [0.229], [0.183, 0.229],
	# [1], [0.39, 1], [0.22, 0.39, 1], 
	# [0.17, 0.22, 0.39, 1],
	# [0.39], [0.29, 0.39], [0.22, 0.29, 0.39], 
	[0.17, 0.22, 0.29, 0.39],
	# [0.57], [0.23, 0.57], 
	# [0.57], [0.05, 0.57], 
	# [0.18, 0.23, 0.57], [0.03, 0.05, 0.57], [0.006, 0.009, 0.57], [0.006, 0.009, 0.012]
]
col = [
	[6, 5], [5, 5], [5, 4], [6, 5, 5], [6, 5, 5, 4],
	[6, 6], [6, 5], [5, 5], [6, 6, 5], [6, 6, 5, 5],
	# [4, 4], [4, 4], [4, 4, 4],
	# [4, 4], [5, 4, 4], [5, 5, 4, 4], 
	# [6, 5, 5, 4, 4],
	# [5, 5], [5, 5, 5], [6, 5, 5, 5], 
	[6, 6, 5, 5, 5],
	# [4, 4], [4, 4, 4], 
	# [4, 4], [4, 4, 4], 
	# [4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4]
]

def calc_reversed_bw(s, c, r1, r2, r3=None, aggr=False):
	assert len(c) == len(s) + 1
	res = c[0]/r1
	ac_sel = 1
	for j in range(len(s)-1):
		ac_sel *= s[j]
		res += ac_sel*c[j+1]/r1
	if aggr:
		res += ac_sel*s[-1]*c[-1]/r3
	else:
		res += ac_sel*s[-1]*c[-1]/r2
	return res

# Define the error function
def cpu_error(params):
	r1, r2, r3 = params
	predicted = np.zeros(len(cpu_bw))
	assert len(cpu_bw) == len(sel) == len(col)
	for i in range(len(cpu_bw)):
		if aggr[i]:
			predicted[i] = calc_reversed_bw(sel[i], col[i], r1, r2, r3, True)
		else:
			predicted[i] = calc_reversed_bw(sel[i], col[i], r1, r2)
	actual = 1 / cpu_bw
	return np.sum((predicted - actual)**2)

def gpu_error(params):
	r1, r2, r3 = params
	predicted = np.zeros(len(gpu_bw))
	assert len(gpu_bw) == len(sel) == len(col)
	for i in range(len(gpu_bw)):
		if aggr[i]:
			predicted[i] = calc_reversed_bw(sel[i], col[i], r1, r2, r3, True)
		else:
			predicted[i] = calc_reversed_bw(sel[i], col[i], r1, r2)
	actual = 1 / gpu_bw
	return np.sum((predicted - actual)**2)

initial_guess = [60, 6, 1000]
cpu_est = minimize(cpu_error, initial_guess, method='Nelder-Mead')
cpu_r1, cpu_r2, cpu_r3 = cpu_est.x
initial_guess = [1500, 150, 100]
gpu_est = minimize(gpu_error, initial_guess, method='Nelder-Mead')
gpu_r1, gpu_r2, gpu_r3 = gpu_est.x

if __name__ == '__main__':
	print(f"Estimated CPU r1: {cpu_r1}")
	print(f"Estimated CPU r2: {cpu_r2}")
	print(f"Estimated CPU r3: {cpu_r3}")
	print(f"Estimated GPU r1: {gpu_r1}")
	print(f"Estimated GPU r2: {gpu_r2}")
	print(f"Estimated GPU r3: {gpu_r3}")

	# Q31
	test_sel = [0.18, 0.23]
	test_c = [4, 4, 4]
	test_sel = [0.18]
	test_c = [4, 4]
	test_sel = [0.23]
	test_c = [4, 4]
	# Q42
	test_sel = [0.17, 0.22, 0.29]
	test_c = [6, 6, 5, 5]

	cpu_pred = calc_reversed_bw(test_sel, test_c, cpu_r1, cpu_r2)
	gpu_pred = calc_reversed_bw(test_sel, test_c, gpu_r1, gpu_r2)
	print(f"Predict CPU BW: {1/cpu_pred}")
	print(f"Predict GPU BW: {1/gpu_pred}")

	# Q42
	test_sel = [0.39]
	test_c = [5, 5]

	cpu_pred = calc_reversed_bw(test_sel, test_c, cpu_r1, cpu_r2, cpu_r3, True)
	gpu_pred = calc_reversed_bw(test_sel, test_c, gpu_r1, gpu_r2, gpu_r3, True)
	print(f"Predict CPU BW (aggr): {1/cpu_pred}")
	print(f"Predict GPU BW (aggr): {1/gpu_pred}")