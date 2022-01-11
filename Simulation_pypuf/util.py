# coding = utf-8
import random
import numpy as np

# def hybrid_flipping(value_original, bias):
# 	value_updated = value_original

# 	value_p = random.random()

# 	if value_p >= max(bias, 1-bias):
# 		value_updated = -value_original
# 	else:
# 		pass

# 	return value_updated

def hybrid_flipping(value_original, bias):
	# value_updated = value_original

	value_p = random.random()

	if value_p >= bias:
		value_updated = 1.0
	else:
		value_updated = -1.0

	return value_updated	

def crp_apuf(n, steps=10):
	crps = np.array([])
	N = 1000
	step = 0
	if n == 32:
		step = 3e3
	elif n == 64:
		step = 20e3
	elif n == 128:
		step = 50e3

	for i in range(steps):
		crps = np.append(crps, N)
		N = N + step

	return crps
