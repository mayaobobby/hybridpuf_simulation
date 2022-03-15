# coding = utf-8
import random
import numpy as np

def hybrid_flipping(value_original, coe_hpuf):
	value_updated = value_original

	value_p = random.random()

	if value_p >= coe_hpuf:
		value_updated = -value_original
	else:
		pass

	return value_updated

'''
# of CRPs. Users can adapt their needs
'''
def crp_apuf(n, steps=10):
	crps = np.array([])
	N = 1000
	step = 0
	if n == 32:
		step = 10e3
	elif n == 64:
		step = 65e3
	elif n == 128:
		step = 500e3

	for i in range(steps):
		crps = np.append(crps, N)
		N = N + step

	return crps

