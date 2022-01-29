# coding = utf-8

import pypuf.simulation, pypuf.io
import pypuf.attack
import pypuf.metrics

import numpy as np 
import matplotlib.pyplot as plt
import math
import sys, os, random

from util import *

# def puf_reliability(puf):
# 	seed_reliability = int.from_bytes(os.urandom(4), "big")
# 	reliability_instance = np.average(pypuf.metrics.reliability(puf, seed_reliability))
	
# 	return reliability_instance


# def puf_uniqueness(n, m):
# 	seed_uniqueness = int.from_bytes(os.urandom(4), "big")
# 	instances_opt = [pypuf.simulation.IntegratedOpticalPUF(n=n_size,m=m_size, seed=i) for i in range(5)]
# 	uniqueness = pypuf.metrics.uniqueness(instances_opt, seed=seed_uniqueness, N=1000)

# 	return uniqueness


def instance_one_opuf_attack(puf, num_crps):
	seed_instance = int.from_bytes(os.urandom(4), "big")
	crps = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=num_crps, seed=seed_instance)
	threshold = lambda r: np.sign(r - np.quantile(r.flatten(), .5))

	crps.responses = threshold(crps.responses)


	seed_instance_train = int.from_bytes(os.urandom(4), "big")

	feature_map = pypuf.attack.LeastSquaresRegression.feature_map_optical_pufs_reloaded_improved
	attack = pypuf.attack.LeastSquaresRegression(crps, feature_map=feature_map)
	model = attack.fit()

	seed_instance_test = int.from_bytes(os.urandom(4), "big")
	crps_test = pypuf.io.ChallengeResponseSet.from_simulation(puf_opt, N=1000, seed=seed_instance_test)

	# accuracy = pypuf.metrics.correlation(model, crps_test).mean()

	# With post-processing 
	
	accuracy = pypuf.metrics.correlation(model, crps_test, postprocessing=threshold).mean()


	return accuracy


def instance_one_opuf_attack_n(puf, crps, repeat_experiment, steps=10):
	accuracy_opuf = np.array([])
	for i in range(steps):
		print(i)
		instance_accuracy_opuf_repeat = np.zeros(repeat_experiment)
		N = int(crps[i])
		for j in range(repeat_experiment):
			instance_accuracy_opuf_repeat[j] = instance_one_opuf_attack(puf, N)


		instance_accuracy_opuf = np.mean(instance_accuracy_opuf_repeat)	
	
		accuracy_opuf = np.append(accuracy_opuf, instance_accuracy_opuf)

	return accuracy_opuf

def instance_one_hybrid_opuf_attack(puf, num_crps):
	# bias_basis = puf_bias(puf_basis)
	# bias_bit = puf_bias(puf_bit)
	seed_instance = int.from_bytes(os.urandom(4), "big")
	crps = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=num_crps, seed=seed_instance)

	threshold = lambda r: np.sign(r - np.quantile(r.flatten(), .5))
	crps.responses = threshold(crps.responses)

	# print(crps.responses[0][0])

	res_cpy = np.copy(crps.responses)
	seed_instance_train = int.from_bytes(os.urandom(4), "big")
	
	for k in range(num_crps):
		for i in range(crps.responses.shape[1]):
			if i%2 == 0:
				crps.responses[k][i][0] = hybrid_flipping(crps.responses[k][i][0], 0.5)
			elif i%2 == 1 and crps.responses[k][i-1][0] != res_cpy[k][i-1][0]:
				crps.responses[k][i][0] = hybrid_flipping(crps.responses[k][i][0], 0.5)
			else:
				pass
				
	feature_map = pypuf.attack.LeastSquaresRegression.feature_map_optical_pufs_reloaded_improved
	attack = pypuf.attack.LeastSquaresRegression(crps, feature_map=feature_map)
	model = attack.fit()

	seed_instance_test = int.from_bytes(os.urandom(4), "big")
	crps_test = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=1000, seed=seed_instance_test)
	# accuracy = pypuf.metrics.correlation(model, crps_test).mean()

	# With post-processing 
	accuracy = pypuf.metrics.correlation(model, crps_test, postprocessing=threshold).mean()		

	return accuracy

def instance_one_hybrid_opuf_attack_n(puf, crps, repeat_experiment, steps=10):
	accuracy_hpuf = np.array([])
	for i in range(steps):
		print(i)
		instance_accuracy_hpuf_repeat = np.zeros(repeat_experiment)
		N = int(crps[i])
		for j in range(repeat_experiment):
			instance_accuracy_hpuf_repeat[j] = instance_one_hybrid_opuf_attack(puf, N)

		instance_accuracy_hpuf = np.mean(instance_accuracy_hpuf_repeat)	
	
		accuracy_hpuf = np.append(accuracy_hpuf, instance_accuracy_hpuf)

	return accuracy_hpuf


def crp_opuf(n, steps=10):
	crps = np.array([])
	N = 1e2
	step = 0
	if n == 32:
		step = 5e2
	elif n == 64:
		step = 20e3
	elif n == 128:
		step = 5e2

	for i in range(steps):
		crps = np.append(crps, N)
		N = N + step

	return crps

'''
template of usage
'''
if __name__ == '__main__':
	n_size = 32
	m_size = 26
	N_sample = int(40e4)
	seed_instance = int.from_bytes(os.urandom(4), "big")
	puf_opt = pypuf.simulation.IntegratedOpticalPUF(n=n_size,m=m_size, seed=seed_instance)

	# crps = pypuf.io.ChallengeResponseSet.from_simulation(puf_opt, N=N_sample, seed=2)
	# print(crps.responses.shape[1])

	repeat_experiment = 1

	crps = crp_opuf(n_size, steps=10)



	accuracy_c = instance_one_opuf_attack(puf_opt, N_sample)
	print(accuracy_c)
	accuracy_h = instance_one_hybrid_opuf_attack(puf_opt, N_sample)
	print(accuracy_h)
	# accuracy_c = instance_one_opuf_attack_n(puf_opt, crps, repeat_experiment, steps=10)
	# accuracy_h = instance_one_hybrid_opuf_attack_n(puf_opt, crps, repeat_experiment, steps=10)

	