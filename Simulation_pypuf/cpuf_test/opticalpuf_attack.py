# coding = utf-8

import pypuf.simulation, pypuf.io
import pypuf.attack
import pypuf.metrics

import numpy as np 
import matplotlib.pyplot as plt
import math
import sys, os, random

from util import *

def select_odd_responses(responses, num_crps):
    odd_responses = np.empty((num_crps,13,1), float)
    for k in range(num_crps):
        for i in range(responses.shape[1]):
            if(i%2 == 1):
                odd_responses[k][int(i/2)] = responses[k][i].copy()
    return odd_responses

def instance_one_opuf_crps(puf, num_crps):
	seed_instance = int.from_bytes(os.urandom(4), "big")
	crps = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=num_crps, seed=seed_instance)
	# threshold = lambda r: np.sign(r - np.quantile(r.flatten(), .5))

	# crps.responses = threshold(crps.responses)
	return crps

def instance_one_opuf_attack(puf, num_crps):
	seed_instance = int.from_bytes(os.urandom(4), "big")
	crps = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=num_crps, seed=seed_instance)
	threshold = lambda r: np.sign(r - np.quantile(r.flatten(), .5))

	crps.responses = threshold(crps.responses)
	crps.responses = select_odd_responses(crps.responses, num_crps)

	seed_instance_train = int.from_bytes(os.urandom(4), "big")

	feature_map = pypuf.attack.LeastSquaresRegression.feature_map_optical_pufs_reloaded_improved
	attack = pypuf.attack.LeastSquaresRegression(crps, feature_map=feature_map)
	model = attack.fit()

	seed_instance_test = int.from_bytes(os.urandom(4), "big")
	crps_test = pypuf.io.ChallengeResponseSet.from_simulation(puf_opt, N=1000, seed=seed_instance_test)
	crps_test.responses = select_odd_responses(crps_test.responses, 1000)
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

def instance_one_hybrid_opuf_bit_attack(puf, num_crps):
	# bias_basis = puf_bias(puf_basis)
	# bias_bit = puf_bias(puf_bit)
	seed_instance = int.from_bytes(os.urandom(4), "big")
	crps = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=num_crps, seed=seed_instance)

	threshold = lambda r: np.sign(r - np.quantile(r.flatten(), .5))
	crps.responses = threshold(crps.responses)
	# print(crps.responses[0][0])

	res_cpy = np.copy(crps.responses)
	seed_instance_train = int.from_bytes(os.urandom(4), "big")

	crps.responses = select_odd_responses(crps.responses, num_crps)
	p_guess = 0.5*(1+np.sqrt(0.5))
	for k in range(num_crps):
		for i in range(crps.responses.shape[1]):
			crps.responses[k][i][0] = hybrid_flipping(crps.responses[k][i][0], p_guess) #Flip the response with probability 1 - P(guessing basis)	
	

	feature_map = pypuf.attack.LeastSquaresRegression.feature_map_optical_pufs_reloaded_improved
	attack = pypuf.attack.LeastSquaresRegression(crps, feature_map=feature_map)
	model = attack.fit()

	seed_instance_test = int.from_bytes(os.urandom(4), "big")
	crps_test = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=1000, seed=seed_instance_test)
	crps_test.responses = select_odd_responses(crps_test.responses, 1000)
	# accuracy = pypuf.metrics.correlation(model, crps_test).mean()

	# With post-processing 
	accuracy = pypuf.metrics.correlation(model, crps_test, postprocessing=threshold).mean()		

	return accuracy

def instance_one_hybrid_opuf_attack_n(puf, crps,repeat_experiment, attack = "both", steps=10):
	accuracy_hpuf = np.array([])
	for i in range(steps):
		print(i)
		instance_accuracy_hpuf_repeat = np.zeros(repeat_experiment)
		N = int(crps[i])
		for j in range(repeat_experiment):
			if(attack == "both"):
				instance_accuracy_hpuf_repeat[j] = instance_one_hybrid_opuf_attack(puf, N)
			elif(attack == "bit"):
				instance_accuracy_hpuf_repeat[j] = instance_one_hybrid_opuf_bit_attack(puf, N)

		instance_accuracy_hpuf = np.mean(instance_accuracy_hpuf_repeat)	
	
		accuracy_hpuf = np.append(accuracy_hpuf, instance_accuracy_hpuf)

	return accuracy_hpuf

def crp_opuf(n, steps=10):
	crps = np.array([])
	N = 1e2
	step = 0
	if n == 32:
		step = 1e4
	elif n == 64:
		step = 20e3
	elif n == 128:
		step = 5e2

	for i in range(steps):
		crps = np.append(crps, N)
		N = N + step

	return crps


if __name__ == '__main__':
	n_size = 32
	m_size = 1
	seed_instance = int.from_bytes(os.urandom(4), "big")
	puf_opt = pypuf.simulation.IntegratedOpticalPUF(n=n_size,m=m_size, seed=seed_instance)


	num_crps = 10000

	crps = instance_one_opuf_crps(puf_opt, num_crps)


	''' 
	Description: It shows that the distribution intensity of electronmagnetic
	field is not uniform, which leads to a diffculty of MUB encoding with higher dimension.
	'''
	val = 0. # the data to appear on the y-axis(start point).
	ar = crps.responses.flatten()
	# print(ar)
	plt.plot(ar, np.zeros_like(ar) + val, 'x')
	plt.show()



'''
template of usage
'''
'''
if __name__ == '__main__':
	n_size = 32
	m_size = 26
	seed_instance = int.from_bytes(os.urandom(4), "big")
	puf_opt = pypuf.simulation.IntegratedOpticalPUF(n=n_size,m=m_size, seed=seed_instance)

	repeat_experiment = 1

	num_crps = crp_opuf(n_size, steps=10)


	# accuracy_c = instance_one_opuf_attack(puf_opt, N_sample)
	# accuracy_h = instance_one_hybrid_opuf_attack(puf_opt, N_sample)

	accuracy_c = instance_one_opuf_attack_n(puf_opt, num_crps, repeat_experiment)
	print(accuracy_c)
	#accuracy_h1 = instance_one_hybrid_opuf_attack_n(puf_opt, num_crps, repeat_experiment, "both")
	#print(accuracy_h1)
	accuracy_h2 = instance_one_hybrid_opuf_attack_n(puf_opt, num_crps, repeat_experiment, "bit")
	print(accuracy_h2)

	np.save('./data/crps_opuf_'+str(n_size)+'_'+str(m_size)+'.npy', num_crps)
	np.save('./data/classical_opuf_accuracy'+str(n_size)+'_'+str(m_size)+'.npy', accuracy_c)
	#np.save('./data/hybrid_opuf_accuracy'+str(n)+'_'+str(m)+'.npy', accuracy_h2)
	np.save('./data/hybrid_opuf_odd_accuracy'+str(n_size)+'_'+str(m_size)+'.npy', accuracy_h2)

	plt.title('Optical PUF with Classical/Hybrid Construction')
	plt.plot(num_crps, accuracy_c, label='cpuf')
	plt.plot(num_crps, accuracy_h2, label='hpuf_odd')
	plt.xlabel('Number of CRPs')
	plt.ylabel('Accuracy (x100%)')
	plt.legend()
	plt.show()
'''
