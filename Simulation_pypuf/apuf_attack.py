# coding = utf-8

import pypuf.simulation, pypuf.io
import pypuf.attack
import pypuf.metrics

import numpy as np 
import matplotlib.pyplot as plt
import math
import sys, os, random
from pathlib import Path

from util import *

'''
Description: Return the uniqness of pufs
'''
def uniqueness(n, noisiness_cpuf, weight_bias, bias, k):
	seed_uniqness = int.from_bytes(os.urandom(4), "big")
	instances = [instance_one_puf(n, noisiness_cpuf, weight_bias, bias, k=k) for i in range(5)]
	uniqness = pypuf.metrics.uniqueness(instances, seed=seed_uniqness, N=1000)

	return uniqness

'''
Description: Return the bias of a puf
'''	
def puf_bias(puf):
	seed_bias = int.from_bytes(os.urandom(4), "big")
	bias_value =  pypuf.metrics.bias(puf, seed_bias, N=300000)
	bias_prob = round(1/2 - bias_value/2,2)

	return bias_prob


'''
Description: Instantiate an arbiter PUF chain
'''
def instance_one_puf(size_challenge, noisiness_cpuf, weight_bias, bias, k=1):
	seed_instance = int.from_bytes(os.urandom(4), "big")
	if k == 1:
		puf = pypuf.simulation.ArbiterPUF(n=size_challenge, noisiness=noisiness_cpuf, seed=seed_instance)
	else:
		puf = pypuf.simulation.XORArbiterPUF(n=size_challenge, noisiness=noisiness_cpuf, seed=seed_instance, k=k)
	
	while puf_bias(puf) < bias-0.005:
		for i in range(k):
			weight_bias = weight_bias - 0.05
			puf.weight_array[i][size_challenge] = np.array([[weight_bias]])

	# print(weight_bias)
	# print(puf_bias(puf))
	return puf

'''
Description: Return the reliability of a puf
'''
def puf_reliability(puf):
	seed_reliability = int.from_bytes(os.urandom(4), "big")
	reliability_instance = np.average(pypuf.metrics.reliability(puf, seed_reliability))
	
	return reliability_instance

'''
Description: Arbiter-based PUF under ML attacks with classical structure. (one chain)
'''
def instance_one_apuf_attack(puf, num_crps, num_bs, num_epochs):
	seed_instance = int.from_bytes(os.urandom(4), "big")
	crps = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=num_crps, seed=seed_instance)

	crps_test = int(num_crps*0.9)
	if crps_test > int(1e4):
		crps_test = int(1e4)
	seed_instance_train = int.from_bytes(os.urandom(4), "big")
	attack = pypuf.attack.LRAttack2021(crps, seed=seed_instance_train, k=puf.k, bs=num_bs, lr=.001, epochs=num_epochs)
	attack.fit()

	model = attack.model

	seed_instance_test = int.from_bytes(os.urandom(4), "big")
	accuracy = pypuf.metrics.similarity(puf, model, seed=seed_instance_test, N=crps_test)

	return accuracy


'''
Description: Arbiter-based PUF under ML attacks with classical structure. (one chain with repetitions)
'''
def instance_one_apuf_attack_n(puf, crps, repeat_experiment, steps):
	total_feas_rate = np.array([])
	total_accuracy = np.array([])
	for i in range(steps):
		instance_accuracy_cpuf_repeat = np.zeros(repeat_experiment)
		N = int(crps[i])
		for j in range(repeat_experiment):
			instance_accuracy_cpuf_repeat[j] = instance_one_apuf_attack(puf, N, num_bs=1000, num_epochs=100)
		instance_accuracy_cpuf = np.mean(instance_accuracy_cpuf_repeat)	

		feas_count = 0
		instance_feas_rate = 0
		for k in instance_accuracy_cpuf_repeat:
			if k >= .95:
				feas_count += 1
		instance_feas_rate = feas_count/instance_accuracy_cpuf_repeat.shape[0]
		total_feas_rate = np.append(total_feas_rate, instance_feas_rate)
		total_accuracy = np.append(total_accuracy, instance_accuracy_cpuf)
		with open('./data/xorpuf'+str(puf.k)+'/c_summary_xorpuf'+str(puf.k)+'_'+str(puf.n)+'.txt', 'a') as f:
			print("CRPs:", N, file=f)
			print(instance_accuracy_cpuf_repeat, file=f)
			print("CPUF Median:", np.median(instance_accuracy_cpuf_repeat), file=f)
			print("CPUF Mean:", instance_accuracy_cpuf, file=f)
			print("Success Rate:", instance_feas_rate, file=f)


	return total_accuracy


'''
Description: Arbiter PUF under ML attacks with classical structure. (two chains)
'''
def instance_two_apuf_attack(puf_bit, puf_basis, crps, repeat_experiment):
	accuracy_basis = instance_one_apuf_attack_n(puf_basis, crps, repeat_experiment)
	accuracy_bit = instance_one_apuf_attack_n(puf_bit, crps, repeat_experiment)

	accuracy_two_apuf = accuracy_basis * accuracy_bit

	return accuracy_two_apuf

'''
Description: Arbiter PUF under ML attacks with hybrid structure. (one chain)
'''
def instance_one_hybrid_apuf_attack(success_prob, puf_bit, puf_basis, num_crps, position, num_bs, num_epochs):
	bias_basis = puf_bias(puf_basis)
	# bias_bit = puf_bias(puf_bit)
	seed_basis = int.from_bytes(os.urandom(4), "big")
	seed_bit = int.from_bytes(os.urandom(4), "big")
	crps_basis = pypuf.io.ChallengeResponseSet.from_simulation(puf_basis, N=num_crps, seed=seed_basis)
	crps_bit = pypuf.io.ChallengeResponseSet.from_simulation(puf_bit, N=num_crps, seed=seed_bit)
	res_basis_cpy = np.copy(crps_basis.responses)
	res_bit_cpy = np.copy(crps_bit.responses)
	seed_instance_train = int.from_bytes(os.urandom(4), "big")

	crps_test = int(num_crps*0.9)
	if crps_test > int(1e4):
		crps_test = int(1e4)
	
	if position == 'bit':
		'''
		bit value 
		'''
		for i in range(crps_bit.responses.size):
			crps_bit.responses[i][0][0] = hybrid_flipping(crps_bit.responses[i][0][0], success_prob)
			
		# print('Similarity basis:', pypuf.metrics.similarity_data(crps_basis.responses, res_basis_cpy))		
		# print('Similarity bit:', pypuf.metrics.similarity_data(crps_bit.responses, res_bit_cpy))	
		attack = pypuf.attack.LRAttack2021(crps_bit, seed=seed_instance_train, k=puf_bit.k, bs=num_bs, lr=.001, epochs=num_epochs)
		attack.fit()
		model = attack.model
		seed_instance_test = int.from_bytes(os.urandom(4), "big")
		accuracy = pypuf.metrics.similarity(puf_bit, model, seed=seed_instance_test, N=crps_test)	

	elif position == 'basis':
		'''
		basis value
		'''	
		for i in range(crps_basis.responses.size):
			crps_basis.responses[i][0][0] = hybrid_flipping(crps_basis.responses[i][0][0], success_prob)

		attack = pypuf.attack.LRAttack2021(crps_basis, seed=seed_instance_train, k=puf_basis.k, bs=num_bs, lr=.001, epochs=num_epochs)
		attack.fit()
		model = attack.model
		seed_instance_test = int.from_bytes(os.urandom(4), "big")	
		accuracy = pypuf.metrics.similarity(puf_basis, model, seed=seed_instance_test, N=crps_test)

	return accuracy

'''
Description: Arbiter-based PUF under ML attacks with hybrid structure. (one chain with repetitions)
'''
def instance_one_hybrid_apuf_attack_n(success_prob, puf_bit, puf_basis, crps, position, repeat_experiment, steps):
	Path("./data/xorpuf"+str(puf_basis.k)).mkdir(parents=True, exist_ok=True)
	total_feas_rate = np.array([])
	total_accuracy = np.array([])
	for i in range(steps):
		instance_accuracy_hpuf_repeat = np.zeros(repeat_experiment)
		N = int(crps[i])
		for j in range(repeat_experiment):
			instance_accuracy_hpuf_repeat[j] = instance_one_hybrid_apuf_attack(success_prob, puf_bit, puf_basis, N, position, num_bs=1000, num_epochs=100)
		instance_accuracy_hpuf = np.mean(instance_accuracy_hpuf_repeat)	
		feas_count = 0
		instance_feas_rate = 0
		for k in instance_accuracy_hpuf_repeat:
			if k >= .95:
				feas_count += 1
		instance_feas_rate = feas_count/instance_accuracy_hpuf_repeat.shape[0]
		total_feas_rate = np.append(total_feas_rate, instance_feas_rate)
		total_accuracy = np.append(total_accuracy, instance_accuracy_hpuf)
		with open('./data/xorpuf'+str(puf_basis.k)+'/h_summary_xorpuf'+str(puf_basis.k)+'_'+str(puf_basis.n)+'.txt', 'a') as f:
			print("CRPs:", N, file=f)
			print(instance_accuracy_hpuf_repeat, file=f)
			print("HPUF Median:", np.median(instance_accuracy_hpuf_repeat), file=f)
			print("HPUF Mean:", instance_accuracy_hpuf, file=f)
			print("Success Rate:", instance_feas_rate, file=f)

	return total_accuracy



	



