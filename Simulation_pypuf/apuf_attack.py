# coding = utf-8

import pypuf.simulation, pypuf.io
import pypuf.attack
import pypuf.metrics

import numpy as np 
import matplotlib.pyplot as plt
import math
import sys, os, random

from util import *

'''
Description: Return the uniqness of pufs
'''
def uniqueness(n, noisiness, weight_bias, bias, k):
	seed_uniqness = int.from_bytes(os.urandom(4), "big")
	instances = [instance_one_puf(n, noisiness, weight_bias, bias, k=k) for i in range(5)]
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
def instance_one_puf(size_challenge, level_noisiness, weight_bias, bias, k=1):
	seed_instance = int.from_bytes(os.urandom(4), "big")
	if k == 1:
		puf = pypuf.simulation.ArbiterPUF(n=size_challenge, noisiness=level_noisiness, seed=seed_instance)
	else:
		puf = pypuf.simulation.XORArbiterPUF(n=size_challenge, noisiness=level_noisiness, seed=seed_instance, k=k)
	
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
	# Logistic Regression Attack on XORPUF
	# bs: Number of training examples that are processed together. Larger block size benefits from higher confidence of gradient direction and better computational performance, smaller block size benefits from earlier feedback of the weight adoption on following training steps.
	# lr: Learning rate of the Adam optimizer used for optimization.
	# epoch: Maximum number of epochs performed (i.e. the number of passes of the entire training dataset that the ml algorithm has complete).
	seed_instance_train = int.from_bytes(os.urandom(4), "big")
	attack = pypuf.attack.LRAttack2021(crps, seed=seed_instance_train, k=puf.k, bs=num_bs, lr=.001, epochs=num_epochs)
	attack.fit()

	model = attack.model

	seed_instance_test = int.from_bytes(os.urandom(4), "big")
	accuracy = pypuf.metrics.similarity(puf, model, seed=seed_instance_test)

	return accuracy


def instance_one_apuf_attack_n(puf, crps, repeat_experiment, steps=10):
	accuracy_cpuf = np.array([])
	for i in range(steps):
		instance_accuracy_cpuf_repeat = np.zeros(repeat_experiment)
		N = int(crps[i])
		for j in range(repeat_experiment):
			instance_accuracy_cpuf_repeat[j] = instance_one_apuf_attack(puf, N, num_bs=1000, num_epochs=100)

		instance_accuracy_cpuf = np.mean(instance_accuracy_cpuf_repeat)	
	
		accuracy_cpuf = np.append(accuracy_cpuf, instance_accuracy_cpuf)

	return accuracy_cpuf

'''
Description: Arbiter PUF under ML attacks with classical structure. (two chains)
'''
def instance_two_apuf_attack(puf_bit, puf_basis, crps, repeat_experiment):
	accuracy_basis = instance_one_apuf_attack_n(puf_basis, crps, repeat_experiment)
	accuracy_bit = instance_one_apuf_attack_n(puf_bit, crps, repeat_experiment)

	accuracy_two_apuf = accuracy_basis * accuracy_bit

	return accuracy_two_apuf

'''
Description: Arbiter PUF under ML attacks with hybrid structure (alternative). BAD!
'''
def instance_one_hybrid_apuf_attack_alternative(puf_1, puf_2, num_crps, num_bs, num_epochs):
	bias_puf_1 = puf_bias(puf_1)
	bias_puf_2 = puf_bias(puf_2)
	seed_puf_1 = int.from_bytes(os.urandom(4), "big")
	seed_puf_2 = int.from_bytes(os.urandom(4), "big")
	crps_puf_1 = pypuf.io.ChallengeResponseSet.from_simulation(puf_1, N=num_crps, seed=seed_puf_1)
	crps_puf_2 = pypuf.io.ChallengeResponseSet.from_simulation(puf_2, N=num_crps, seed=seed_puf_2)
	res_puf_1_cpy = np.copy(crps_puf_1.responses)
	res_puf_2_cpy = np.copy(crps_puf_2.responses)
	seed_instance_train_1 = int.from_bytes(os.urandom(4), "big")
	seed_instance_train_2 = int.from_bytes(os.urandom(4), "big")

	for i in range(num_crps):
		if random.randint(0,1) == 0:
			if random.randint(0,1) == 0:
				crps_puf_1.responses[i][0][0] = crps_puf_1.responses[i][0][0]
				crps_puf_2.responses[i][0][0] = -crps_puf_2.responses[i][0][0]
			else:
				crps_puf_1.responses[i][0][0] = -crps_puf_1.responses[i][0][0]
				crps_puf_2.responses[i][0][0] = crps_puf_2.responses[i][0][0]
		else:
			pass
	print('Similarity PUF 1:', pypuf.metrics.similarity_data(crps_puf_1.responses, res_puf_1_cpy))		
	print('Similarity PUF 2:', pypuf.metrics.similarity_data(crps_puf_2.responses, res_puf_2_cpy))		

	attack_puf_1 = pypuf.attack.LRAttack2021(crps_puf_1, seed=seed_instance_train_1, k=puf_1.k, bs=num_bs, lr=.001, epochs=num_epochs)
	attack_puf_1.fit()
	model_puf_1 = attack_puf_1.model
	seed_instance_test_1 = int.from_bytes(os.urandom(4), "big")
	accuracy_puf_1 = pypuf.metrics.similarity(puf_1, model_puf_1, seed=seed_instance_test_1)

	attack_puf_2 = pypuf.attack.LRAttack2021(crps_puf_2, seed=seed_instance_train_2, k=puf_2.k, bs=num_bs, lr=.001, epochs=num_epochs)
	attack_puf_2.fit()
	model_puf_2 = attack_puf_2.model
	seed_instance_test_2 = int.from_bytes(os.urandom(4), "big")
	accuracy_puf_2 = pypuf.metrics.similarity(puf_2, model_puf_2, seed=seed_instance_test_1)		


	return accuracy_puf_1*accuracy_puf_2


'''
Description: Arbiter PUF under ML attacks with hybrid structure. (one chain)
'''
def instance_one_hybrid_apuf_attack(puf_bit, puf_basis, num_crps, position, num_bs, num_epochs):
	bias_basis = puf_bias(puf_basis)
	# bias_bit = puf_bias(puf_bit)
	seed_basis = int.from_bytes(os.urandom(4), "big")
	seed_bit = int.from_bytes(os.urandom(4), "big")
	crps_basis = pypuf.io.ChallengeResponseSet.from_simulation(puf_basis, N=num_crps, seed=seed_basis)
	crps_bit = pypuf.io.ChallengeResponseSet.from_simulation(puf_bit, N=num_crps, seed=seed_bit)
	res_basis_cpy = np.copy(crps_basis.responses)
	res_bit_cpy = np.copy(crps_bit.responses)
	seed_instance_train = int.from_bytes(os.urandom(4), "big")
	
	if position == 'bit':
		'''
		bit value 
		'''
		for i in range(crps_basis.responses.size):
			'''
			Here is how an adversary obatin the training data from hpuf on state value (Different between #1 and #2)
			'''
			# print(crps_basis.responses[i][0][0])
			
			crps_basis.responses[i][0][0] = hybrid_flipping(crps_basis.responses[i][0][0], bias_basis)
			# crps_basis.responses[i][0][0] = -1.0
			if crps_basis.responses[i][0][0] != res_basis_cpy[i][0][0]:
			# if crps_basis.responses[i][0][0] != 2*random.randint(0,1)-1:
				crps_bit.responses[i][0][0] = hybrid_flipping(crps_bit.responses[i][0][0], 0.5)
			else:
				pass
			
		# print('Similarity basis:', pypuf.metrics.similarity_data(crps_basis.responses, res_basis_cpy))		
		# print('Similarity bit:', pypuf.metrics.similarity_data(crps_bit.responses, res_bit_cpy))	
		attack = pypuf.attack.LRAttack2021(crps_bit, seed=seed_instance_train, k=puf_bit.k, bs=num_bs, lr=.001, epochs=num_epochs)
		attack.fit()
		model = attack.model
		seed_instance_test = int.from_bytes(os.urandom(4), "big")
		accuracy = pypuf.metrics.similarity(puf_bit, model, seed=seed_instance_test)		

	elif position == 'basis':
		'''
		basis value
		'''	
		for i in range(crps_basis.responses.size):
			'''
			Here is how an adversary obatin the training data from hpuf on basis value (Different between #1 and #2)
			'''
			# print(crps_basis.responses[i][0][0])
			crps_basis.responses[i][0][0] = hybrid_flipping(crps_basis.responses[i][0][0], bias_basis)
			# crps_basis.responses[i][0][0] = -1.0

		# print('Similarity basis:', pypuf.metrics.similarity_data(crps_basis.responses, res_basis_cpy))	
		attack = pypuf.attack.LRAttack2021(crps_basis, seed=seed_instance_train, k=puf_basis.k, bs=num_bs, lr=.001, epochs=num_epochs)
		attack.fit()
		model = attack.model
		seed_instance_test = int.from_bytes(os.urandom(4), "big")	
		accuracy = pypuf.metrics.similarity(puf_basis, model, seed=seed_instance_test)


	# Alternative solution
	# N_test_set = 1000
	# test_set = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=N_test_set, seed=seed_instance_test)
	# # for j in range(N_test_set):
	# # 	test_set.responses[j][0][0] != 2*random.randint(0,1)-1
	# accuracy = pypuf.metrics.accuracy(model, test_set)

	return accuracy

def instance_one_hybrid_apuf_attack_n(puf_bit, puf_basis, crps, position, repeat_experiment, steps=10):
	accuracy_hpuf = np.array([])
	for i in range(steps):
		instance_accuracy_hpuf_repeat = np.zeros(repeat_experiment)
		N = int(crps[i])
		for j in range(repeat_experiment):
			instance_accuracy_hpuf_repeat[j] = instance_one_hybrid_apuf_attack(puf_bit, puf_basis, N, position, num_bs=1000, num_epochs=100)

		instance_accuracy_hpuf = np.mean(instance_accuracy_hpuf_repeat)	
	
		accuracy_hpuf = np.append(accuracy_hpuf, instance_accuracy_hpuf)

	return accuracy_hpuf

'''
Description: Arbiter PUF under ML attacks with hybrid structure. (two chains for encoding one qubit, puf_bit encodes bit value,  puf_basis encodes basis value)
'''
def instance_two_hybrid_apuf_attack(puf_bit, puf_basis, crps, repeat_experiment):

	accuracy_basis = instance_one_hybrid_apuf_attack_n(puf_bit, puf_basis, crps, 'basis', repeat_experiment)
	accuracy_bit = instance_one_hybrid_apuf_attack_n(puf_bit, puf_basis, crps, 'bit', repeat_experiment)

	accuracy_two_apuf = np.zeros(accuracy_basis.size)

	# Consider the Unambiguous State Discrimination v.0
	p_0 = puf_bias(puf_basis)
	p_c = 0
	for i in range(accuracy_bit.size):
		if 1/3 <= p_0 <= 2/3:
			p_c = ((1-math.sqrt(2*p_0*(1-p_0)))*accuracy_bit[i])
		elif p_0 > 2/3:
			p_c = ((1-(p_0/2 + (1-p_0)))*accuracy_bit[i])
		elif p_0 < 1/3:
			p_c = ((1-(p_0 + (1-p_0)/2))*accuracy_bit[i])
		else:
			pass
		accuracy_two_apuf[i] = p_c

	return accuracy_two_apuf

'''
Description: Modeling Result of ML attacks comparison (one chain)
'''
def comparison_crps_accuracy_single_apuf_plotting(n, noisiness, weight_bias, bias, position, repeat_experiment, k):

	
	puf_bit = instance_one_puf(n, noisiness, weight_bias, bias, k=k)
	puf_basis = instance_one_puf(n, noisiness, weight_bias, bias, k=k)
	
	

	crps = crp_apuf(n)
	accuracy_cpuf = instance_one_apuf_attack_n(puf_bit, crps, repeat_experiment)
	accuracy_hpuf = instance_one_hybrid_apuf_attack_n(puf_bit, puf_basis, crps, position, repeat_experiment)

	keyword = str(n)
	bias_str = str(bias)

	np.save('./data/crps_single_apuf_'+str(n)+'_'+str(bias)+'_'+'.npy', crps)
	np.save('./data/accuracy_cpuf_single_apuf_'+str(n)+'_'+str(bias)+'.npy', accuracy_cpuf)
	np.save('./data/accuracy_hpuf_single_apuf_'+str(n)+'_'+str(bias)+'_'+position+'.npy', accuracy_hpuf)


	a = np.load('./data/crps_single_apuf_'+str(n)+'_'+str(bias)+'_'+'.npy')
	b = np.load('./data/accuracy_cpuf_single_apuf_'+str(n)+'_'+str(bias)+'.npy')
	c = np.load('./data/accuracy_hpuf_single_apuf_'+str(n)+'_'+str(bias)+'_'+position+'.npy')



	plt.plot(a, b, label = 'cpuf')
	plt.plot(a, c, label = 'hpuf')
	plt.xlabel("Number of CRPs")
	plt.ylabel("Accuracy (x100%)")
	plt.legend()
	plt.show()

'''
Description: Modeling Result of ML attacks comparison (two chains for encoding one qubit):
'''
def comparison_crps_accuracy_two_apuf_plotting(n, noisiness, weight_bias, bias, repeat_experiment, k):

	puf_bit = instance_one_puf(n, noisiness, weight_bias, bias, k=k)
	puf_basis = instance_one_puf(n, noisiness, weight_bias, bias, k=k)

	
	keyword = str(n)

	crps = crp_apuf(n)
	accuracy_cpuf = instance_two_apuf_attack(puf_bit, puf_basis, crps, repeat_experiment)
	accuracy_hpuf = instance_two_hybrid_apuf_attack(puf_bit, puf_basis, crps, repeat_experiment)


	np.save('./data/crps_two_apuf_'+str(n)+'_'+str(bias)+'.npy', crps)
	np.save('./data/accuracy_cpuf_two_apuf_'+str(n)+'_'+str(bias)+'.npy', accuracy_cpuf)
	np.save('./data/accuracy_hpuf_two_apuf_'+str(n)+'_'+str(bias)+'.npy', accuracy_hpuf)


	a = np.load('./data/crps_two_apuf_'+str(n)+'_'+str(bias)+'.npy')
	b = np.load('./data/accuracy_cpuf_two_apuf_'+str(n)+'_'+str(bias)+'.npy')
	c = np.load('./data/accuracy_hpuf_two_apuf_'+str(n)+'_'+str(bias)+'.npy')

	plt.plot(a, b, label = 'cpuf')
	plt.plot(a, c, label = 'hpuf')
	plt.xlabel("Number of CRPs")
	plt.ylabel("Accuracy (x100%)")
	plt.legend()
	plt.show()



