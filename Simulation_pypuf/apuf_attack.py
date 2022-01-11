# coding = utf-8

import pypuf.simulation, pypuf.io
import pypuf.attack
import pypuf.metrics

import numpy as np 
import matplotlib.pyplot as plt
import math
import sys, os

from util import *


'''
Description: Return the bias of a puf
'''	
def puf_bias(puf):
	seed_bias = int.from_bytes(os.urandom(4), "big")
	bias_value =  pypuf.metrics.bias(puf, seed_bias, N=100000)
	bias_prob = 1/2 - bias_value/2

	return bias_prob


'''
Description: Instantiate an arbiter PUF chain
'''
def instance_one_apuf(size_challenge, level_noisiness, weight_bias, bias):
	seed_instance = int.from_bytes(os.urandom(4), "big")
	puf = pypuf.simulation.ArbiterPUF(n=size_challenge, noisiness=level_noisiness, seed=seed_instance)
	
	while puf_bias(puf) < bias-0.005:
		weight_bias = weight_bias - 0.1
		puf.weight_array[0][size_challenge] = np.array([[weight_bias]])

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
Description: Arbiter PUF under ML attacks with classical structure. (one chain)
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
	accuracy = pypuf.metrics.similarity(puf, model, seed=seed_instance_test, N=round(num_crps*0.1))


	return accuracy

'''
Description: Arbiter PUF under ML attacks with different number of CRPs. (one chain)
'''
def instance_one_apuf_attack_n(puf, crps, repeat_experiment):
	accuracy_cpuf = np.array([])
	for i in range(10):
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
# def instance_two_apuf_attack(puf_bit, puf_basis, num_crps, num_bs, num_epochs):
# 	accuracy_1 = instance_one_apuf_attack(puf_bit, num_crps, num_bs, num_epochs)
# 	accuracy_2 = instance_one_apuf_attack(puf_basis, num_crps, num_bs, num_epochs)

# 	accuracy_two_apuf = min(accuracy_1, accuracy_2) 

# 	return accuracy_two_apuf

'''
Description: Arbiter PUF under ML attacks with hybrid structure. (one chain)
'''
def instance_one_hybrid_apuf_attack(puf, num_crps, position, num_bs, num_epochs):
	bias = puf_bias(puf)
	seed_instance = int.from_bytes(os.urandom(4), "big")
	crps_basis = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=num_crps, seed=seed_instance)
	crps_bit = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=num_crps, seed=seed_instance)
	res_basis_cpy = np.copy(crps_basis.responses)
	res_bit_cpy = np.copy(crps_bit.responses)
	seed_instance_train = int.from_bytes(os.urandom(4), "big")
	
	if position == 'bit':
		'''
		bit value 
		'''
		for i in range(crps_basis.responses.size):
			# print(crps.responses[i][0][0])
			'''
			Here is how an adversary obatin the training data from hpuf on state value (Different between #1 and #2)
			'''
			crps_basis.responses[i][0][0] = hybrid_flipping(crps_basis.responses[i][0][0], bias)
			# crps_basis.responses[i][0][0] = -1.0
			if crps_basis.responses[i][0][0] != res_basis_cpy[i][0][0]:
				crps_bit.responses[i][0][0] = hybrid_flipping(crps_bit.responses[i][0][0], 0.5)

		# print('Similarity basis:', pypuf.metrics.similarity_data(crps_basis.responses, res_basis_cpy))		
		# print('Similarity bit:', pypuf.metrics.similarity_data(crps_bit.responses, res_bit_cpy))	
		attack = pypuf.attack.LRAttack2021(crps_bit, seed=seed_instance_train, k=puf.k, bs=num_bs, lr=.001, epochs=num_epochs)		

	elif position == 'basis':
		'''
		basis value
		'''	
		for i in range(crps_basis.responses.size):
			# print(crps.responses[i][0][0])
			'''
			Here is how an adversary obatin the training data from hpuf on basis value (Different between #1 and #2)
			'''
			crps_basis.responses[i][0][0] = hybrid_flipping(crps_basis.responses[i][0][0], bias)
			# crps_basis.responses[i][0][0] = -1.0

		print('Similarity basis:', pypuf.metrics.similarity_data(crps_basis.responses, res_basis_cpy))	
		attack = pypuf.attack.LRAttack2021(crps_basis, seed=seed_instance_train, k=puf.k, bs=num_bs, lr=.001, epochs=num_epochs)

	# print('Similarity:',pypuf.metrics.similarity_data(crps.responses, res_cpy))
	attack.fit()

	model = attack.model

	seed_instance_test = int.from_bytes(os.urandom(4), "big")
	accuracy = pypuf.metrics.similarity(puf, model, seed=seed_instance_test)

	return accuracy

'''
Description: Arbiter PUF under ML attacks with hybrid structure with different number of CRPs. (one chain)
'''
def instance_one_hybrid_apuf_attack_n(puf, crps, position, repeat_experiment):
	accuracy_hpuf = np.array([])
	for i in range(10):
		instance_accuracy_hpuf_repeat = np.zeros(repeat_experiment)
		N = int(crps[i])
		for j in range(repeat_experiment):
			instance_accuracy_hpuf_repeat[j] = instance_one_hybrid_apuf_attack(puf, N, position, num_bs=1000, num_epochs=100)

		instance_accuracy_hpuf = np.mean(instance_accuracy_hpuf_repeat)	
	
		accuracy_hpuf = np.append(accuracy_hpuf, instance_accuracy_hpuf)

	return accuracy_hpuf



'''
Description: Arbiter PUF under ML attacks with hybrid structure. (two chains for encoding one qubit, puf_bit encodes bit value,  puf_basis encodes basis value)
'''
# def instance_two_hybrid_apuf_attack(puf_bit, puf_basis, bias_bit, bias_basis, num_crps, num_bs, num_epochs):

# 	# bias_bit = puf_bias(puf_bit)
# 	# bias_basis = puf_bias(puf_basis)

# 	seed_instance_bit = int.from_bytes(os.urandom(4), "big")
# 	seed_instance_basis = int.from_bytes(os.urandom(4), "big")

# 	crps_bit = pypuf.io.ChallengeResponseSet.from_simulation(puf_bit, N=num_crps, seed=seed_instance_bit)
# 	crps_basis = pypuf.io.ChallengeResponseSet.from_simulation(puf_basis, N=num_crps, seed=seed_instance_basis)

# 	for i in range(crps_basis.responses.size):
# 		crps_basis_tmp = hybrid_flipping(crps_basis.responses[i][0][0], bias_basis)
# 		if crps_basis.responses[i][0][0] == crps_basis_tmp:
# 			pass
# 		else:
# 			crps_basis.responses[i][0][0] = crps_basis_tmp
# 			crps_bit.responses[i][0][0] = hybrid_flipping(crps_bit.responses[i][0][0], bias_bit)

# 	seed_instance_train = int.from_bytes(os.urandom(4), "big")

# 	attack_bit = pypuf.attack.LRAttack2021(crps_bit, seed=seed_instance_train, k=puf_bit.k, bs=num_bs, lr=.001, epochs=num_epochs)

# 	model_bit = attack_bit.fit()

# 	attack_basis = pypuf.attack.LRAttack2021(crps_basis, seed=seed_instance_train, k=puf_basis.k, bs=num_bs, lr=.001, epochs=num_epochs)
# 	model_basis = attack_basis.fit()

# 	seed_instance_test = int.from_bytes(os.urandom(4), "big")

# 	accuracy_bit = pypuf.metrics.similarity(puf_bit, model_bit, seed=seed_instance_test)

# 	# The random seed guarantees that attacker cannot learn basis chain by bit chain (independency)
# 	accuracy_basis = pypuf.metrics.similarity(puf_basis, model_basis, seed=seed_instance_test)

# 	accuracy_two_apuf = min(accuracy_bit, accuracy_basis) 

# 	return accuracy_two_apuf

'''
Description: Modeling Result of ML attacks comparison (one chain)

In our experiment we choose the #CRPs as follows: 
CPUF/HPUF 
32 bits challenge: 1000 - 30000 : int(1e3 + i*3e3)
64 bits challenge: 20000 - 40000 : int(1e3 + i*20e3)
128 bits challnge: 50000 - 300000 : int(1e3 + i*50e3)

'''
# def comparison_crps_accuracy_single_apuf_plotting(size_challenge, level_noisiness, param_bias, num_bs, num_epochs, position, repeat_experiment):

	
# 	puf = instance_one_apuf(size_challenge, level_noisiness, param_bias)
# 	bias = puf_bias(puf)
# 	print('Bias = ', bias)
	
# 	keyword = str(size_challenge)

# 	crps = np.array([])
# 	accuracy_cpuf = np.array([])
# 	accuracy_hpuf = np.array([])

# 	for i in range(10):

# 		instance_crps = int(1e3 + i*50e3)
# 		instance_accuracy_cpuf_repeat = np.zeros(repeat_experiment)
# 		instance_accuracy_hpuf_repeat = np.zeros(repeat_experiment)

# 		for j in range(repeat_experiment):

# 			instance_accuracy_cpuf_repeat[j] = instance_one_apuf_attack(puf, instance_crps, num_bs, num_epochs)
# 			instance_accuracy_hpuf_repeat[j] = instance_one_hybrid_apuf_attack(puf, bias, instance_crps, position, num_bs, num_epochs)

# 		instance_accuracy_cpuf = np.mean(instance_accuracy_cpuf_repeat)
# 		instance_accuracy_hpuf = np.mean(instance_accuracy_hpuf_repeat)
# 		crps = np.append(crps, instance_crps)
# 		accuracy_cpuf = np.append(accuracy_cpuf, instance_accuracy_cpuf)
# 		accuracy_hpuf = np.append(accuracy_hpuf, instance_accuracy_hpuf)

# 	bias_str = str(bias)

# 	np.save('./data/crps_single_apuf_'+keyword+'_'+bias_str+'_'+position+'_'+'constantbasis'+'.npy', crps)
# 	np.save('./data/accuracy_cpuf_single_apuf_'+keyword+'_'+bias_str+'_'+position+'_'+'constantbasis'+'.npy', accuracy_cpuf)
# 	np.save('./data/accuracy_hpuf_single_apuf_'+keyword+'_'+bias_str+'_'+position+'_'+'constantbasis'+'.npy', accuracy_hpuf)


# 	a = np.load('./data/crps_single_apuf_'+keyword+'_'+bias_str+'_'+position+'_'+'constantbasis'+'.npy')
# 	b = np.load('./data/accuracy_cpuf_single_apuf_'+keyword+'_'+bias_str+'_'+position+'_'+'constantbasis'+'.npy')
# 	c = np.load('./data/accuracy_hpuf_single_apuf_'+keyword+'_'+bias_str+'_'+position+'_'+'constantbasis'+'.npy')



# 	plt.plot(a, b, label = 'cpuf')
# 	plt.plot(a, c, label = 'hpuf')
# 	plt.xlabel("Number of CRPs")
# 	plt.ylabel("Accuracy (x100%)")
# 	plt.legend()
# 	plt.show()

'''
Description: Modeling Result of ML attacks comparison (two chains for encoding one qubit):

In our experiment we choose the #CRPs as follows: 
CPUF/HPUF 
32 bits challenge: 1000 - 30000 : int(1e3 + i*3e3)
64 bits challenge: 20000 - 40000 : int(8e3 + i*20e3)
128 bits challnge: 50000 - 300000 : int(1e4 + i*50e3)
'''
# def comparison_crps_accuracy_two_apuf_plotting(size_challenge, level_noisiness, param_bias, num_bs, num_epochs, repeat_experiment):

# 	puf_bit = instance_one_apuf(size_challenge, level_noisiness, param_bias)
# 	puf_basis = instance_one_apuf(size_challenge, level_noisiness, param_bias)

# 	bias_bit = puf_bias(puf_bit)
# 	bias_basis = puf_bias(puf_basis)

# 	bias = np.average(np.array([bias_bit, bias_basis]))
# 	print('Bias = ', bias)
	
# 	keyword = str(size_challenge)

# 	crps = np.array([])
# 	accuracy_cpuf = np.array([])
# 	accuracy_hpuf = np.array([])

# 	for i in range(10):
# 		instance_crps = int(1e3 + i*3e3)
# 		instance_accuracy_cpuf_repeat = np.zeros(repeat_experiment)
# 		instance_accuracy_hpuf_repeat = np.zeros(repeat_experiment)
# 		for j in range(repeat_experiment):
# 			instance_accuracy_cpuf_repeat[j] = instance_two_apuf_attack(puf_bit, puf_basis, instance_crps, num_bs, num_epochs)
# 			instance_accuracy_hpuf_repeat[j] = instance_two_hybrid_apuf_attack(puf_bit, puf_basis, bias_bit, bias_basis, instance_crps, num_bs, num_epochs)

# 		instance_accuracy_cpuf = np.mean(instance_accuracy_cpuf_repeat)
# 		instance_accuracy_hpuf = np.mean(instance_accuracy_hpuf_repeat)


# 		crps = np.append(crps, instance_crps)
# 		accuracy_cpuf = np.append(accuracy_cpuf, instance_accuracy_cpuf)
# 		accuracy_hpuf = np.append(accuracy_hpuf, instance_accuracy_hpuf)


# 	bias_str = str(bias)

# 	np.save('./data/crps_two_apuf_'+keyword+'_'+bias_str+'.npy', crps)
# 	np.save('./data/accuracy_cpuf_two_apuf_'+keyword+'_'+bias_str+'.npy', accuracy_cpuf)
# 	np.save('./data/accuracy_hpuf_two_apuf_'+keyword+'_'+bias_str+'.npy', accuracy_hpuf)


# 	a = np.load('./data/crps_two_apuf_'+keyword+'_'+bias_str+'.npy')
# 	b = np.load('./data/accuracy_cpuf_two_apuf_'+keyword+'_'+bias_str+'.npy')
# 	c = np.load('./data/accuracy_hpuf_two_apuf_'+keyword+'_'+bias_str+'.npy')

# 	plt.plot(a, b, label = 'cpuf')
# 	plt.plot(a, c, label = 'hpuf')
# 	plt.xlabel("Number of CRPs")
# 	plt.ylabel("Accuracy (x100%)")
# 	plt.legend()
# 	plt.show()



