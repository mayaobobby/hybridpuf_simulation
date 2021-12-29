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
Description: Instantiate a xorPUF chain
'''
def instance_one_xorpuf(size_challenge, num_apuf_xor, level_noisiness):
	# Number of parallel arbiter puf for each xorpuf
	seed_instance = int.from_bytes(os.urandom(4), "big")

	xorpuf = pypuf.simulation.XORArbiterPUF(n=size_challenge, k=num_apuf_xor, seed=seed_instance, noisiness=level_noisiness)
	return xorpuf	

'''
Description: Xor PUF under ML attacks with classical structure. (one chain)
'''
def instance_one_xorpuf_attack(puf, num_crps, num_bs, num_epochs):
	seed_instance = int.from_bytes(os.urandom(4), "big")
	# The number of CRPs(param N) optimised for attacks should adapt to param k (See performance in doc)
	crps = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=num_crps, seed=seed_instance)

	# Logistic Regression Attack on XORPUF
	# bs: Number of training examples that are processed together. Larger block size benefits from higher confidence of gradient direction and better computational performance, smaller block size benefits from earlier feedback of the weight adoption on following training steps.
	# lr: Learning rate of the Adam optimizer used for optimization.
	# epoch: Maximum number of epochs performed (i.e. the number of passes of the entire training dataset that the ml algorithm has complete).
	attack = pypuf.attack.LRAttack2021(crps, seed=3, k=puf.k, bs=num_bs, lr=.001, epochs=num_epochs)
	attack.fit()

	model = attack.model

	accuracy = pypuf.metrics.similarity(puf, model, seed=4)

	return accuracy

'''
Description: Xor PUF under ML attacks with classical structure. (two chains)
'''
def instance_two_xorpuf_attack(puf_bit, puf_basis, num_crps, num_bs, num_epochs):
	accuracy_1 = instance_one_xorpuf_attack(puf_bit, num_crps, num_bs, num_epochs)
	accuracy_2 = instance_one_xorpuf_attack(puf_basis, num_crps, num_bs, num_epochs)

	accuracy_two_apuf = min(accuracy_1, accuracy_2) 

	return accuracy_two_apuf

'''
Description: Xor PUF under ML attacks with hybrid structure. (one chain)
'''
def instance_one_hybrid_xorpuf_attack(puf, num_crps , num_bs, num_epochs, position):
	seed_instance = int.from_bytes(os.urandom(4), "big")
	# The number of CRPs(param N) optimised for attacks should adapt to param k (See performance in doc)
	crps = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=num_crps, seed=seed_instance)

	if position == 'bit':
		'''
		bit value 
		'''
		for i in range(crps.responses.size):
			# print(crps.responses[i][0][0])
			value = random.randint(1, 2)
			if value == 1:
				crps.responses[i][0][0] = hybrid_flipping(crps.responses[i][0][0], 2)
			else:
				pass

	elif position == 'basis':
		'''
		basis value
		'''	
		for i in range(crps.responses.size):
			# print(crps.responses[i][0][0])
			crps.responses[i][0][0] = hybrid_flipping(crps.responses[i][0][0], 2)
			# print(crps.responses[i][0][0])

	# Logistic Regression Attack on XORPUF
	# bs: Number of training examples that are processed together. Larger block size benefits from higher confidence of gradient direction and better computational performance, smaller block size benefits from earlier feedback of the weight adoption on following training steps.
	# lr: Learning rate of the Adam optimizer used for optimization.
	# epoch: Maximum number of epochs performed (i.e. the number of passes of the entire training dataset that the ml algorithm has complete).
	attack = pypuf.attack.LRAttack2021(crps, seed=3, k=puf.k, bs=num_bs, lr=.001, epochs=num_epochs)
	attack.fit()
	model = attack.model
	accuracy = pypuf.metrics.similarity(puf, model, seed=4)
	return accuracy

'''
Description: Xor PUF under ML attacks with hybrid structure. (two chains for encoding one qubit, puf_1 encodes bit value,  puf_2 encodes basis value)
'''
def instance_two_hybrid_xorpuf_attack(puf_1, puf_2, num_crps, num_bs, num_epochs):
	seed_instance_bit = int.from_bytes(os.urandom(4), "big")
	seed_instance_basis = int.from_bytes(os.urandom(4), "big")

	crps_bit = pypuf.io.ChallengeResponseSet.from_simulation(puf_1, N=num_crps, seed=seed_instance_bit)
	crps_basis = pypuf.io.ChallengeResponseSet.from_simulation(puf_2, N=num_crps, seed=seed_instance_basis)

	for i in range(crps_basis.responses.size):
		crps_basis_tmp = hybrid_flipping(crps_basis.responses[i][0][0], 2)
		if crps_basis.responses[i][0][0] == crps_basis_tmp:
			pass
		else:
			crps_basis.responses[i][0][0] = crps_basis_tmp
			crps_bit.responses[i][0][0] = hybrid_flipping(crps_bit.responses[i][0][0], 2)


	attack_bit = pypuf.attack.LRAttack2021(crps_bit, seed=3, k=puf_1.k, bs=num_bs, lr=.001, epochs=num_epochs)
	attack_bit.fit()
	model_bit = attack_bit.model

	attack_basis = pypuf.attack.LRAttack2021(crps_basis, seed=3, k=puf_2.k, bs=num_bs, lr=.001, epochs=num_epochs)
	attack_basis.fit()
	model_basis = attack_basis.model

	accuracy_bit = pypuf.metrics.similarity(puf_1, model_bit, seed=4)
	accuracy_basis = pypuf.metrics.similarity(puf_2, model_basis, seed=4)

	accuracy_two_xorpuf = min(accuracy_bit, accuracy_basis)

	return accuracy_two_xorpuf

'''
Description: Modeling Result of ML attacks comparison (one chain)

In our experiment we choose the #CRPs as follows: (num_apuf_xor = 4)
CPUF/HPUF 
32 bits challenge: 1000 - 30000 : int(1e3 + i*3e3)
64 bits challenge: 20000 - 40000 : int(8e3 + i*20e3)
128 bits challnge: 50000 - 300000 : int(1e4 + i*50e3)
'''
def comparison_crps_accuracy_single_xorpuf_plotting(size_challenge, num_apuf_xor, level_noisiness, num_bs, num_epochs, position, repeat_experiment):

	puf = instance_one_xorpuf(size_challenge, num_apuf_xor, level_noisiness)

	crps = np.array([])
	accuracy_cpuf = np.array([])
	accuracy_hpuf = np.array([])
	keyword = str(size_challenge)
	k = str(num_apuf_xor)

	for i in range(10):

		instance_crps = int(1e3 + i*3e3)
		instance_accuracy_cpuf_repeat = np.zeros(repeat_experiment)
		instance_accuracy_hpuf_repeat = np.zeros(repeat_experiment)

		for j in range(repeat_experiment):
			instance_accuracy_cpuf_repeat[j] = instance_one_xorpuf_attack(puf, instance_crps, num_bs, num_epochs)
			instance_accuracy_hpuf_repeat[j] = instance_one_hybrid_xorpuf_attack(puf, instance_crps, num_bs, num_epochs, position)


		instance_accuracy_cpuf = np.mean(instance_accuracy_cpuf_repeat)
		instance_accuracy_hpuf = np.mean(instance_accuracy_hpuf_repeat)


		crps = np.append(crps, instance_crps)
		accuracy_cpuf = np.append(accuracy_cpuf, instance_accuracy_cpuf)
		accuracy_hpuf = np.append(accuracy_hpuf, instance_accuracy_hpuf)

	np.save('./data/crps_single_xorpuf_'+keyword+'.npy', crps)
	np.save('./data/accuracy_cpuf_single_xorpuf_'+keyword+'.npy', accuracy_cpuf)
	np.save('./data/accuracy_hpuf_single_xorpuf_'+keyword+'.npy', accuracy_hpuf)


	a = np.load('./data/crps_single_xorpuf_'+keyword+'.npy')
	b = np.load('./data/accuracy_cpuf_single_xorpuf_'+keyword+'.npy')
	c = np.load('./data/accuracy_hpuf_single_xorpuf_'+keyword+'.npy')

	plt.plot(a, b, label = 'cpuf')
	plt.plot(a, c, label = 'hpuf')
	# plt.title("Modeling Attack on PUF")
	plt.xlabel("Number of CRPs")
	plt.ylabel("Accuracy (x100%)")
	plt.legend()
	plt.show()

'''
Description: Modeling Result of ML attacks comparison (two chains for encoding one qubit)

In our experiment we choose the #CRPs as follows: (num_apuf_xor = 4)
CPUF/HPUF 
32 bits challenge: 1000 - 30000 : int(1e3 + i*3e3)
64 bits challenge: 20000 - 40000 : int(8e3 + i*20e3)
128 bits challnge: 50000 - 300000 : int(1e4 + i*50e3)

'''
def comparison_crps_accuracy_two_xorpuf_plotting(size_challenge, num_apuf_xor, level_noisiness, num_bs, num_epochs, repeat_experiment):


	puf_bit = instance_one_xorpuf(size_challenge, num_apuf_xor, level_noisiness)
	puf_basis = instance_one_xorpuf(size_challenge, num_apuf_xor, level_noisiness)

	crps = np.array([])
	accuracy_cpuf = np.array([])
	accuracy_hpuf = np.array([])
	keyword = str(size_challenge)
	k = str(num_apuf_xor)

	for i in range(10):

		instance_crps = int(1e3 + i*3e3)
		instance_accuracy_cpuf_repeat = np.zeros(repeat_experiment)
		instance_accuracy_hpuf_repeat = np.zeros(repeat_experiment)

		for j in range(repeat_experiment):
			instance_accuracy_cpuf_repeat[j] = instance_two_xorpuf_attack(puf_bit, puf_basis, instance_crps, num_bs, num_epochs)
			instance_accuracy_hpuf_repeat[j] = instance_two_hybrid_xorpuf_attack(puf_bit, puf_basis, instance_crps, num_bs, num_epochs)


		instance_accuracy_cpuf = np.mean(instance_accuracy_cpuf_repeat)
		instance_accuracy_hpuf = np.mean(instance_accuracy_hpuf_repeat)


		crps = np.append(crps, instance_crps)
		accuracy_cpuf = np.append(accuracy_cpuf, instance_accuracy_cpuf)
		accuracy_hpuf = np.append(accuracy_hpuf, instance_accuracy_hpuf)

	np.save('./data/crps_two_xorpuf_'+keyword+'.npy', crps)
	np.save('./data/accuracy_cpuf_two_xorpuf_'+keyword+'.npy', accuracy_cpuf)
	np.save('./data/accuracy_hpuf_two_xorpuf_'+keyword+'.npy', accuracy_hpuf)


	a = np.load('./data/crps_two_xorpuf_'+keyword+'.npy')
	b = np.load('./data/accuracy_cpuf_two_xorpuf_'+keyword+'.npy')
	c = np.load('./data/accuracy_hpuf_two_xorpuf_'+keyword+'.npy')

	plt.plot(a, b, label = 'cpuf')
	plt.plot(a, c, label = 'hpuf')
	# plt.title("Modeling Attack on PUF")
	plt.xlabel("Number of CRPs")
	plt.ylabel("Accuracy (x100%)")
	plt.legend()
	plt.show()
