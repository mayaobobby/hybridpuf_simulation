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
Description: Instantiate an arbiter PUF chain
'''
def instance_one_apuf(size_challenge, level_noisiness):
	seed_instance = int.from_bytes(os.urandom(4), "big")
	puf = pypuf.simulation.ArbiterPUF(n=64, noisiness=level_noisiness, seed=seed_instance)
	
	return puf

'''
Description: Arbiter PUF under ML attacks with classical structure. (one chain)
'''
def instance_one_apuf_attack(puf, num_crps):
	seed_instance = int.from_bytes(os.urandom(4), "big")
	crps = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=num_crps, seed=seed_instance)

	attack = pypuf.attack.LeastSquaresRegression(crps, feature_map=lambda cs: pypuf.simulation.ArbiterPUF.transform_atf(cs, k=1)[:, 0, :])
	model = attack.fit()


	model.postprocessing = model.postprocessing_threshold
	accuracy = pypuf.metrics.similarity(puf, model, seed=4)

	return accuracy

'''
Description: Arbiter PUF under ML attacks with classical structure. (two chains)
'''
def instance_two_apuf_attack(puf_bit, puf_basis, num_crps):
	accuracy_1 = instance_one_apuf_attack(puf_bit, num_crps)
	accuracy_2 = instance_one_apuf_attack(puf_basis, num_crps)

	accuracy_two_apuf = min(accuracy_1, accuracy_2) 

	return accuracy_two_apuf

'''
Description: Arbiter PUF under ML attacks with hybrid structure. (one chain)
'''
def instance_one_hybrid_apuf_attack(puf, num_crps, position):
	seed_instance = int.from_bytes(os.urandom(4), "big")
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


	attack = pypuf.attack.LeastSquaresRegression(crps, feature_map=lambda cs: pypuf.simulation.ArbiterPUF.transform_atf(cs, k=1)[:, 0, :])
	model = attack.fit()


	model.postprocessing = model.postprocessing_threshold
	accuracy = pypuf.metrics.similarity(puf, model, seed=4)

	return accuracy

'''
Description: Arbiter PUF under ML attacks with hybrid structure. (two chains for encoding one qubit, puf_bit encodes bit value,  puf_basis encodes basis value)
'''
def instance_two_hybrid_apuf_attack(puf_bit, puf_basis, num_crps):

	seed_instance_bit = int.from_bytes(os.urandom(4), "big")
	seed_instance_basis = int.from_bytes(os.urandom(4), "big")

	crps_bit = pypuf.io.ChallengeResponseSet.from_simulation(puf_bit, N=num_crps, seed=seed_instance_bit)
	crps_basis = pypuf.io.ChallengeResponseSet.from_simulation(puf_basis, N=num_crps, seed=seed_instance_basis)

	for i in range(crps_basis.responses.size):
		crps_basis_tmp = hybrid_flipping(crps_basis.responses[i][0][0], 2)
		if crps_basis.responses[i][0][0] == crps_basis_tmp:
			pass
		else:
			crps_basis.responses[i][0][0] = crps_basis_tmp
			crps_bit.responses[i][0][0] = hybrid_flipping(crps_bit.responses[i][0][0], 2)

	attack_bit = pypuf.attack.LeastSquaresRegression(crps_bit, feature_map=lambda cs: pypuf.simulation.ArbiterPUF.transform_atf(cs, k=1)[:, 0, :])
	model_bit = attack_bit.fit()

	attack_basis = pypuf.attack.LeastSquaresRegression(crps_basis, feature_map=lambda cs: pypuf.simulation.ArbiterPUF.transform_atf(cs, k=1)[:, 0, :])
	model_basis = attack_basis.fit()

	model_bit.postprocessing = model_bit.postprocessing_threshold
	accuracy_bit = pypuf.metrics.similarity(puf_bit, model_bit, seed=4)

	# The random seed guarantees that attacker cannot learn basis chain by bit chain (independency)
	model_basis.postprocessing = model_basis.postprocessing_threshold
	accuracy_basis = pypuf.metrics.similarity(puf_basis, model_basis, seed=4)

	accuracy_two_apuf = min(accuracy_bit, accuracy_basis) 

	return accuracy_two_apuf

'''
Description: Modeling Result of ML attacks comparison (one chain)
'''
def comparison_crps_accuracy_single_apuf_plotting(size_challenge, level_noisiness, position, repeat_experiment):

	puf = instance_one_apuf(size_challenge, level_noisiness)
	
	keyword = str(size_challenge)

	crps = np.array([])
	accuracy_cpuf = np.array([])
	accuracy_hpuf = np.array([])

	for i in range(10):

		instance_crps = int(1e2 + i*100e3)
		instance_accuracy_cpuf_repeat = np.zeros(repeat_experiment)
		instance_accuracy_hpuf_repeat = np.zeros(repeat_experiment)

		for j in range(repeat_experiment):

			instance_accuracy_cpuf_repeat[j] = instance_one_apuf_attack(puf, instance_crps)
			instance_accuracy_hpuf_repeat[j] = instance_one_hybrid_apuf_attack(puf, instance_crps, position)

		instance_accuracy_cpuf = np.mean(instance_accuracy_cpuf_repeat)
		instance_accuracy_hpuf = np.mean(instance_accuracy_hpuf_repeat)
		crps = np.append(crps, instance_crps)
		accuracy_cpuf = np.append(accuracy_cpuf, instance_accuracy_cpuf)
		accuracy_hpuf = np.append(accuracy_hpuf, instance_accuracy_hpuf)


	np.save('./data/crps_single_apuf_'+keyword+'.npy', crps)
	np.save('./data/accuracy_cpuf_single_apuf_'+keyword+'.npy', accuracy_cpuf)
	np.save('./data/accuracy_hpuf_single_apuf_'+keyword+'.npy', accuracy_hpuf)

	a = np.load('./data/crps_single_apuf_'+keyword+'.npy')
	b = np.load('./data/accuracy_cpuf_single_apuf_'+keyword+'.npy')
	c = np.load('./data/accuracy_hpuf_single_apuf_'+keyword+'.npy')

	plt.plot(a, b, label = 'cpuf')
	plt.plot(a, c, label = 'hpuf')
	# plt.title("Modeling Attack on PUF")
	plt.xlabel("Number of CRPs")
	plt.ylabel("Accuracy (x100%)")
	plt.legend()
	plt.show()

'''
Description: Modeling Result of ML attacks comparison (two chains for encoding one qubit)
'''
def comparison_crps_accuracy_two_apuf_plotting(size_challenge, level_noisiness, repeat_experiment):

	puf_bit = instance_one_apuf(size_challenge, level_noisiness)
	puf_basis = instance_one_apuf(size_challenge, level_noisiness)
	
	keyword = str(size_challenge)

	crps = np.array([])
	accuracy_cpuf = np.array([])
	accuracy_hpuf = np.array([])

	for i in range(10):
		instance_crps = int(1e2 + i*10e3)
		instance_accuracy_cpuf_repeat = np.zeros(repeat_experiment)
		instance_accuracy_hpuf_repeat = np.zeros(repeat_experiment)
		for j in range(repeat_experiment):
			instance_accuracy_cpuf_repeat[j] = instance_two_apuf_attack(puf_bit, puf_basis, instance_crps)
			instance_accuracy_hpuf_repeat[j] = instance_two_hybrid_apuf_attack(puf_bit, puf_basis, instance_crps)

		instance_accuracy_cpuf = np.mean(instance_accuracy_cpuf_repeat)
		instance_accuracy_hpuf = np.mean(instance_accuracy_hpuf_repeat)


		crps = np.append(crps, instance_crps)
		accuracy_cpuf = np.append(accuracy_cpuf, instance_accuracy_cpuf)
		accuracy_hpuf = np.append(accuracy_hpuf, instance_accuracy_hpuf)


	np.save('./data/crps_two_apuf_'+keyword+'.npy', crps)
	np.save('./data/accuracy_cpuf_two_apuf_'+keyword+'.npy', accuracy_cpuf)
	np.save('./data/accuracy_hpuf_two_apuf_'+keyword+'.npy', accuracy_hpuf)


	a = np.load('./data/crps_two_apuf_'+keyword+'.npy')
	b = np.load('./data/accuracy_cpuf_two_apuf_'+keyword+'.npy')
	c = np.load('./data/accuracy_hpuf_two_apuf_'+keyword+'.npy')

	plt.plot(a, b, label = 'cpuf')
	plt.plot(a, c, label = 'hpuf')
	# plt.title("Modeling Attack on PUF")
	plt.xlabel("Number of CRPs")
	plt.ylabel("Accuracy (x100%)")
	plt.legend()
	plt.show()



