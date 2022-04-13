# coding = utf-8

import pypuf.simulation, pypuf.io
import pypuf.attack
import pypuf.metrics

import os, sys, random

import numpy as np

'''
Description: Pseudorandom Linear Boolean Function
'''
def linear_boolean_function(n, weight):
	a = np.repeat(1,weight)
	b = np.repeat(0,n-weight)
	function_generation = np.random.permutation(np.concatenate((a,b)))

	return function_generation
'''
Description: Generation of CRPs
'''
def crps_generation(puf_lbf, challenges, N): 
	challenges_update = (challenges+1)/2
	responses = np.reshape(2*(np.inner(puf_lbf, challenges_update) % 2)-1, (N,1,1))
	crps = pypuf.io.ChallengeResponseSet(challenges, responses)
	return crps

'''
Description: Logistic Regression algorithm with Linear Boolean Function
'''
def lbf_lr(puf_lbf, k, crps_training, crps_test):
	seed_instance_train = int.from_bytes(os.urandom(4), "big")
	seed_instance_test = int.from_bytes(os.urandom(4), "big")

	attack = pypuf.attack.LRAttack2021(crps_training, seed=seed_instance_train, k=k, bs=1000, lr=.001, epochs=100)

	attack.fit()
	model = attack.model

	accuracy = pypuf.metrics.similarity_data(crps_test.responses, model.eval(crps_test.challenges))

	return accuracy
'''
Description: LMN algorithm with Linear Boolean Function
'''
def lbf_lmn(puf_lbf, crps):
	attack = pypuf.attack.LMNAttack(crps, deg=2)

	attack.fit()
	model = attack.model


	N_test = 1000
	challenges_test = pypuf.io.random_inputs(n=crps.challenges.shape[1], N=N_test, seed=seed_instance_test)

	crps_reference = crps_generation(puf_lbf, challenges_test, N_test)
	crps_model = pypuf.io.ChallengeResponseSet(challenges_test, model.eval(challenges_test))

	accuracy = pypuf.metrics.similarity_data(crps_reference.responses, crps_model.responses)

	return accuracy

'''
Description: Logistic regression on Linear Boolean function with classical structure
'''
def lbf_lr_cpuf(puf_lbf, N, challenges, k):
	
	crps = crps_generation(puf_lbf, challenges, N)
	crps_training, crps_test = crps[:int(N*0.9),:], crps[int(N*0.9):,:]

	accuracy = lbf_lr(puf_lbf, k, crps_training, crps_test)

	return accuracy

'''
Description: Logistic regression on Linear Boolean function with hybrid structure
'''
def lbf_lr_hpuf(puf_lbf, N, challenges, k):
	crps = crps_generation(puf_lbf, challenges, N)
	crps_training, crps_test = crps[:int(N*0.9),:], crps[int(N*0.9):,:]

	for i in range(crps_training.responses.size):
		crps_training.responses[i][0][0] = hybrid_flipping_0(crps_training.responses[i][0][0], 0.85)

	accuracy = lbf_lr(puf_lbf, k, crps_training, crps_test)
	return accuracy


def hybrid_flipping_0(value_original, noisiness_hdata):
	value_updated = value_original

	value_p = random.random()

	if value_p > noisiness_hdata:
		value_updated = -value_original
	else:
		pass

	return value_updated


# Template of usage
if __name__ == '__main__':
	n = 8
	N = int(16)
	weight = 5
	# Different k for LR attack test
	k_steps = 10 
	# Different permutation test
	p_steps = 3

	seed_instance = int.from_bytes(os.urandom(4), "big")
	challenges = pypuf.io.random_inputs(n=n, N=N, seed=seed_instance)

	accuracy_cpuf = np.zeros((k_steps, p_steps))
	accuracy_hpuf = np.zeros((k_steps, p_steps))
	indices = np.zeros((p_steps, weight))
	k = np.arange(1,k_steps+1,1)
	

	# for i in range(p_steps):
	# 	puf_lbf = linear_boolean_function(n,weight)
	# 	indices[i,:] = np.nonzero(puf_lbf)[0]

	# 	for j in range(k_steps):
	# 		accuracy_cpuf[j,i] = lbf_lr_cpuf(puf_lbf, N, challenges, k[j])
	# 		accuracy_hpuf[j,i] = lbf_lr_hpuf(puf_lbf, N, challenges, k[j])
		
		
	
	# print("With %d samples in total we obtain:" %N)

	# for i in range(p_steps):
	# 	print("The Pseudorandom Linear Boolean Function represents a bitwise XOR of challenge with position:", indices[i,:])
	# 	for j in range(k_steps): 
	# 		print("k=%d, clbf_accuracy=%f, hlpf_accuracy=%f" %(k[j], accuracy_cpuf[j,i], accuracy_hpuf[j,i]))
		



