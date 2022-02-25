# coding = utf-8

import pypuf.simulation, pypuf.io
import pypuf.attack
import pypuf.metrics

import os, sys, random

import numpy as np

from util import *

def linear_boolean_function(n, weight, position=0):
	function_generation = np.zeros(n)
	for i in range(weight):
		function_generation[i] = 1
	'''
	ones = np.random.choice(np.arange(0,n),replace = False, size = weight)
	for position in ones:
		function_generation[position] = 1
	'''

	return function_generation

def crps_generation(puf_linear, challenges, N): 
	challenges_update = (challenges+1)/2
	responses = np.reshape(2*(np.inner(puf_linear, challenges_update) % 2)-1, (N,1,1))
	crps = pypuf.io.ChallengeResponseSet(challenges, responses)
	return crps

def instance_one_lbf_attack(puf, n, N, k = 5, num_bs = 1000,num_epochs = 100):
	seed_instance = int.from_bytes(os.urandom(4), "big")
	seed_instance_train = int.from_bytes(os.urandom(4), "big")
	seed_instance_test = int.from_bytes(os.urandom(4), "big")
	
	challenges = pypuf.io.random_inputs(n=n, N=N, seed=seed_instance)
	crps = crps_generation(puf, challenges, N)

	attack = pypuf.attack.LRAttack2021(crps, seed=seed_instance_train, k = k, bs = num_bs, lr=.001, epochs=num_epochs)
	attack.fit()

	model = attack.model


	N_test = 1000
	challenges_test = pypuf.io.random_inputs(n=n, N=N_test, seed=seed_instance_test)

	crps_reference = crps_generation(puf, challenges_test, N_test)
	crps_model = pypuf.io.ChallengeResponseSet(challenges_test, model.eval(challenges_test))

	accuracy = pypuf.metrics.similarity_data(crps_reference.responses, crps_model.responses)

	return accuracy

def instance_one_hybrid_lbf_attack(puf, n, N, k = 4, num_bs = 1000,num_epochs = 100):
	seed_instance = int.from_bytes(os.urandom(4), "big")
	seed_instance_train = int.from_bytes(os.urandom(4), "big")
	seed_instance_test = int.from_bytes(os.urandom(4), "big")
	
	challenges = pypuf.io.random_inputs(n=n, N=N, seed=seed_instance)
	crps = crps_generation(puf, challenges, N)
	for i in range(N):
		crps.responses[i][0][0] = hybrid_flipping(crps.responses[i][0][0], 0.75)

	attack = pypuf.attack.LRAttack2021(crps, seed=seed_instance_train, k = k, bs = num_bs, lr=.001, epochs=num_epochs)
	attack.fit()

	model = attack.model


	N_test = 1000
	challenges_test = pypuf.io.random_inputs(n=n, N=N_test, seed=seed_instance_test)

	crps_reference = crps_generation(puf, challenges_test, N_test)
	crps_model = pypuf.io.ChallengeResponseSet(challenges_test, model.eval(challenges_test))

	accuracy = pypuf.metrics.similarity_data(crps_reference.responses, crps_model.responses)

	return accuracy

# Template of usage
if __name__ == '__main__':
	n = 32
	N = int(260e3)
	weight = 8
	puf_lbf = linear_boolean_function(n,weight)

	accuracy_c = instance_one_lbf_attack(puf_lbf, n, N)
	accuracy_h = instance_one_hybrid_lbf_attack(puf_lbf, n, N)
	print(accuracy_c)
	print(accuracy_h)
	
	


	

