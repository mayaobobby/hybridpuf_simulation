# coding = utf-8

import pypuf.simulation, pypuf.io
import pypuf.attack
import pypuf.metrics

import os, sys, random

import numpy as np

def linear_boolean_function(weight, position=0):
	a = np.repeat(1,weight)
	b = np.repeat(0,n-weight)
	function_generation = np.concatenate((a,b))

	return function_generation

def crps_generation(puf_linear, challenges, N): 
	challenges_update = (challenges+1)/2
	responses = np.reshape(2*(np.inner(puf_linear, challenges_update) % 2)-1, (N,1,1))
	crps = pypuf.io.ChallengeResponseSet(challenges, responses)
	return crps

def hybrid_flipping_0(value_original, bias):
	value_updated = value_original

	value_p = random.random()

	if value_p >= bias:
		value_updated = -value_original
	else:
		pass

	return value_updated


# Template of usage
'''
if __name__ == '__main__':
	n = 32
	N = int(260e3)
	weight = 4
	seed_instance = int.from_bytes(os.urandom(4), "big")
	seed_instance_test = int.from_bytes(os.urandom(4), "big")

	challenges = pypuf.io.random_inputs(n=n, N=N, seed=seed_instance)
	puf_lbf = linear_boolean_function(weight)
	crps = crps_generation(puf_lbf, challenges, N)

	for i in range(N):
		crps.responses[i][0][0] = hybrid_flipping_0(crps.responses[i][0][0], 0.75)



	attack = pypuf.attack.LMNAttack(crps, deg=2)
	attack.fit()

	model = attack.model


	N_test = 1000
	challenges_test = pypuf.io.random_inputs(n=n, N=N_test, seed=seed_instance_test)

	crps_reference = crps_generation(puf_lbf, challenges_test, N_test)
	crps_model = pypuf.io.ChallengeResponseSet(challenges_test, model.eval(challenges_test))

	accuracy = pypuf.metrics.similarity_data(crps_reference.responses, crps_model.responses)

	print(accuracy)
'''

