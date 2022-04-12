# coding = utf-8

import pypuf.simulation, pypuf.io
import pypuf.attack
import pypuf.metrics

import numpy as np 
import matplotlib.pyplot as plt
import math
import sys, os, random

import scipy.stats

# Default for radnom challenges
def random_inputs(n, N, seed):
    return 2 * np.random.RandomState(seed).randint(0, 2, (N, n)) - 1

def random_challenges_crps(puf, n, N, challenges):
	# challenges = pypuf.io.random_inputs(n, N, seed_challenges)
	responses = puf.r_eval(1,challenges)
	crps = pypuf.io.ChallengeResponseSet(challenges, responses)

	return crps

def non_uniform_challenges_crps(puf, n, N, seed, quantile):
	challenges =  np.random.random_sample((N,n))
	challenges_postprocessing = np.sign(challenges - quantile)

	responses = puf.r_eval(1,challenges_postprocessing)

	crps = pypuf.io.ChallengeResponseSet(challenges_postprocessing, responses)
	return crps

def vec_bin_array(arr, m):
    to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(m))
    strs = to_str_func(arr)
    ret = np.zeros(list(arr.shape) + [m], dtype=np.int8)
    for bit_ix in range(0, m):
        fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == '1')
        ret[...,bit_ix] = fetch_bit_func(strs).astype("int8")

    return ret 

def normal_challenges_crps(puf, n, N, seed, mu, sigma):

	challenges = scipy.stats.norm.ppf(np.random.random(N), loc=mu, scale=sigma).astype(np.uintc)
	challenges = np.reshape(challenges, (N, 1)) 
	challenges_postprocessing = 2*vec_bin_array(challenges, n)-1
	responses = puf.r_eval(1,challenges_postprocessing)

	crps = pypuf.io.ChallengeResponseSet(challenges_postprocessing, responses)
	return crps

# Template of usage
if __name__ == '__main__':
	seed_puf = int.from_bytes(os.urandom(4), "big")
	seed_challenges = int.from_bytes(os.urandom(4), "big")
	challenges = pypuf.io.random_inputs(n, N, seed_challenges)
	challenges_postprocessing = (1 - challenges) // 2
	# print(challenges_postprocessing)
	puf = pypuf.simulation.XORArbiterPUF(n=32, noisiness=0, seed=seed_puf, k=5)


	crps = random_challenges_crps(puf, 32, 10, challenges)
	print('challenges:', crps.challenges)
	print('responses:', crps.responses)


'''
if __name__ == '__main__':

	n = 8
	noisiness = 0
	bias_challenge = 0.7
	N = int(20e3)

	seed_instance = int.from_bytes(os.urandom(4), "big")
	apuf = pypuf.simulation.ArbiterPUF(n=n, noisiness=noisiness, seed=seed_instance)
	
	# crps = non_uniform_challenges_crps(apuf, n, N, seed_instance, bias_challenge)
	# print(crps.challenges)

	mu = 3000
	sigma = 25
	crps = normal_challenges_crps(apuf, n, N, seed_instance, mu, sigma)
'''