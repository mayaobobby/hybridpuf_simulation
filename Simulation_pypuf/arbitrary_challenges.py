# coding = utf-8

import pypuf.simulation, pypuf.io
import pypuf.attack
import pypuf.metrics

import numpy as np 
import matplotlib.pyplot as plt
import math
import sys, os, random

import scipy.stats

'''
Description: CRPs generation with uniform random distributed challenges
'''
def random_challenges_crps(puf, n, N, challenges):
	# challenges = pypuf.io.random_inputs(n, N, seed_challenges)
	responses = puf.r_eval(1,challenges)
	crps = pypuf.io.ChallengeResponseSet(challenges, responses)

	return crps

'''
Description: CRPs generation with a biased challenges (# of -1/1 for each challenge)
'''
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

'''
Description: CRPs generation with normal distribution challenges
'''
def normal_challenges_crps(puf, n, N, seed, mu, sigma):

	challenges = scipy.stats.norm.ppf(np.random.random(N), loc=mu, scale=sigma).astype(np.uintc)
	challenges = np.reshape(challenges, (N, 1)) 
	challenges_postprocessing = 2*vec_bin_array(challenges, n)-1
	responses = puf.r_eval(1,challenges_postprocessing)

	crps = pypuf.io.ChallengeResponseSet(challenges_postprocessing, responses)
	return crps

# Template of usage
if __name__ == '__main__':

	n = 8
	noisiness = 0
	bias_challenge = 1
	N = int(20e3)

	seed_instance = int.from_bytes(os.urandom(4), "big")
	apuf = pypuf.simulation.ArbiterPUF(n=n, noisiness=noisiness, seed=seed_instance)
	
	# crps = non_uniform_challenges_crps(apuf, n, N, seed_instance, bias_challenge)
	# print(crps.challenges)

	# mu = 3000
	# sigma = 25
	# crps = normal_challenges_crps(apuf, n, N, seed_instance, mu, sigma)
