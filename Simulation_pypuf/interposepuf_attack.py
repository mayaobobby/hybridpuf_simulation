# coding = utf-8

import pypuf.simulation, pypuf.io
import pypuf.attack
import pypuf.metrics

import numpy as np 
import matplotlib.pyplot as plt
import math
import sys, os, random

from util import *

def interpose(challenges, prev_responses):
	N = challenges.shape[0]
	n = challenges.shape[1]
	new_challenges = np.empty((N,n+1))
	for i in range(N):
		new_challenges[i] = np.insert(challenges[i], int(n/2), prev_responses[i])
	return new_challenges
	
def heuristic(challenges,responses,model):
	N,n = challenges.shape
	count_up_challenges = 0
	challenges_h = np.empty((0,n))
	responses_h = np.empty((0))
	for i in range(N):
		pos_challenge = np.insert(challenges[i],int(n/2),1)
		neg_challenge = np.insert(challenges[i],int(n/2),-1)
		test_responses = model.eval(np.array([pos_challenge,neg_challenge]))
		if(test_responses[0] == test_responses[1]):
			continue
		count_up_challenges+=1
		challenges_h = np.append(challenges_h,challenges[i])
		if(test_responses[0] == responses[i]):
			responses_h  = np.append(responses_h,1)
		else:
			responses_h  = np.append(responses_h,-1)
	challenges_h = np.reshape(challenges_h,(count_up_challenges,n))
	return pypuf.io.ChallengeResponseSet(challenges_h,responses_h)
			
def interpose_accuracy(lower_model,upper_model,crps):
	upper_responses = upper_model.eval(crps.challenges)
	lower_challenges = interpose(crps.challenges,upper_responses)
	lower_responses = lower_model.eval(lower_challenges)
	return pypuf.metrics.similarity_data(lower_responses,crps.responses)

def instance_one_hybrid_interposepuf_attack(puf, num_crps,p_guess):
	seed_instance = int.from_bytes(os.urandom(4), "big")
	crps = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=num_crps, seed=seed_instance)
	
	for k in range(num_crps):
		for i in range(crps.responses.shape[1]):
			crps.responses[k][i][0] = hybrid_flipping(crps.responses[k][i][0], p_guess) #Flip the response with probability 1 - P(guessing basis)

	crps_lower = pypuf.io.ChallengeResponseSet(crps.challenges.copy(),crps.responses.copy())
	random_guesses = 1-2*np.random.randint(2, size = num_crps)
	crps_lower.challenges = interpose(crps_lower.challenges,random_guesses)
	
	seed_instance_train = int.from_bytes(os.urandom(4), "big")
	lower_model = pypuf.attack.LRAttack2021(crps_lower, seed=seed_instance_train, k = 2, bs = 1000, lr=.001, epochs = 10).fit()
	upper_model = None
	
	MAX_LOOPS = 3
	loops = 0
	accuracy = 0
	while (loops < MAX_LOOPS):
		crps_upper = heuristic(crps.challenges,crps.responses,lower_model)
		
		seed_instance_train = int.from_bytes(os.urandom(4), "big")
		upper_model = pypuf.attack.LRAttack2021(crps_upper, seed=seed_instance_train, k = 2, bs = 1000, lr=.001, epochs = 10).fit()
		
		upper_responses = upper_model.eval(crps.challenges).flatten()
		crps_lower.challenges = interpose(crps.challenges, upper_responses)

		seed_instance_train = int.from_bytes(os.urandom(4), "big")
		lower_model = pypuf.attack.LRAttack2021(crps_lower, seed=seed_instance_train, k = 2, bs = 1000, lr=.001, epochs = 10).fit()
		loops+=1

	seed_instance_test = int.from_bytes(os.urandom(4), "big")
	crps_test = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=1000, seed=seed_instance)
	accuracy = interpose_accuracy(lower_model,upper_model,crps_test)
	return accuracy

def instance_one_hybrid_interposepuf_attack_n(puf, crps, repeat_experiment,p_guess = 0.5*(1+np.sqrt(0.5)), steps=10):
	accuracy_hpuf = np.array([])
	for i in range(steps):
		instance_accuracy_hpuf_repeat = np.zeros(repeat_experiment)
		N = int(crps[i])
		for j in range(repeat_experiment):
			instance_accuracy_hpuf_repeat[j] = instance_one_hybrid_interposepuf_attack(puf, N,p_guess)

		instance_accuracy_hpuf = np.mean(instance_accuracy_hpuf_repeat)	
	
		accuracy_hpuf = np.append(accuracy_hpuf, instance_accuracy_hpuf)

	return accuracy_hpuf

def crp_ipuf(n, steps=10):
	crps = np.array([])
	N = 1e2
	step = 0
	if n == 32:
		step = 1e4
	elif n == 64:
		step = 20e3
	elif n == 128:
		step = 5e2

	for i in range(steps):
		crps = np.append(crps, N)
		N = N + step

	return crps




'''
template of usage
'''
if __name__ == '__main__':
	n_size = 32
	k_up = 2
	k_down = 2
	
	seed_instance = int.from_bytes(os.urandom(4), "big")
	puf_interpose = pypuf.simulation.InterposePUF(n=n_size,k_up = k_up,k_down = k_down, seed=seed_instance)
	
	repeat_experiment = 1
	crps = crp_ipuf(n_size, steps = 20)

	p_guess = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	np.save('./data/crps_ipuf_n'+str(n_size)+'_kup'+str(k_up)+'_kdown'+str(k_down)+'.npy', crps)
	for i in range(len(p_guess)):
		accuracy_h = instance_one_hybrid_interposepuf_attack_n(puf_interpose, crps, repeat_experiment,p_guess[i], steps = 20)
		np.save('./data/hybrid_ipuf_accuracy_p'+str(p_guess[i])+'_n'+str(n_size)+'_kup'+str(k_up)+'_kdown'+str(k_down)+'.npy', accuracy_h)
	
	