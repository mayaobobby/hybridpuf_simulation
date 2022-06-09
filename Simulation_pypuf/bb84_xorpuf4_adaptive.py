import netsquid as ns
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from pathlib import Path

import pypuf.simulation, pypuf.io
from challenge_test import arbitrary_challenges

from apuf_attack import *

seed_puf_instances = []
seed_challenge_instances = []

'''
Description: The number of CRPs per step.
'''
def crp_apuf(n, steps=20):
	crps = np.array([])
	N = 1000
	step = 0
	if n == 64:
		step = 10e3
	elif n == 128:
		step = 40e3

	for i in range(steps):
		crps = np.append(crps, N)
		N += step

	return crps


'''
Parameter of CPUF:
n: input size
m: output size
N: N-sample of CRPs
k: For XORPUF (k-APUF in parallel)
noisiness_cpuf: CPUF device noise
'''
def CPUF_param(samples_instance, qubit_size):
	n = 128
	m = qubit_size*2
	N_samples = samples_instance
	k = 4
	noisiness_cpuf = 0

	return n, m, N_samples, k, noisiness_cpuf

'''
CPUF instance (n-bit challenge, m-bit response)
'''
def CPUF_gen(n, m, k, noisiness_cpuf):
	global seed_puf_instances
	seed_puf_instances, puf = [], []

	for i in range(m):
		seed_puf_instances.append(int.from_bytes(os.urandom(4), "big")) 
		puf.append(pypuf.simulation.XORArbiterPUF(n=n, noisiness=noisiness_cpuf, seed=seed_puf_instances[i], k=k)) 
	
	return puf

'''
CRP instances(n-bit challenge, m-bit response)
'''
def CRP_gen_one(n, m, N_samples, puf):
	seed_challenges = int.from_bytes(os.urandom(4), "big")
	challenges_instance = pypuf.io.random_inputs(n, 1, seed_challenges)
	challenges = np.zeros((N_samples, n))
	responses = np.zeros((N_samples, m))
	for i in range(m):
		crps_instances = arbitrary_challenges.random_challenges_crps(puf[i], n, 1, challenges_instance)
		responses[0][i] = crps_instances.responses

	for j in range(N_samples):
		challenges[j,:] = challenges_instance
		responses[j,:] = responses[0,:]

	global seed_challenge_instances

	
	seed_challenge_instances.append(seed_challenges)
	# pp: Post-processing
	challenges_pp = (1 - challenges) // 2
	responses_pp = (1 - responses) // 2

	return challenges_pp, responses_pp
		

'''
Program to measure a qubit with a basis by optimal adversary (Aadpative)
'''
def OptimalMeasurement_adptive(bases_eve, bases_reference, states_reference):
	states_eve = np.zeros(len(states_reference))
	for i in range(len(bases_reference)):
		if bases_eve[i] == bases_reference[i]:
			states_eve[i] = states_reference[i]
		else:
			states_eve[i] = np.random.randint(2, size=1)
	return states_eve
		


def run_experiment_adptive_attack(n, m, N_samples, k, noisiness_cpuf, qubit_size):

	puf = CPUF_gen(n, m, k, noisiness_cpuf)
	challenges_pp, responses_pp = CRP_gen_one(n, m, N_samples, puf)

	runs = N_samples

	bases_correct = []
	counter_bases_match = 0

	incorrect_basis = []

	bases_eve = np.random.randint(2, size=qubit_size)

	states_eve_record = np.zeros((runs, qubit_size))

	for i in range(runs):
		for j in range(qubit_size):

			bases_reference = responses_pp[0][1::2]
			states_reference = responses_pp[0][0::2]

			states_eve = OptimalMeasurement_adptive(bases_eve, bases_reference, states_reference)
			states_eve_record[i][:] = states_eve


			if i >= 1:	
				if j in bases_correct:
					pass
				elif states_eve_record[i][j] != states_eve_record[i-1][j]:
					bases_eve[j] = np.abs(1-bases_eve[j])
					bases_correct.append(j)

		if i >= 1:	
			if (bases_eve == bases_reference).all():
				counter_bases_match = i+1
				incorrect_basis[:qubit_size] = [0]*qubit_size
				break
		
		if i >= 4:
			for k in range(qubit_size):
				if bases_eve[k] != bases_reference[k]:
					incorrect_basis.append(1)
				else:
					incorrect_basis.append(0)
			break

	return counter_bases_match, incorrect_basis


def run_experiment_adptive_attack_average(n, m, N_samples, k, noisiness_cpuf, repeat_time=1000, qubit_size=1):

	counter_queries = []
	iteration = 1000

	correct_bases = [1000]*qubit_size
	incorrect_basis = []

	for i in range(iteration):

		counter_bases_match = 0
		counter_bases_match, incorrect_basis = run_experiment_adptive_attack(n, m, N_samples, k, noisiness_cpuf, qubit_size)
		
		counter_queries.append(counter_bases_match)
		correct_bases = [x1 - x2 for (x1, x2) in zip(correct_bases, incorrect_basis)]

	query_adaptive = sum(counter_queries)/len(counter_queries)
	accuracy = sum([x / iteration for x in correct_bases])/qubit_size

	print("Number of adpative queries:", query_adaptive)
	print("Correct rate:", accuracy)
	return query_adaptive, accuracy 

'''
Description: Emulation the underlying CPUF of HPUF with BB84 encoding
'''
def bb84_xorpuf4(puf_bit, puf_basis, position, k, steps, success_prob, query_adaptive):

	# Times of repeat experiment (for each number of CRPs)
	repeat_experiment = 20

	###################################################################
	# Obtain simulation result of HPUF under logistic regression attack
	###################################################################
	crps = crp_apuf(n, steps)
	
	if position == 'basis':
		accuracy_hpuf = instance_one_hybrid_apuf_attack_n(success_prob, puf_bit, puf_basis, crps, position, repeat_experiment, steps)
		np.save('./data/xorpuf4/'+str(n)+'n_xorpuf4_adaptive_crps.npy', crps*query_adaptive)
		np.save('./data/xorpuf4/'+str(n)+'h_xorpuf4_adaptive_'+position+'_a.npy', accuracy_hpuf)

	return crps, accuracy_hpuf

if __name__ == '__main__':
	'''
	Simulation of HPUF with BB84 encoding and an underlying of 4XORPUF against adaptive adversaries
	
	Variables:
	n: length of challenge, it defaults to 64 bits
	noisiness: noisiness, it effects the reliability(robustness) of CPUF instance
 	k: k value (CPUF construction per bit of response)

 	Steps:
 	1. Emulate the accuracy of obatined CRPs with adaptive queries.
	2. Emulate the input-output behavior of CPUF that encodes basis value with the number of required CRPs. 
	3. Plot the result (Also find the plot.py script in data folder if running separately).
	'''
	# Enable GPU/CPU (optional)
	# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
	Path("./data/xorpuf4").mkdir(parents=True, exist_ok=True)

	qubit_size = 1
	samples=100
	n, m, N_samples, k, noisiness_cpuf = CPUF_param(samples, qubit_size)

	# Times of iterations (with increment of CRPs)
	steps = 20
	
	###########################
	# Create instances of CPUFs
	###########################
	puf_instances = CPUF_gen(n, m, k, noisiness_cpuf)

	puf_bit = puf_instances[0]
	puf_basis = puf_instances[1]
	query_adaptive, accuracy = run_experiment_adptive_attack_average(n, m, N_samples, k, noisiness_cpuf, qubit_size)

	success_prob = accuracy

	crps, accuracy_hpuf = bb84_xorpuf4(puf_bit, puf_basis, 'basis', k, steps, success_prob, query_adaptive)

	print(query_adaptive)