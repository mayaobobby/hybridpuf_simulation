import netsquid as ns
import numpy as np
import os, sys
import matplotlib.pyplot as plt

import pypuf.simulation, pypuf.io
from challenge_test import arbitrary_challenges

seed_puf_instances = []
seed_challenge_instances = []


# N_runs samples for each run
N_runs = 100
'''
Parameter of CPUF:
n: input size
m: output size
N: N-sample of CRPs
k: For XORPUF (k-APUF in parallel)
puf_noisiness: CPUF device noise
'''
def CPUF_param(qubit_size):
	n = 64
	m = qubit_size*2
	N = N_runs
	k = 4
	puf_noisiness = 0

	return n, m, N, k, puf_noisiness

'''
CPUF instance (n-bit challenge, m-bit response)
'''
def CPUF_gen(n, m, k, puf_noisiness):
	global seed_puf_instances
	seed_puf_instances, puf = [], []

	for i in range(m):
		seed_puf_instances.append(int.from_bytes(os.urandom(4), "big")) 
		puf.append(pypuf.simulation.XORArbiterPUF(n=n, noisiness=puf_noisiness, seed=seed_puf_instances[i], k=k)) 
	
	return puf

'''
CRP instances(n-bit challenge, m-bit response)
'''
def CRP_gen(n, m, N, puf):
	seed_challenges = int.from_bytes(os.urandom(4), "big")
	challenges = pypuf.io.random_inputs(n, N, seed_challenges)
	responses = np.zeros((N, m))
	for i in range(m):
		crps_instances = arbitrary_challenges.random_challenges_crps(puf[i], n, N, challenges)
		# print(crps_instances.responses)
		# sys.exit()
		for j in range(N):
			responses[j][i] = crps_instances.responses[j][0][0]

	global seed_challenge_instances

	
	seed_challenge_instances.append(seed_challenges)
	# pp: Post-processing
	challenges_pp = (1 - challenges) // 2
	responses_pp = (1 - responses) // 2

	return challenges_pp, responses_pp

'''
CRP instances(n-bit challenge, m-bit response)
'''
def CRP_gen_one(n, m, N, puf):
	seed_challenges = int.from_bytes(os.urandom(4), "big")
	challenges_instance = pypuf.io.random_inputs(n, 1, seed_challenges)
	challenges = np.zeros((N, n))
	responses = np.zeros((N, m))
	for i in range(m):
		crps_instances = arbitrary_challenges.random_challenges_crps(puf[i], n, 1, challenges_instance)
		responses[0][i] = crps_instances.responses

	for j in range(N):
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
		


def run_experiment_adptive_attack(qubit_size=128):

	n, m, N, k, puf_noisiness = CPUF_param(qubit_size)

	puf = CPUF_gen(n, m, k, puf_noisiness)
	challenges_pp, responses_pp = CRP_gen_one(n, m, N, puf)

	# print("challenge:", challenges_pp)
	# print("responses:", responses_pp)

	runs = N

	bases_correct = []
	counter_bases_match = 0

	incorrect_basis = []

	bases_eve = np.random.randint(2, size=qubit_size)

	states_eve_record = np.zeros((runs, qubit_size))

	for i in range(runs):
		for j in range(qubit_size):

			bases_reference = responses_pp[0][1::2]
			states_reference = responses_pp[0][0::2]

			# print("bases_reference:", bases_reference)
			# print("states_reference:", states_reference)

			states_eve = OptimalMeasurement_adptive(bases_eve, bases_reference, states_reference)
			states_eve_record[i][:] = states_eve
			# print("bases_eve:", bases_eve)
			# print("states_eve:", states_eve_record[i][:])

			# sys.exit()

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
			# counter_bases_match = i+1
			# incorrect_basis = 1
			# break
			for k in range(qubit_size):
				if bases_eve[k] != bases_reference[k]:
					incorrect_basis.append(1)
				else:
					incorrect_basis.append(0)
			break


	return counter_bases_match, incorrect_basis


if __name__ == '__main__':

	qubit_size = 16

	counter_queries = []
	iteration = 1000

	correct_bases = [1000]*qubit_size
	incorrect_basis = []

	for i in range(iteration):

		counter_bases_match = 0
		counter_bases_match, incorrect_basis = run_experiment_adptive_attack(qubit_size)
		
		counter_queries.append(counter_bases_match)
		# print(counter_bases_match, incorrect_basis)
		# incorrect_bases = incorrect_bases + incorrect_basis
		correct_bases = [x1 - x2 for (x1, x2) in zip(correct_bases, incorrect_basis)]

	print("Number of adpative queries:", sum(counter_queries)/len(counter_queries))
	print("Correct rate:", [x / iteration for x in correct_bases])


