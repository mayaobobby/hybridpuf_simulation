# coding = utf-8


import numpy as np 
import matplotlib.pyplot as plt
import math
import sys, os
from pathlib import Path

from apuf_attack import *

'''
Parameter of CPUF:
n: input size
m: output size
N: N-sample of CRPs
k: For XORPUF (k-APUF in parallel)
noisiness_cpuf: CPUF device noise
'''
def CPUF_param(qubit_size):
	n = 32
	m = qubit_size*2
	k = 5
	noisiness_cpuf = 0

	return n, m, k, noisiness_cpuf


'''
Description: The number of CRPs per step.
'''
def crp_apuf(n, steps=20):
	crps = np.array([])
	N = 1000
	step = 0
	if n == 32:
		step = 1e4
	elif n == 64:
		step = 32.5e3
	elif n == 128:
		step = 25e4

	for i in range(steps):
		crps = np.append(crps, N)
		N += step

	return crps

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
Description: Emulation the underlying CPUF of HPUF with BB84 encoding
'''
def bb84_xorpuf5(puf_bit, puf_basis, position, k, steps, success_prob):

	# Times of repeat experiment (for each number of CRPs)
	repeat_experiment = 20

	
	###################################################################
	# Obtain simulation result of HPUF under logistic regression attack
	###################################################################
	crps = crp_apuf(n, steps)
	# accuracy_cpuf = instance_one_apuf_attack_n(puf_bit, crps, repeat_experiment, steps)
	# accuracy_hpuf = instance_one_hybrid_apuf_attack_n(success_prob, puf_bit, puf_basis, crps, position, repeat_experiment, steps)
	
	if position == 'bit':
		accuracy_cpuf = instance_one_apuf_attack_n(puf_bit, crps, repeat_experiment, steps)
		accuracy_hpuf = instance_one_hybrid_apuf_attack_n(success_prob, puf_bit, puf_basis, crps, position, repeat_experiment, steps)
		# np.save('./data/xorpuf5/'+str(n)+'n_xorpuf5_crps.npy', crps)
		# np.save('./data/xorpuf5/'+str(n)+'c_xorpuf5_a.npy', accuracy_cpuf)
		# np.save('./data/xorpuf5/'+str(n)+'h_xorpuf5_'+position+'_a.npy', accuracy_hpuf)
	elif position == 'basis':
		accuracy_cpuf = np.ones(steps)
		accuracy_hpuf = instance_one_hybrid_apuf_attack_n(success_prob, puf_bit, puf_basis, crps, position, repeat_experiment, steps)
		# np.save('./data/xorpuf5/'+str(n)+'h_xorpuf5_'+position+'_a.npy', accuracy_hpuf)
	
	return crps, accuracy_cpuf, accuracy_hpuf

'''
Description: Plot the result.
'''
def plot(crps_bit, accuracy_cpuf_bit, accuracy_hpuf_bit, accuracy_hpuf_basis):
	a = crps_bit
	b = accuracy_cpuf_bit
	c = accuracy_hpuf_bit
	d = accuracy_hpuf_basis

	for i in range(a.size):
		if c[i] >= .95:
			crps_bit_threshold = a[i]
			crps_bit_threshold -= 1000
			accuracy_bit_threshold = c[i]
			count = i
			break

	a_final = np.concatenate((a[:count],a+crps_bit_threshold))

	b_add = np.repeat(None, a_final.size-a.size)
	c_add = np.repeat(None, a_final.size-a.size)
	d_add = np.repeat(None, count)

	b_final = np.concatenate((b,b_add))
	c_final = np.concatenate((c,c_add))
	d_final = np.concatenate((d_add,d))

	plt.title('n='+str(n)+', k=5', fontsize=15)
	plt.plot(a_final, b_final, label = 'cpuf')
	plt.plot(a_final, c_final, label = 'hpuf:state', linestyle='dashed')
	plt.plot(a_final, d_final, label = 'hpuf:basis/both')
	plt.vlines(a_final[count], c_final[count], d_final[count], linestyles='dotted', label='basis learing start')
	plt.xlabel("Number of CRPs", fontsize=12)
	plt.ylabel("Success probability of modeling CPUFs(x100%)", fontsize=12)
	plt.legend(loc='lower right')
	plt.show()


if __name__ == '__main__':
	'''
	Simulation of HPUF with BB84 encoding and an underlying of 5XORPUF against weak adversaries
	
	Variables:
	n: length of challenge, it defaults to 64 bits
	noisiness: noisiness, it effects the reliability(robustness) of CPUF instance
 	k: k value (CPUF construction per bit of response)

 	Steps:
 	1. Emulate the input-output behavior of CPUF that encodes bit value with the number of required CRPs.
	2. Emulate the input-output behavior of CPUF that encodes basis value with the number of required CRPs.
	3. Obtain the result of total number of required CRPs that emulates CPUFs that encodes a qubit. 
	4. Plot the result (Also find the plot.py script in data folder if running separately).
	'''
	# Enable GPU/CPU (optional)
	os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
	# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
	Path("./data/xorpuf5").mkdir(parents=True, exist_ok=True)

	

	###########################
	# Create instances of CPUFs
	###########################
	qubit_size = 1
	n, m, k, noisiness_cpuf = CPUF_param(qubit_size)
	# Times of iterations (with increment of CRPs)
	steps = 20
	# Optimal success probability of obtain actual bit/basis value with bb84 encoding 
	success_prob = 0.85
	
	puf_instances = CPUF_gen(n, m, k, noisiness_cpuf)

	puf_bit = puf_instances[0]
	puf_basis = puf_instances[1]

	crps_bit, accuracy_cpuf_bit, accuracy_hpuf_bit = bb84_xorpuf5(puf_bit, puf_basis, 'bit', k, steps, success_prob)
	_, _, accuracy_hpuf_basis = bb84_xorpuf5(puf_bit, puf_basis, 'basis', k, steps, success_prob)

	# plot(crps_bit, accuracy_cpuf_bit, accuracy_hpuf_bit, accuracy_hpuf_basis)