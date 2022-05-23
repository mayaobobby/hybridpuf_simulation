# coding = utf-8

from apuf_attack import *
import numpy as np
import matplotlib.pyplot as plt

'''
Description: The number of CRPs per step.
'''
def crp_apuf(n, steps=10):
	crps = np.array([])
	N = 1000
	step = 25000
	if n == 64:
		step = 65e3
	elif n == 128:
		step = 50e4

	for i in range(steps):
		crps = np.append(crps, N)
		N += step

	return crps

'''
Simulation of HPUFs with various adversarial guessing probabilities
Data can be plotted by running noise_data/plot.py
'''
p_guess = [0.5,0.55,0.6,0.7]
repeat_experiment = 30
n = 32
noisiness_cpuf = 0
k = 5
steps = 20
crps = crp_apuf(n,steps)
seed_instance_bit = int.from_bytes(os.urandom(4), "big")
seed_instance_basis = int.from_bytes(os.urandom(4), "big")

puf_bit = pypuf.simulation.XORArbiterPUF(n=n, noisiness=noisiness_cpuf, seed=seed_instance_bit, k=k)
puf_basis = pypuf.simulation.XORArbiterPUF(n=n, noisiness=noisiness_cpuf, seed=seed_instance_basis, k=k)
position = 'bit'

np.save('./noise_data/xorpuf'+str(k)+'_n'+str(n)+'_reps'+str(repeat_experiment)+'_crps.npy',crps)
for i in range(len(p_guess)):
    accuracy_hpuf = instance_one_hybrid_apuf_attack_n(p_guess[i], puf_bit, puf_basis, crps, position, repeat_experiment, steps)
    np.save('./noise_data/xorpuf'+str(k)+'_n'+str(n)+'_p'+str(p_guess[i])+'_reps'+str(repeat_experiment)+'_accuracy.npy',accuracy_hpuf)