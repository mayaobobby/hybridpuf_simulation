# coding = utf-8


import numpy as np 
import matplotlib.pyplot as plt
import math
import sys, os
from pathlib import Path

from apuf_attack import *
from util import *


def template(n, noisiness_cpuf, bias, position, k, steps, coe_hdata):

	# Times of repeat experiment 
	repeat_experiment = 10

	# Step 1: Create an instance of PUF

	weight_bias_initial = 0
	
	# puf_bit = instance_one_puf(n, noisiness_cpuf, weight_bias_initial, bias, k)
	# puf_basis = instance_one_puf(n, noisiness_cpuf, weight_bias_initial, bias, k)

	seed_instance_bit = int.from_bytes(os.urandom(4), "big")
	seed_instance_basis = int.from_bytes(os.urandom(4), "big")

	if k == 1:
		puf_bit = pypuf.simulation.ArbiterPUF(n=n, noisiness=noisiness_cpuf, seed=seed_instance_bit)
		puf_basis = pypuf.simulation.ArbiterPUF(n=n, noisiness=noisiness_cpuf, seed=seed_instance_basis)
	else:
		puf_bit = pypuf.simulation.XORArbiterPUF(n=n, noisiness=noisiness_cpuf, seed=seed_instance_bit, k=k)
		puf_basis = pypuf.simulation.XORArbiterPUF(n=n, noisiness=noisiness_cpuf, seed=seed_instance_basis, k=k)
	

	crps_bit_threshold,_ = basis_prediction_dict(n,k)

	
	# step 2: Obatin simluiation result of such puf under logistic regresssion attack
	
	crps = crp_apuf(n, steps)
	accuracy_cpuf_1 = instance_one_apuf_attack_n(puf_bit, crps, repeat_experiment, steps)
	input("Press Enter to continue...")
	accuracy_hpuf_1 = instance_one_hybrid_apuf_attack_n(coe_hdata, puf_bit, puf_basis, crps, position, repeat_experiment, steps)
	

	if position == 'basis':
		crps += crps_bit_threshold
	
	np.save('./data/'+str(n)+'n_xorpuf'+str(k)+'_'+position+'_crps.npy', crps)
	# np.save('./data/'+str(n)+'c_xorpuf'+str(k)+'_a.npy', accuracy_cpuf_1)
	np.save('./data/'+str(n)+'h_xorpuf'+str(k)+'_'+position+'_a.npy', accuracy_hpuf_1)

	

	
	# Step 3: Plot 

	plt.title("Modeling attack on CPUF")
	plt.plot(crps, accuracy_cpuf_1, label='cpuf')
	plt.plot(crps, accuracy_hpuf_1, label='hpuf')
	plt.xlabel("Number of CRPs")
	plt.ylabel("Accuracy (x100%)")
	plt.legend()
	plt.show()


if __name__ == '__main__':

	'''
	To run the template program, run the command in the termninal as follows: 
	python main_template.py varible1 variable2 variable3 variable4

	or 

	python main_template.py 

	with defaults

	For the variables:

	variable1: length of challenge, it defaults to 64 bits
	variable2: noisiness, it effects the reliability(robustness) of CPUF instance, it defaults to 0.1 (It turns out to be 95% around of reliability)
	variable3: the biased distribution on responses, it defaults to 0.5. 
 	variable4: basis, bit or bothaccuracy of hpuf, it defaults to 'bit'
 	variable5: kXORPUF
	'''
	# Enable GPU/CPU (optional)
	# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

	if len(sys.argv) == 6:
		n = int(sys.argv[1])
		noisiness = float(sys.argv[2])
		bias = float(sys,argv[3])
		position = sys.argv[4]
		k = int(sys.argv[5])
	else:
		n = 128
		noisiness_cpuf = 0
		bias = 0.5
		position = 'bit'
		k = 5

	steps = 1
	coe_hdata = 0.85

	Path("./data").mkdir(parents=True, exist_ok=True)
	template(n, noisiness_cpuf, bias, position, k, steps, coe_hdata)
	