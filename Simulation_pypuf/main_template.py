# coding = utf-8


import numpy as np 
import matplotlib.pyplot as plt
import math
import sys, os
from pathlib import Path

from xorpuf_attack import *
from apuf_attack import *
from util import *

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
	if len(sys.argv) == 5:
		n = int(sys.argv[1])
		noisiness = float(sys.argv[2])
		bias = float(sys,argv[3])
		position = sys.argv[4]
		k = int(sys.argv[5])
	else:
		n = 32
		noisiness = 0.1	
		bias_1 = 0.5
		bias_2 = 0.53
		bias_3 = 0.55
		position = 'bit'
		k = 5

	Path("./data").mkdir(parents=True, exist_ok=True)
	# Times of repeat experiment 
	repeat_experiment = 3
	'''
	Step 1: Create an instance of PUF
	'''
	weight_bias_initial = 0
	# puf_1 = instance_one_puf(n, noisiness, weight_bias_initial, bias_1, k)
	# puf_2 = instance_one_puf(n, noisiness, weight_bias_initial, bias_2, k)
	# puf_3 = instance_one_puf(n, noisiness, weight_bias_initial, bias_3, k)

	puf_bit = instance_one_puf(n, noisiness, weight_bias_initial, bias_2, k)
	puf_basis = instance_one_puf(n, noisiness, weight_bias_initial, bias_2, k)
	# instances = [puf_bit, puf_basis]
	# print(pypuf.metrics.uniqueness(instances, seed=31214, N=1000))
	# sys.exit()
	
	'''
	Step 2: Obatin simluiation result of such puf under logistic regresssion attack
	'''
	crps = crp_apuf(n, steps=20)
	# accuracy_cpuf = instance_one_apuf_attack_n(puf, crps, repeat_experiment, steps=10)
	# accuracy_cpuf_1 = instance_one_apuf_attack_n(puf_bit, crps, repeat_experiment, steps=20)
	accuracy_cpuf_2 = instance_one_apuf_attack_n(puf_bit, crps, repeat_experiment, steps=20)
	# accuracy_cpuf_3 = instance_one_apuf_attack_n(puf_bit, crps, repeat_experiment, steps=10)
	# accuracy_hpuf_1 = instance_one_hybrid_apuf_attack_n(puf_bit, puf_basis, crps, position, repeat_experiment, steps=20)
	accuracy_hpuf_2 = instance_one_hybrid_apuf_attack_n(puf_bit, puf_basis, crps, position, repeat_experiment, steps=20)
	# accuracy_hpuf_3 = instance_one_hybrid_apuf_attack_n(puf_bit, puf_basis, crps, position, repeat_experiment, steps=10)


	np.save('./data/crps_single_apuf_'+str(n)+'_'+str(k)+'.npy', crps)
	# np.save('./data/accuracy_cpuf_single_apuf_'+str(n)+'_'+str(bias_1)+'_'+str(k)+'.npy', accuracy_cpuf_1)
	np.save('./data/accuracy_cpuf_single_apuf_'+str(n)+'_'+str(bias_2)+'_'+str(k)+'.npy', accuracy_cpuf_2)
	# np.save('./data/accuracy_cpuf_single_apuf_'+str(n)+'_'+str(bias_3)+'_'+str(k)+'.npy', accuracy_cpuf_3)
	# np.save('./data/accuracy_hpuf_single_apuf_'+str(n)+'_'+str(bias_1)+'_'+position+'_'+str(k)+'.npy', accuracy_hpuf_1)
	np.save('./data/accuracy_hpuf_single_apuf_'+str(n)+'_'+str(bias_2)+'_'+position+'_'+str(k)+'.npy', accuracy_hpuf_2)
	# np.save('./data/accuracy_hpuf_single_apuf_'+str(n)+'_'+str(bias_3)+'_'+position+'_'+str(k)+'.npy', accuracy_hpuf_3)


	'''
	Step 3: Plot 
	'''
	# plt.plot(crps, accuracy_cpuf, label = 'cpuf')
	# plt.plot(crps, accuracy_hpuf_1, label = 'hpuf_1')
	plt.title("Modeling attack on CPUF")
	# plt.plot(crps, accuracy_cpuf_1, label='cpuf_bias_0.5')
	plt.plot(crps, accuracy_cpuf_2, label='cpuf_bias_0.53')
	# plt.plot(crps, accuracy_cpuf_3, label='cpuf_bias_0.6')
	# plt.plot(crps, accuracy_hpuf_1, label='hpuf_bit_bias_0.5')
	plt.plot(crps, accuracy_hpuf_2, label='hpuf_bit_bias_0.53')
	# plt.plot(crps, accuracy_hpuf_3, label='hpuf_bit_bias_0.6')
	plt.xlabel("Number of CRPs")
	plt.ylabel("Accuracy (x100%)")
	plt.legend()
	plt.show()
	


