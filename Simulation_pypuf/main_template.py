# coding = utf-8


import numpy as np 
import matplotlib.pyplot as plt
import math
import sys, os
from pathlib import Path


import pypuf.simulation, pypuf.io
import pypuf.attack
import pypuf.metrics

from xorpuf_attack import *
from apuf_attack import *
from util import *

if __name__ == '__main__':
	

	'''
	To run the template program, run the command in the termninal as follows: 
	python main_template.py varible1 variable2 variable3 

	For the variables:

	variable1: length of challenge, it defaults to 64 bits
	variable2: noisiness, it effects the reliability(robustness) of CPUF instance, it defaults to 0.1 (It turns out to be 95% around of reliability)
	variable3: the biased distribution on responses, it defaults to 0.5. 
 	variable4: basis, bit or bothaccuracy of hpuf, it defaults to 'bit'


	'''
	if len(sys.argv) == 5:
		n = int(sys.argv[1])
		noisiness = float(sys.argv[2])
		bias = float(sys,argv[3])
		position = sys.argv[4]
	else:
		n = 64
		noisiness = 0.1	
		bias = 0.6
		position = 'bit'

	Path("./data").mkdir(parents=True, exist_ok=True)
	# Times of repeat experiment 
	repeat_experiment = 5
	'''
	Step 1: Create and instance of PUF
	'''
	weight_bias_initial = 0
	puf = instance_one_apuf(n, noisiness, weight_bias_initial, bias)
	# bias = puf_bias(puf)
	# print(bias)

	'''
	Step 2: Obatin simluiation result of such puf under logistic regresssion attack
	'''
	crps = np.array([])
	accuracy_cpuf = np.array([])
	
	crps = crp_apuf(crps, n)
	accuracy_cpuf = instance_one_apuf_attack_n(puf, crps, repeat_experiment)
	accuracy_hpuf = instance_one_hybrid_apuf_attack_n(puf, crps, position, repeat_experiment)

	np.save('./data/crps_single_apuf_'+str(n)+'_'+str(bias)+'_'+'.npy', crps)
	np.save('./data/accuracy_cpuf_single_apuf_'+str(n)+'_'+str(bias)+'.npy', accuracy_cpuf)
	np.save('./data/accuracy_hpuf_single_apuf_'+str(n)+'_'+str(bias)+'_'+position+'.npy', accuracy_hpuf)


	'''
	Step 3: Plot 
	'''
	plt.plot(crps, accuracy_cpuf, label = 'cpuf')
	plt.plot(crps, accuracy_hpuf, label = 'hpuf')
	plt.xlabel("Number of CRPs")
	plt.ylabel("Accuracy (x100%)")
	plt.legend()
	plt.show()
	


