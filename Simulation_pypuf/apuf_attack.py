# coding = utf-8

import pypuf.simulation, pypuf.io
import pypuf.attack
import pypuf.metrics

import numpy as np 
import matplotlib.pyplot as plt
import math
import sys, os, random

from util import *

parampuf_dic = {
	'n64k4': './state_value/xorpuf_4_collection_bit/64h_xorpuf4_a.npy',
	'n128k4': './state_value/xorpuf_4_collection_bit/128h_xorpuf4_a.npy',
	'n64k5': './state_value/xorpuf_5_collection_bit/64h_xorpuf5_a.npy',
	'n128k5': './state_value/xorpuf_5_collection_bit/128h_xorpuf5_a.npy'
}

crps_dic = {
	'n64k4': './state_value/xorpuf_4_collection_bit/64n_xorpuf4_crps.npy',
	'n128k4': './state_value/xorpuf_4_collection_bit/128n_xorpuf4_crps.npy',
	'n64k5': './state_value/xorpuf_5_collection_bit/64n_xorpuf5_crps.npy',
	'n128k5': './state_value/xorpuf_5_collection_bit/128n_xorpuf5_crps.npy'
}

'''
Description: Dictionary of CRPs & accuracy on state value
'''
def basis_prediction_dict(n,k):
	keyword_accuracy_bit = 'n'+str(n)+'k'+str(k)
	accuracy_bit = np.load(parampuf_dic[keyword_accuracy_bit])
	
	keyword_crps = 'n'+str(n)+'k'+str(k)
	crps_bit = np.load(crps_dic[keyword_crps])

	for i in range(crps_bit.size):
		if accuracy_bit[i] >.95:
			crps_bit_threshold = crps_bit[i]
			crps_bit_threshold -= 1000
			accuracy_bit_threshold = accuracy_bit[i]
			break

	return crps_bit_threshold, accuracy_bit_threshold

'''
Description: Return the uniqness of pufs
'''
def uniqueness(n, noisiness_cpuf, weight_bias, bias, k):
	seed_uniqness = int.from_bytes(os.urandom(4), "big")
	instances = [instance_one_puf(n, noisiness_cpuf, weight_bias, bias, k=k) for i in range(5)]
	uniqness = pypuf.metrics.uniqueness(instances, seed=seed_uniqness, N=1000)

	return uniqness

'''
Description: Return the bias of a puf
'''	
def puf_bias(puf):
	seed_bias = int.from_bytes(os.urandom(4), "big")
	bias_value =  pypuf.metrics.bias(puf, seed_bias, N=300000)
	bias_prob = round(1/2 - bias_value/2,2)

	return bias_prob


'''
Description: Instantiate an arbiter PUF chain
'''
def instance_one_puf(size_challenge, noisiness_cpuf, weight_bias, bias, k=1):
	seed_instance = int.from_bytes(os.urandom(4), "big")
	if k == 1:
		puf = pypuf.simulation.ArbiterPUF(n=size_challenge, noisiness=noisiness_cpuf, seed=seed_instance)
	else:
		puf = pypuf.simulation.XORArbiterPUF(n=size_challenge, noisiness=noisiness_cpuf, seed=seed_instance, k=k)
	
	while puf_bias(puf) < bias-0.005:
		for i in range(k):
			weight_bias = weight_bias - 0.05
			puf.weight_array[i][size_challenge] = np.array([[weight_bias]])

	# print(weight_bias)
	# print(puf_bias(puf))
	return puf

'''
Description: Return the reliability of a puf
'''
def puf_reliability(puf):
	seed_reliability = int.from_bytes(os.urandom(4), "big")
	reliability_instance = np.average(pypuf.metrics.reliability(puf, seed_reliability))
	
	return reliability_instance

'''
Description: Arbiter-based PUF under ML attacks with classical structure. (one chain)
'''
def instance_one_apuf_attack(puf, num_crps, num_bs, num_epochs):
	seed_instance = int.from_bytes(os.urandom(4), "big")
	crps = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=num_crps, seed=seed_instance)

	crps_test = int(num_crps*0.9)
	if crps_test > int(1e4):
		crps_test = int(1e4)
	seed_instance_train = int.from_bytes(os.urandom(4), "big")
	attack = pypuf.attack.LRAttack2021(crps, seed=seed_instance_train, k=puf.k, bs=num_bs, lr=.001, epochs=num_epochs)
	attack.fit()

	model = attack.model

	seed_instance_test = int.from_bytes(os.urandom(4), "big")
	accuracy = pypuf.metrics.similarity(puf, model, seed=seed_instance_test, N=crps_test)

	return accuracy


'''
Description: Arbiter-based PUF under ML attacks with classical structure. (one chain with repetitions)
'''
def instance_one_apuf_attack_n(puf, crps, repeat_experiment, steps):
	accuracy_cpuf = np.array([])
	for i in range(steps):
		instance_accuracy_cpuf_repeat = np.zeros(repeat_experiment)
		N = int(crps[i])
		for j in range(repeat_experiment):
			instance_accuracy_cpuf_repeat[j] = instance_one_apuf_attack(puf, N, num_bs=1000, num_epochs=100)
		print(instance_accuracy_cpuf_repeat)
		print("CPUF Median:", np.median(instance_accuracy_cpuf_repeat))
		instance_accuracy_cpuf = np.mean(instance_accuracy_cpuf_repeat)	
		print("CPUF Mean:", instance_accuracy_cpuf)
	
		accuracy_cpuf = np.append(accuracy_cpuf, instance_accuracy_cpuf)

	return accuracy_cpuf


'''
Description: Arbiter PUF under ML attacks with classical structure. (two chains)
'''
def instance_two_apuf_attack(puf_bit, puf_basis, crps, repeat_experiment):
	accuracy_basis = instance_one_apuf_attack_n(puf_basis, crps, repeat_experiment)
	accuracy_bit = instance_one_apuf_attack_n(puf_bit, crps, repeat_experiment)

	accuracy_two_apuf = accuracy_basis * accuracy_bit

	return accuracy_two_apuf



'''
Description: Arbiter PUF under ML attacks with hybrid structure. (one chain)
'''
def instance_one_hybrid_apuf_attack(coe_hdata, puf_bit, puf_basis, num_crps, position, num_bs, num_epochs):
	bias_basis = puf_bias(puf_basis)
	# bias_bit = puf_bias(puf_bit)
	seed_basis = int.from_bytes(os.urandom(4), "big")
	seed_bit = int.from_bytes(os.urandom(4), "big")
	crps_basis = pypuf.io.ChallengeResponseSet.from_simulation(puf_basis, N=num_crps, seed=seed_basis)
	crps_bit = pypuf.io.ChallengeResponseSet.from_simulation(puf_bit, N=num_crps, seed=seed_bit)
	res_basis_cpy = np.copy(crps_basis.responses)
	res_bit_cpy = np.copy(crps_bit.responses)
	seed_instance_train = int.from_bytes(os.urandom(4), "big")

	crps_test = int(num_crps*0.9)
	if crps_test > int(1e4):
		crps_test = int(1e4)
	
	if position == 'bit':
		'''
		bit value 
		'''
		for i in range(crps_bit.responses.size):
			crps_bit.responses[i][0][0] = hybrid_flipping(crps_bit.responses[i][0][0], coe_hdata)
			
		# print('Similarity basis:', pypuf.metrics.similarity_data(crps_basis.responses, res_basis_cpy))		
		# print('Similarity bit:', pypuf.metrics.similarity_data(crps_bit.responses, res_bit_cpy))	
		attack = pypuf.attack.LRAttack2021(crps_bit, seed=seed_instance_train, k=puf_bit.k, bs=num_bs, lr=.001, epochs=num_epochs)
		attack.fit()
		model = attack.model
		seed_instance_test = int.from_bytes(os.urandom(4), "big")
		accuracy = pypuf.metrics.similarity(puf_bit, model, seed=seed_instance_test, N=crps_test)	

	elif position == 'basis':
		'''
		basis value
		'''	
		_,accuracy_bit_threshold = basis_prediction_dict(puf_basis.n,puf_basis.k)
		for i in range(crps_basis.responses.size):
			crps_basis.responses[i][0][0] = hybrid_flipping(crps_basis.responses[i][0][0], coe_hdata*accuracy_bit_threshold)

		attack = pypuf.attack.LRAttack2021(crps_basis, seed=seed_instance_train, k=puf_basis.k, bs=num_bs, lr=.001, epochs=num_epochs)
		attack.fit()
		model = attack.model
		seed_instance_test = int.from_bytes(os.urandom(4), "big")	
		accuracy = pypuf.metrics.similarity(puf_basis, model, seed=seed_instance_test, N=crps_test)

	return accuracy

'''
Description: Arbiter-based PUF under ML attacks with hybrid structure. (one chain with repetitions)
'''
def instance_one_hybrid_apuf_attack_n(coe_hdata, puf_bit, puf_basis, crps, position, repeat_experiment, steps):
	accuracy_hpuf = np.array([])
	for i in range(steps):
		instance_accuracy_hpuf_repeat = np.zeros(repeat_experiment)
		N = int(crps[i])
		for j in range(repeat_experiment):
			instance_accuracy_hpuf_repeat[j] = instance_one_hybrid_apuf_attack(coe_hdata, puf_bit, puf_basis, N, position, num_bs=1000, num_epochs=100)
		print(instance_accuracy_hpuf_repeat)
		print("HPUF Median:", np.median(instance_accuracy_hpuf_repeat))
		instance_accuracy_hpuf = np.mean(instance_accuracy_hpuf_repeat)	
		print("HPUF Mean:", instance_accuracy_hpuf)
	
		accuracy_hpuf = np.append(accuracy_hpuf, instance_accuracy_hpuf)

	return accuracy_hpuf

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
	



