import numpy as np 
import matplotlib.pyplot as plt
import math
import random
import sys, os

def readline_from_summary(filename, puf_type):
	with open(filename,"r") as fi:
		crps, accuracy_avg = [], []
		if puf_type == 'CPUF':
			for ln in fi:
				if ln.startswith("CRPs:"):
					value = float(ln[5:])
					crps.append(value)
				if ln.startswith("CPUF Mean:"):
					value = float(ln[10:])
					accuracy_avg.append(value)
				if ln.startswith("Success Rate:"):
					value = float(ln[13:])

		elif puf_type == 'HPUF':
			for ln in fi:
				if ln.startswith("Adaptive Query:"):
					coe = float(ln[15:])
				if ln.startswith("CRPs:"):
					value = float(ln[5:])
					crps.append(value)
				if ln.startswith("HPUF Mean:"):
					value = float(ln[10:])
					accuracy_avg.append(value)
				if ln.startswith("Success Rate:"):
					value = float(ln[13:])

			crps = [x*coe for x in crps]

		elif puf_type == 'HLPUF':
			for ln in fi:
				if ln.startswith("CRPs:"):
					value = float(ln[5:])
					crps.append(value)
				if ln.startswith("HPUF Mean:"):
					value = float(ln[10:])
					accuracy_avg.append(value)
				if ln.startswith("Success Rate:"):
					value = float(ln[13:])

		else:
			pass

	return crps, accuracy_avg


def hybrid_flipping(value_original, success_prob):
	
	value_updated = np.ones((int(j)),dtype=int)
	for i in range(value_original.size):
		value_p = random.random()
		if value_p >= success_prob:
			value_updated[i] = -value_updated[i]
		else:
			pass
	return value_updated

if __name__ == '__main__':	

	cpuf_filename = "c_summary_xorpuf4_128.txt"
	hpuf_filename = "h_summary_xorpuf4_128.txt"
	hlpuf_filename = "hl_summary_xorpuf4_128.txt"


	cpuf_crps, cpuf_accuracy_avg = readline_from_summary(cpuf_filename, 'CPUF')
	hpuf_crps, hpuf_accuracy_avg = readline_from_summary(hpuf_filename, 'HPUF')
	hlpuf_crps, hlpuf_accuracy_avg = readline_from_summary(hlpuf_filename, 'HLPUF')
	hlpuf_bit_crps, hlpuf_basis_crps =  hlpuf_crps[:20], hlpuf_crps[20:]
	hlpuf_bit_accuracy_avg, hlpuf_basis_accuracy_avg = hlpuf_accuracy_avg[:20], hlpuf_accuracy_avg[20:]

	for i in range(len(hlpuf_bit_crps)):
		if hlpuf_bit_accuracy_avg[i] >.95:
			count = i
			crps_bit_threshold = hlpuf_bit_crps[count]
			crps_bit_threshold -= 1000
			break
	
	hlpuf_basis_crps = [x+crps_bit_threshold for x in hlpuf_basis_crps]

	for i in range(len(hpuf_accuracy_avg)):
		if hpuf_accuracy_avg[i] >= .95:
			hpuf_accuracy_avg[i+1:] = [None for x in hpuf_accuracy_avg[i+1:]]
			break

	m = np.linspace(1, 64, num=64)


	fig, ax2 = plt.subplots()
	for i in range(len(hlpuf_basis_accuracy_avg)):
		hlpuf_prob_m = [hlpuf_basis_accuracy_avg[i]**j for j in m]

		if i in [1,2,3,5,8,10,19]:
			ax2.plot(m, hlpuf_prob_m, label = '%.2g'%hlpuf_basis_crps[i])
	ax2.title.set_text('n=128, k=4')
	ax2.set_xlabel("Size of response m (qubits)", fontsize=12)
	ax2.set_ylabel("HLPUF Success probability of guessing (x100%)", fontsize=12)
	ax2.legend(loc='lower right')
	ax2.get_legend().set_title("CRPs")
	
	plt.show()
