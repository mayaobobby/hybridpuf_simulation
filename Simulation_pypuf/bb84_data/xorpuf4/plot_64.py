import numpy as np 
import matplotlib.pyplot as plt
import math
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

			crps = [x*coe for x in crps]

		elif puf_type == 'HLPUF':
			for ln in fi:
				if ln.startswith("CRPs:"):
					value = float(ln[5:])
					crps.append(value)
				if ln.startswith("HPUF Mean:"):
					value = float(ln[10:])
					accuracy_avg.append(value)

		else:
			pass

	return crps, accuracy_avg


def accuracy_plot_64bits(cpuf_crps, cpuf_accuracy_avg, hpuf_crps, hpuf_accuracy_avg, hlpuf_bit_crps, hlpuf_basis_crps, hlpuf_bit_accuracy_avg, hlpuf_basis_accuracy_avg):
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


	fig = plt.figure()
	plt.title('n=64, k=4',fontsize=15)
	plt.plot(cpuf_crps, cpuf_accuracy_avg, label = 'CPUF', color='blue')
	plt.plot(hlpuf_bit_crps, hlpuf_bit_accuracy_avg, label = 'HLPUF_adaptive:bit', linestyle='dashed', color='orange')
	plt.plot(hlpuf_basis_crps, hlpuf_basis_accuracy_avg, label = 'HLPUF_adaptive:basis', color='green')
	plt.plot(hpuf_crps, hpuf_accuracy_avg, label = 'HPUF_adaptive', color='red')
	plt.vlines(hlpuf_bit_crps[count], hlpuf_basis_accuracy_avg[0], hlpuf_bit_accuracy_avg[count], linestyle='dotted', label='Basis learning starts')
	plt.xlabel("Number of CRPs", fontsize=12)
	plt.ylabel("${p_{forge}^{quantum}}$ (x100%)", fontsize=12)
	plt.legend(loc='lower right')


if __name__ == '__main__':	

	cpuf_filename = "c_summary_xorpuf4_64.txt"
	hpuf_filename = "h_summary_xorpuf4_64.txt"
	hlpuf_filename = "hl_summary_xorpuf4_64.txt"


	cpuf_crps, cpuf_accuracy_avg = readline_from_summary(cpuf_filename, 'CPUF')
	hpuf_crps, hpuf_accuracy_avg = readline_from_summary(hpuf_filename, 'HPUF')
	hlpuf_crps, hlpuf_accuracy_avg = readline_from_summary(hlpuf_filename, 'HLPUF')
	hlpuf_bit_crps, hlpuf_basis_crps =  hlpuf_crps[:20], hlpuf_crps[20:]
	hlpuf_bit_accuracy_avg, hlpuf_basis_accuracy_avg = hlpuf_accuracy_avg[:20], hlpuf_accuracy_avg[20:]

	np.save('./c_xorpuf_n64k4_bb84_crps.npy', cpuf_crps)
	np.save('./c_xorpuf_n64k4_bb84_accuracy.npy', cpuf_accuracy_avg)

	np.save('./h_xorpuf_n64k4_bb84_crps.npy', hpuf_crps)
	np.save('./h_xorpuf_n64k4_bb84_accuracy.npy', hpuf_accuracy_avg)

	np.save('./hl_xorpuf_bit_n64k4_bb84_crps.npy', hlpuf_bit_crps)
	np.save('./hl_xorpuf_bit_n64k4_bb84_accuracy.npy', hlpuf_bit_accuracy_avg)

	np.save('./hl_xorpuf_basis_n64k4_bb84_crps.npy', hlpuf_basis_crps)
	np.save('./hl_xorpuf_basis_n64k4_bb84_accuracy.npy', hlpuf_basis_accuracy_avg)



	accuracy_plot_64bits(cpuf_crps, cpuf_accuracy_avg, hpuf_crps, hpuf_accuracy_avg, hlpuf_bit_crps, hlpuf_basis_crps, hlpuf_bit_accuracy_avg, hlpuf_basis_accuracy_avg)
	
	plt.show()
	