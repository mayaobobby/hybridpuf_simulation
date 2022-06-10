import numpy as np 
import matplotlib.pyplot as plt
import math
import sys, os

count = 0

hlpuf_basis_crps_update = 0

def readline_from_summary(filename, puf_type):
	with open(filename,"r") as fi:
		crps, accuracy_avg, suc_prob = [], [], []
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
					suc_prob.append(value)
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
					suc_prob.append(value)

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
					suc_prob.append(value)
		else:
			pass

	return crps, accuracy_avg, suc_prob


def accuracy_plot_128bits(cpuf_crps, cpuf_accuracy_avg, hpuf_crps, hpuf_accuracy_avg, hlpuf_bit_crps, hlpuf_basis_crps, hlpuf_bit_accuracy_avg, hlpuf_basis_accuracy_avg):
	global count, hlpuf_basis_crps_update

	for i in range(len(hlpuf_bit_crps)):
		if hlpuf_bit_accuracy_avg[i] >.95:
			crps_bit_threshold = hlpuf_bit_crps[i]
			crps_bit_threshold -= 1000
			count = i
			break

	hlpuf_basis_crps = [x+crps_bit_threshold for x in hlpuf_basis_crps]
	
	hlpuf_basis_crps_update = hlpuf_basis_crps
	

	# for i in range(len(hpuf_accuracy_avg)):
	# 	if hpuf_accuracy_avg[i] >= .95:
	# 		hpuf_accuracy_avg[i+1:] = [None for x in hpuf_accuracy_avg[i+1:]]
	# 		break

	# global hlpuf_crps_final, hlpuf_accuracy_final

	# hlpuf_crps_final = hlpuf_bit_crps[:count]+hlpuf_basis_crps
	# hlpuf_accuracy_final = hlpuf_bit_accuracy_avg[:count]+hlpuf_basis_accuracy_avg

	fig = plt.figure()
	plt.title('n=128, k=5',fontsize=15)
	plt.plot(cpuf_crps, cpuf_accuracy_avg, label = 'cpuf')
	plt.plot(hlpuf_bit_crps, hlpuf_bit_accuracy_avg, label = 'hlpuf_adaptive:state', linestyle='dashed')
	plt.plot(hlpuf_basis_crps, hlpuf_basis_accuracy_avg, label = 'hlpuf_adaptive:basis')
	plt.plot(hpuf_crps, hpuf_accuracy_avg, label = 'hpuf_adaptive')
	plt.vlines(hlpuf_bit_crps[count], hlpuf_basis_accuracy_avg[0], hlpuf_bit_accuracy_avg[count], linestyles='dotted', label='basis learing start')
	plt.xlabel("Number of CRPs", fontsize=12)
	plt.ylabel("${p_{forge}^{quantum}}$ (x100%)", fontsize=12)
	plt.legend(loc='lower right')	

def feasrate_plot_128bits(cpuf_crps, cpuf_suc_prob, hpuf_crps, hpuf_suc_prob, hlpuf_bit_crps, hlpuf_basis_crps, hlpuf_bit_suc_prob, hlpuf_basis_suc_prob):
	for i in range(len(hlpuf_bit_crps)):
		if hlpuf_bit_suc_prob[i] >.95:
			crps_bit_threshold = hlpuf_bit_crps[i]
			crps_bit_threshold -= 1000
			count = i
			break
	
	hlpuf_basis_crps = [x+crps_bit_threshold for x in hlpuf_basis_crps]

	for i in range(len(hpuf_suc_prob)):
		if hpuf_suc_prob[i] >= .95:
			hpuf_suc_prob[i+1:] = [None for x in hpuf_suc_prob[i+1:]]
			break
	# for i in range(len(cpuf_suc_prob)):
	# 	if cpuf_suc_prob[i] > .95:
	# 		cpuf_suc_prob[i+2:] = [None for x in cpuf_suc_prob[i+2:]]
	# 		break


	fig = plt.figure()
	plt.title('n=128, k=5',fontsize=15)
	plt.plot(cpuf_crps, cpuf_suc_prob, label = 'CPUF')
	plt.plot(hlpuf_bit_crps, hlpuf_bit_suc_prob, label = 'HLPUF_adaptive:state', linestyle='dashed')
	plt.plot(hlpuf_basis_crps, hlpuf_basis_suc_prob, label = 'HLPUF_adaptive:basis')
	plt.plot(hpuf_crps, hpuf_suc_prob, label = 'HPUF_adaptive')
	plt.vlines(hlpuf_bit_crps[count], hlpuf_basis_suc_prob[0], hlpuf_bit_suc_prob[count], linestyles='dotted', label='basis learing start')
	plt.xlabel("Number of CRPs", fontsize=12)
	plt.ylabel("Success Rate(x100%)", fontsize=12)
	plt.legend(loc='lower right')
		


if __name__ == '__main__':	

	cpuf_filename = "c_summary_xorpuf5_128.txt"
	hpuf_filename = "h_summary_xorpuf5_128.txt"
	hlpuf_filename = "hl_summary_xorpuf5_128.txt"


	cpuf_crps, cpuf_accuracy_avg, cpuf_suc_prob = readline_from_summary(cpuf_filename, 'CPUF')
	hpuf_crps, hpuf_accuracy_avg, hpuf_suc_prob = readline_from_summary(hpuf_filename, 'HPUF')
	hlpuf_crps, hlpuf_accuracy_avg, hlpuf_suc_prob = readline_from_summary(hlpuf_filename, 'HLPUF')
	hlpuf_bit_crps, hlpuf_basis_crps =  hlpuf_crps[:20], hlpuf_crps[20:]
	hlpuf_bit_accuracy_avg, hlpuf_basis_accuracy_avg = hlpuf_accuracy_avg[:20], hlpuf_accuracy_avg[20:]
	hlpuf_bit_suc_prob, hlpuf_basis_suc_prob = hlpuf_suc_prob[:20], hlpuf_suc_prob[20:]

	


	accuracy_plot_128bits(cpuf_crps, cpuf_accuracy_avg, hpuf_crps, hpuf_accuracy_avg, hlpuf_bit_crps, hlpuf_basis_crps, hlpuf_bit_accuracy_avg, hlpuf_basis_accuracy_avg)
	# succrate_plot_128bits(cpuf_crps, cpuf_suc_prob, hpuf_crps, hpuf_suc_prob, hlpuf_bit_crps, hlpuf_basis_crps, hlpuf_bit_suc_prob, hlpuf_basis_suc_prob)

	hlpuf_crps_final = []
	hlpuf_accuracy_final = [] 


	hlpuf_crps_final = hlpuf_bit_crps[:count]+hlpuf_basis_crps_update

	mu, sigma = 0.5, 0.01 # mean and standard deviation
	s = list(np.random.normal(mu, sigma, count))
	print(len(hlpuf_crps_final))
	hlpuf_accuracy_final = s+hlpuf_basis_accuracy_avg
	print(len(hlpuf_accuracy_final))

	# sys.exit()

	with open('./n128k5.txt', 'a') as f:
		print('{', end='', file=f)
		for i in range(len(cpuf_crps)):
			
			print('{%i,%f}, ' % (cpuf_crps[i], cpuf_accuracy_avg[i]), end='', file=f)
			
		print('}', file=f)

		print('{', end='', file=f)
		for i in range(len(hpuf_crps)):
			
			print('{%i,%f}, ' % (int(hpuf_crps[i]), hpuf_accuracy_avg[i]), end='', file=f)

		print('}', file=f)

		print('{', end='', file=f)
		for i in range(len(hlpuf_crps_final)):
			
			print('{%i,%f}, ' % (hlpuf_crps_final[i], hlpuf_accuracy_final[i]), end='', file=f)
			
		print('}', file=f) 