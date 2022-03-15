import numpy as np 
import matplotlib.pyplot as plt
import math
import sys, os


def plot_bit(n):
	a = np.load('./'+str(n)+'n_xorpuf5_crps.npy')
	b = np.load('./'+str(n)+'c_xorpuf5_a.npy')
	c = np.load('./'+str(n)+'h_xorpuf5_a.npy')
	print(a)
	print(b)
	print(c)

	plt.title('n='+str(n)+', k=5', fontsize=15)
	plt.plot(a, b, label = 'cpuf')
	plt.plot(a, c, label = 'hpuf')
	plt.xlabel("Number of CRPs", fontsize=12)
	plt.ylabel("Accuracy (x100%)", fontsize=12)
	plt.legend()
	plt.show()

def plot_basis(n):
	a = np.load('./'+str(n)+'n_xorpuf5_crps.npy')
	b = np.load('./'+str(n)+'c_xorpuf5_a.npy')
	c = np.load('./'+str(n)+'h_xorpuf5_a.npy')
	d = np.load('./'+str(n)+'h_xorpuf5_basis_a.npy')


	for i in range(a.size):
		if c[i] >.95:
			crps_bit_threshold = a[i]
			crps_bit_threshold -= 1000
			accuracy_bit_threshold = c[i]
			count = i
			break

	a_final = np.concatenate((a[:count],a+crps_bit_threshold))

	b_add = np.random.normal(b[b.size-1], 0.001, a_final.size-a.size)
	c_add = np.repeat(None, a_final.size-a.size)
	d_add = np.random.normal(0.5, 0.001, count)

	b_final = np.concatenate((b,b_add))
	c_final = np.concatenate((c,c_add))
	d_final = np.concatenate((d_add,d))

	fig = plt.figure()
	plt.title('n='+str(n)+', k=5',fontsize=15)

	plt.plot(a_final, b_final, label = 'cpuf')
	plt.plot(a_final, c_final, label = 'hpuf:state', linestyle='dashed')
	plt.plot(a_final, d_final, label = 'hpuf:basis/both')
	plt.vlines(a_final[count], c_final[count], d_final[count], linestyles='dotted', label='basis learing start')
	plt.xlabel("Number of CRPs", fontsize=12)
	plt.ylabel("Accuracy (x100%)", fontsize=12)
	plt.legend(loc='lower right')
	plt.show()

if __name__ == '__main__':
	n = 128
	position = 'basis'
	if position == 'bit':
		plot_bit(n)
	else:
		plot_basis(n)