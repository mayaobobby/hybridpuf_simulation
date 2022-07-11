import numpy as np 
import matplotlib.pyplot as plt

#accuracy_cpuf_2 = np.load('../hybridpuf_simulation/Simulation_pypuf/data/accuracy_cpuf_single_apuf_32_0.53_5.npy')
##crps = np.load('../hybridpuf_simulation/Simulation_pypuf/data/crps_single_apuf_32_5.npy')
#accuracy_hpuf_2 = np.load('../hybridpuf_simulation/Simulation_pypuf/data/accuracy_hpuf_single_apuf_32_0.53_bit_5.npy')

n_size = 32
k = 5
plt.title('n='+str(n_size)+',k='+str(k), fontsize=15)
repeat_experiment = 60
crps = np.load('./crps_xorpuf_n'+str(n_size)+'k'+str(k)+'_MUB8_rep'+str(repeat_experiment)+'.npy')
accuracy_c = np.load('./classical_xorpuf_accuracy_n'+str(n_size)+'k'+str(k)+'_MUB8_rep'+str(repeat_experiment)+'.npy')
length = len(accuracy_c)
plt.plot(crps[0:length], accuracy_c, color = 'black',label='cpuf')
plt.plot(crps[length-1:length+2],[accuracy_c[-1],accuracy_c[-1],accuracy_c[-1]],color = 'black', linestyle = 'dashed')
count = 0
max_prev = 0
colors = ['green','blue','red']
for i in range(3):
    accuracy_h = np.load('./hybrid_xorpuf_accuracy_bit_'+str(i)+'_n'+str(n_size)+'k'+str(k)+'_MUB8_rep'+str(repeat_experiment)+'.npy')
    if(i > 0):
        plt.vlines(crps[count],accuracy_h[0],1,linestyles='dotted',colors = colors[i],label = "bit "+ str(i) + " learning start")
    length = len(accuracy_h)
    plt.plot(crps[count:count+length], accuracy_h,color = colors[i],label='hpuf:bit '+str(i))
    count+=length
    if(i < 2):
        plt.plot(crps[count-1:count+2],[accuracy_h[-1],accuracy_h[-1],accuracy_h[-1]],color = colors[i], linestyle = 'dashed')
    
with open('./d8n32k5.txt', 'a') as f:
		print('{', end='', file=f)
		for i in range(len(accuracy_c)):
			
			print('{%i,%f}, ' % (crps[i], accuracy_c[i]), end='', file=f)
			
		print('}', file=f)

		accuracy_h_0 = np.load('./hybrid_xorpuf_accuracy_bit_0_n'+str(n_size)+'k'+str(k)+'_MUB8_rep'+str(repeat_experiment)+'.npy')
		accuracy_h_1 = np.load('./hybrid_xorpuf_accuracy_bit_1_n'+str(n_size)+'k'+str(k)+'_MUB8_rep'+str(repeat_experiment)+'.npy')
		accuracy_h_2 = np.load('./hybrid_xorpuf_accuracy_bit_2_n'+str(n_size)+'k'+str(k)+'_MUB8_rep'+str(repeat_experiment)+'.npy')

		mu, sigma = 0.5, 0.01 # mean and standard deviation
		s = list(np.random.normal(mu, sigma, len(accuracy_h_0)+len(accuracy_h_1)))
		hlpuf_accuracy_final = s+accuracy_h_2.flatten().tolist()
		print(hlpuf_accuracy_final)

		print('{', end='', file=f)
		for i in range(len(hlpuf_accuracy_final)):
			
			print('{%i,%f}, ' % (crps[i], hlpuf_accuracy_final[i]), end='', file=f)
			
		print('}', file=f) 


plt.xlabel('Number of CRPs')
plt.ylabel('Accuracy (x100%)')
plt.legend(loc='lower right')
plt.show()