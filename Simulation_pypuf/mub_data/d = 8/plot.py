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
plt.plot(crps[0:length], accuracy_c, color = 'blue',label='CPUF')
plt.plot(crps[length-1:length+2],[accuracy_c[-1],accuracy_c[-1],accuracy_c[-1]], color = 'blue', linestyle = 'dashed')
count = 0
max_prev = 0
colors = ['orange','magenta','green']
for i in range(3):
    accuracy_h = np.load('./hybrid_xorpuf_accuracy_bit_'+str(i)+'_n'+str(n_size)+'k'+str(k)+'_MUB8_rep'+str(repeat_experiment)+'.npy')
    if(i > 0):
        plt.vlines(crps[count],accuracy_h[0],1, linestyle='dotted', color = colors[i], label = "Bit "+ str(i) + " learning starts")
    length = len(accuracy_h)
    if (i < 2):
    	plt.plot(crps[count:count+length], accuracy_h, color = colors[i], label='HLPUF:bit '+str(i), linestyle = 'dashed')
    	count+=length
    	plt.plot(crps[count-1:count+2],[accuracy_h[-1],accuracy_h[-1],accuracy_h[-1]],color = colors[i], linestyle = 'dashed')
    else:
    	plt.plot(crps[count:count+length], accuracy_h, color = colors[i], label='HLPUF:bit '+str(i))
    	# count+=length
    # if(i < 2):
    #     plt.plot(crps[count-1:count+2],[accuracy_h[-1],accuracy_h[-1],accuracy_h[-1]],color = colors[i], linestyle = 'dashed')
    	

plt.xlabel('Number of CRPs')
plt.ylabel('Accuracy (x100%)')
plt.legend(loc='lower right')
plt.show()