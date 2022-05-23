import matplotlib.pyplot as plt
import numpy as np


p_guess = [0.5,0.55,0.6,0.7]
repeat_experiment = 30
n = 32
k = 5

plt.title('n = 32, k = 5')
plt.xlabel("Number of CRPs", fontsize=12)
plt.ylabel("Accuracy (x100%)", fontsize=12)

crps = np.load('./xorpuf'+str(k)+'_n'+str(n)+'_reps'+str(repeat_experiment)+'_crps.npy')
for i in range(len(p_guess)):
    accuracy_hpuf = np.load('./xorpuf'+str(k)+'_n'+str(n)+'_p'+str(p_guess[i])+'_reps'+str(repeat_experiment)+'_accuracy.npy')
    plt.plot(crps,accuracy_hpuf, label = 'p_guess = '+str(p_guess[i]))
plt.legend()
plt.show()