import pypuf.simulation, pypuf.io
import pypuf.attack
import pypuf.metrics

import numpy as np
import math
import sys, os, random

from util import *
from apuf_attack import *
from mub_prob import *

if __name__ == '__main__':
    n_size = 32
    step = 10000
    num_crps = step
    k = 5
    repeat_experiment = 60
    p = mub_probabilities(4)

    accuracies = []
    accuracies_c = []
    num_samples = []
    accuracy = 1

    for i in range(2):
        seed_instance_bit = int.from_bytes(os.urandom(4), "big")
        puf_bit = pypuf.simulation.XORArbiterPUF(n=n_size, noisiness=0, seed=seed_instance_bit,k = k)
        accuracies = []

        threshold_crps = num_crps - step
        accuracy = 0
        while(accuracy < 0.95):    
            accuracy = 0
            for j in range(repeat_experiment):
                accuracy += instance_one_hybrid_apuf_attack(p[i], puf_bit, puf_bit, num_crps-threshold_crps, position = 'bit', num_bs = 1000, num_epochs = 100)
            accuracy/= repeat_experiment
            accuracies.append(accuracy)
            if i == 0:
                accuracy_c = 0
                for j in range(repeat_experiment):
                    accuracy_c+= instance_one_hybrid_apuf_attack(1, puf_bit, puf_bit, num_crps, position = 'bit', num_bs = 1000, num_epochs = 100)
                accuracy_c/=repeat_experiment
                accuracies_c.append(accuracy_c)
            num_samples.append(num_crps)
            num_crps += step 
            np.save('./mub_data/hybrid_xorpuf_accuracy_bit_'+str(i)+'_n'+str(n_size)+'k'+str(k)+'_MUB4_rep'+str(repeat_experiment)+'.npy', accuracies)
            if i == 0:
                np.save('./mub_data/classical_xorpuf_accuracy_n'+str(n_size)+'k'+str(k)+'_MUB4_rep'+str(repeat_experiment)+'.npy', accuracies_c)
    np.save('./mub_data/crps_xorpuf_n'+str(n_size)+'k'+str(k)+'_MUB4_rep'+str(repeat_experiment)+'.npy', num_samples)