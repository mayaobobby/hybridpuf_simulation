# coding = utf-8


import numpy as np 
import matplotlib.pyplot as plt
import math
import sys, os
from pathlib import Path

from xorpuf_attack import *
from apuf_attack import *




if __name__ == '__main__':
	Path("./data").mkdir(parents=True, exist_ok=True)
	
	repeat_experiment = 10
	size_challenge = 32
	level_noisiness = 0.1
	position = 'basis'

	# apuf comparison (call one function per run)
	comparison_crps_accuracy_single_apuf_plotting(size_challenge, level_noisiness, position, repeat_experiment)
	# comparison_crps_accuracy_two_apuf_plotting(size_challenge, level_noisiness, repeat_experiment)


	# # xorpuf comparison (call one function per run)
	num_apuf_xor = 4
	num_bs = 1000
	num_epochs = 100
	# comparison_crps_accuracy_single_xorpuf_plotting(size_challenge, num_apuf_xor, level_noisiness, num_bs, num_epochs, position, repeat_experiment)
	# comparison_crps_accuracy_two_xorpuf_plotting(size_challenge, num_apuf_xor, level_noisiness, num_bs, num_epochs, repeat_experiment)
