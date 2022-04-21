# coding = utf-8
import random
import numpy as np

def hybrid_flipping(value_original, success_prob):
	value_updated = value_original

	value_p = random.random()

	if value_p >= success_prob:
		value_updated = -value_original
	else:
		pass

	return value_updated
