# coding = utf-8
import random

def hybrid_flipping(value_original, dice):
	value_updated = value_original
	if dice == 2:
		value = random.randint(1, dice)
		# print('dice value = ', value)
		if value == 1:
			value_updated = -value_original
	else:
		pass

	return value_updated

