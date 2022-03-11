# coding = utf-8
import numpy as np
import sys
import sympy

from fractions import Fraction
import fieldmath

import pypuf.simulation, pypuf.io
import pypuf.attack
import pypuf.metrics

from boolean_function import *


# Note: Put fieldmath.py together in the same repository

# Description: Learning a LBF(Linear Boolean Function) with GF(2) Elimination

if __name__ == '__main__':

	# Step 1: Initilazation of lbf and I/O pairs
	n = 8
	N = int(2e3)
	a = np.zeros((N,n+1))

	weight = 5
	seed_instance = int.from_bytes(os.urandom(4), "big")
	puf_lbf = linear_boolean_function(n, weight)
	challenges = (pypuf.io.random_inputs(n=n, N=N, seed=seed_instance)+1)/2
	responses = np.inner(puf_lbf, challenges)%2

	# augmented matrix
	a[:,:n] = challenges
	a[:,n] = np.transpose(responses)

	# Step 2: Linear independency check with RREF
	_, inds = sympy.Matrix(challenges).T.rref()
	a_invertible = np.zeros((n,n+1))
	for i in range(n):
		a_invertible[i,:] = a[inds[i],:]

	det = np.linalg.det(a_invertible[:,:n])
	print("Deteminant:",det)

	# Step 3: GF(2) Elimimnation with fieldmath library
	f = fieldmath.BinaryField(2)

	a_invertible_tolist = a_invertible.astype(int).tolist()
	mat = fieldmath.Matrix(len(a_invertible_tolist), len(a_invertible_tolist[0]), f)

	for i in range(mat.row_count()):
		for j in range(mat.column_count()):
			mat.set(i, j, a_invertible_tolist[i][j])	
	
	if (det != 0):
		# x = np.linalg.solve(challenges_square, responses_square)
		# print(x)
		puf_lbf_model = np.zeros(n)
		print("Linear Boolean Function Array:", puf_lbf)
		mat.reduced_row_echelon_form()
		for i in range(mat.row_count()):
			puf_lbf_model[i] = mat.get(i, n)

		print("GF(2) Elimimiation Array:", puf_lbf_model)
		result = np.array_equal(puf_lbf, puf_lbf_model)
		print(result)
		if result == 0:
			responses_model = np.zeros(n)
			for i in range(n):
				responses_model[i] = np.inner(puf_lbf_model, a_invertible[i,:n])%2
			print("Response model:", responses_model)
			print("Response actual:", a_invertible[:,n])
	else:
		sys.exit('Singular Matrix!')


