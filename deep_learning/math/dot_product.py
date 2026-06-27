import numpy as np
import torch

# DOT PRODUCT #

# !!! One of the most important math concepts in all of deep learning!
# The most common notation would be: a transpose b, or aTb.
# Element wise multiple and then add the sum.
# The dot product between two vectors will always be a single number.
# They also have to have the same dimensionality or vector length.

# The dot product of two vectors tells you how similar they are.
# This is literally how search engines, recommendation systems, and RAG work 
# -- find vectors with high dot products.


# !! You can however perform dot product on a 2D vector, This is done in CNN.

# Example:
# Matrix A
#
# [1 2 3]
# [4 5 6]
# [7 8 9]
#
# Matrix B
#
# [9 8 7]
# [6 5 4]
# [3 2 1]
#
# Dot product:
#
# (1*9) + (2*8) + (3*7)
# + (4*6) + (5*5) + (6*4)
# + (7*3) + (8*2) + (9*1)
#
# = 9 + 16 + 21
# + 24 + 25 + 24
# + 21 + 16 + 9
#
# = 165

# !! Again these matrixs are the same size and dimensions





# NUMPY EXAMPLE #

# create a vector 
nv1 = np.array([1,2,3,4])
nv2 = np.array([0,1,0,-1])

# dot product via function
print(np.dot(nv1,nv2))

# dot product via computation

print(np.sum(nv1*nv2))


# PyTorch #
tv1 = torch.tensor([1,2,3,4])
tv2 = torch.tensor([3,2,6,3])

print(torch.dot(tv1,tv2))

# The dot product is - A single number that reflects the commonalities between two objects
#  (vectors, matricies, tensors, signals, images).

