import torch
import numpy as np

# scalar - 7 " a number"
# vector - on physical dimension
# matrix - a 2 dimensional array of numbers - EX: a spead sheet
# any thing high is called a tensor or aka a 3 dimensional matrix

# In PyTorch though all the above are called Tensors to them.



#---------------------------------
# HOW TO TURN REALITY INTO NUMBERS
#---------------------------------
# TYPES OF REALITY:
    # Continuous / Categorical
    # Representing categorical data --- Dummy-coding or One-hot encoding

# Dummy-coding
#   - Assigns two possible values
#   - Creates a single vector
#   - Dummy-coding is for just one feature

# One-Hot Encoding
#   - Multiple categorys
#   - For a multiple categorys



# Vector and Matrix Transpose
# EX: x = [1,0,2,5,-2]T 
# x transposed would not change the order of the numbers but rather the orientation, 
# from horizontal to vertical and vice versa.

#-------------------
# Example in numpy and pytorch
#-------------------

# create a vector
nv = np.array([[1,2,3,4]])
print(nv), print( ' ')

# transpose it
print(nv.T), print(' ') # This is only transposing for the print statement though

# This will transpose the code in memory
nvT = nv.T 
print(nvT)

# Multiple lines
nM = np.array([[1,2,3,4],
               [5,6,7,8]])
print('--- 2D array ---')
print(nM)
print(nM.T)


# ------------
# PyTorch
# ------------

# create a vector
tv = torch.tensor([[1,2,3,4]])
print(tv), print(' ')

# transpose it
print(tv.T), print(' ')


print(f'Variable Type of nv {type(nv)}')
print(f'Variable Type of nM {type(nM)}')
print(f'Variable Type of tv {type(tv)}')
