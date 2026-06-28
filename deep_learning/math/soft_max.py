import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt



# !!! SOFT MAX !!! #

# Natural Exponent: e = 2.718 "This is a constant, an erational number like PI"
# when a set of numbers is put through a softmax function the sum of the number set will always be one.

# manual in numpy
z = [1,2,3]

# compute the soft max
num = np.exp(z) # natural exponent
den = np.sum(np.exp(z))
sigma = num / den

print(sigma)
print(np.sum(sigma))

print('random ints next')


# repeat with some random integers
Z = np.random.randint(-5,high=15,size=25)
print(Z)

# compute the soft max
num = np.exp(Z)
den = np.sum(num)
sigma = num / den

# compare
plt.plot(Z,sigma,'ko')
plt.show()




# PYTORCH

# create soft max
softmax = nn.Softmax(dim=0)

# then apply the data to that function
sigmaT = softmax(torch.Tensor(Z)) # must be a torch tensor

print(sigmaT)

