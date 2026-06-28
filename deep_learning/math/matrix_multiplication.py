import numpy as np
import torch



# Matrix Multiplication # 

# Rule - when is it valid?
#   - Matricies have rows and columns
#   - First we must consider the number of rows to columns between two matricies.
#   - Comparing the two matricies, the two inner columns have to match.
#   - Ex: [4,5] and [5,8] - the two inner numbers must match which in this case both are 5.
#   - The outer dimensions define the dimensions of the output product. Ex: [4,8]

# This example is not possible [2,7] and [5,2]
# This is though [5,7]T and [5,2]



A = np.random.randn(3,4)
B = np.random.randn(4,5)

# try some multiplications...
print(np.round(A@B,2)), print(' ')


# PyTorch
a = torch.randn(3,4)
b = torch.randn(4,5)

print(np.round(a@b,2))

