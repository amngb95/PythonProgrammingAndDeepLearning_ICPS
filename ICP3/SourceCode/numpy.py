import numpy as np
x = np.random.randint(1, 20, size=15)
print("Original vector:")
print(x)
#x[x.argmax()] = 0
x[np.where(x==np.max(x))] = 0
print("Maximum value replaced by 0:")
print(x)