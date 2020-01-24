import math
import numpy as np
d = np.array([0,1,0,2])
a = np.array([1,1,0,0])
print(np.where(d >1, [-1], d))