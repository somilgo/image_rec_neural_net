x = (1, 2, 3)
y = (4, 5, 6)

for x, y in zip(x, y):
	print x,y

import numpy as np 

pixels = [[1,2,3,4,5]]
g = np.array(pixels, dtype=float)
print g