import numpy as np
import time

interval = np.array(range(1, 10))
naturals = np.array(range(1, 10))

slice = interval[:,]

start = time.time()

countt = 1

dividers = np.where((slice % naturals[:,] == 0) & (naturals[:,] != slice), 1, slice)
uniq, count = np.unique(dividers, return_counts=True)
if count[0] != 8:
    pass
else:
    print(dividers)


end = time.time()
print (round((end - start), 2), countt)
