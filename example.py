from optimal_transport.rust import sinkhorn
from time import time
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 100).astype(np.float32)
x0 = 0.2
x1 = 0.8
a = np.exp(-(x-x0)**2 / 0.01)
b = np.exp(-(x-x1)**2 / 0.01)
c = (x[np.newaxis,:] - x[:,np.newaxis])**2
reg = 0.01

t0 = time()
transport_plan = sinkhorn(a, b, c, reg)
print("Optimal transport computed in", time() - t0)

plt.subplot(211)
plt.plot(x, a)
plt.plot(x, b)
plt.subplot(212)
plt.imshow(transport_plan)
plt.show()
