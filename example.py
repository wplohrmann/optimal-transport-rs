from optimal_transport.rust import sinkhorn, greenkhorn, calculate_1D_ot
from time import time
import numpy as np
import matplotlib.pyplot as plt

xa = np.linspace(0, 1, 100).astype(np.float32)
xb = np.linspace(0, 1, 200).astype(np.float32)
x0 = 0.2
x1 = 0.8
a = np.exp(-(xa-x0)**2 / 0.01)
a /= a.sum()
b = np.exp(-(xb-x1)**2 / 0.01)
b /= b.sum()
c = (xb[np.newaxis,:] - xa[:,np.newaxis])**2
reg = 0.01

t0 = time()
transport_plan = sinkhorn(a, b, c, reg)
print("Sinkhorn computed in", time() - t0)

t0 = time()
transport_plan = greenkhorn(a, b, c, reg)
print("Greenkhorn computed in", time() - t0)

plt.subplot(211)
plt.plot(xa, a)
plt.plot(xa, transport_plan.sum(1), "x")
plt.plot(xb, transport_plan.sum(0), "x")
plt.plot(xb, b)
plt.subplot(212)
plt.imshow(transport_plan)
plt.show()

scale = 10000
a_ = np.round(a * scale).astype(np.int32)
b_ = np.round(b * scale).astype(np.int32)
c_ = np.round(c * scale).astype(np.int32)
cost, discrete_plan = calculate_1D_ot(a_, b_, c_)
plt.subplot(211)
plt.imshow(transport_plan)
plt.subplot(212)
plt.imshow(discrete_plan)
plt.show()
