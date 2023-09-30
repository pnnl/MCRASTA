import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0.1, 1.2, 1000)
nd = 1000 * (50 / (101 * 412))
y = np.random.lognormal(2.5, 0.8, len(x))
y2 = np.random.lognormal(np.log(50), 0.8, len(x))
re_d1 = min(y) * (101 * 412) / 1000
re_d2 = max(y) * (101 * 412) / 1000

# a = pm.LogNormal('a', mu=np.log(1000 * 0.003), sigma=0.8)
# b = pm.LogNormal('b', mu=np.log(1000 * 0.003), sigma=0.8)

# for Dc: np.log(50), 0.8
# Dc_nd = Dc / (time_total * vref)
x = np.linspace(0, 1, 100)
mu0 = sp.stats.lognorm([0.25], loc=np.log(0.5))
#
# plt.plot(mu0.pdf(x))
# plt.show()

    # .LogNormal('Dc_nd', mu=np.log(1000 * 0.0012), sigma=0.8)
# mu0 = pm.LogNormal('mu0', mu=0.5, sigma=0.25)


plt.hist(y)
plt.show()

# ln = sp.stats.lognorm.pdf(y)
