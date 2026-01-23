from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

data_path = "/home/cafischer/Documents/HeAT/Data/Conv2d"
device = "cpu"

# E1: Increasing signal, nondistributed fixed kernel
data_e1 = np.genfromtxt(f"{data_path}/Timing_inc-dist-signal_fix-nondist-kernels_{device}.csv", delimiter=",")
ind = np.argsort(data_e1[0,:])
y1 = data_e1[1:,ind]

# E2: Increasing signal, nondistributed increasing kernel
data_e2 = np.genfromtxt(f"{data_path}/Timing_inc-dist-signal_inc-nondist-kernels_{device}.csv", delimiter=",")
ind = np.argsort(data_e2[0,:])
y2 = data_e2[1:,ind]

# E3: Increasing signal, distributed fixed kernel
data_e3 = np.genfromtxt(f"{data_path}/Timing_inc-dist-signal_fix-dist-kernels_{device}.csv", delimiter=",")
ind = np.argsort(data_e3[0,:])
y3 = data_e3[1:,ind]

# E4: Increasing signal, distributed increasing kernel
data_e4 = np.genfromtxt(f"{data_path}/Timing_inc-dist-signal_inc-dist-kernels_{device}.csv", delimiter=",")
ind = np.argsort(data_e4[0,:])
y4 = data_e4[1:,ind]

x = data_e4[0,ind].T
print(x)
print(data_e2[:,ind].shape)

y = np.zeros((len(ind), 4,2))
for i in range(x.shape[0]):
    y[i, 0, 0] = np.mean(y1[:, i])
    y[i, 1, 0] = np.mean(y2[:, i])
    y[i, 2, 0] = np.mean(y3[:, i])
    y[i, 3, 0] = np.mean(y4[:, i])

    y[i, 0, 1] = np.std(y1[:, i])
    y[i, 1, 1] = np.std(y2[:, i])
    y[i, 2, 1] = np.std(y3[:, i])
    y[i, 3, 1] = np.std(y4[:, i])

plt.figure()
plt.plot(x,y[:,:,0])

plt.figure()
plt.plot(x,y[:,:,1])
plt.show()
