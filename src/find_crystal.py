import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


data= np.roll(np.load("../simulate-data/img/image.npy"),(100,0,0))
kernel= np.load("../simulate-data/img/kernel.npy")

conv = signal.fftconvolve(data, kernel, mode='same')
plt.pcolormesh(data)
plt.savefig("input.png")
plt.figure()
print( conv.shape, data.shape)
plt.pcolormesh(conv>conv.max()*0.4)#alpha=0.5, cmap = plt.cm.bone_r)
plt.savefig("output.png")
plt.show()