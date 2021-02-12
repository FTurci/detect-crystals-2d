import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2
import scipy
from scipy import ndimage
data= np.load("../simulate-data/img/image.npy")
# plt.pcolormesh(data)
# plt.show()
kernel = data[100:228, 100:228]
kernel -= kernel.mean()
kernel = ndimage.gaussian_filter(kernel, 0.5)
plt.pcolormesh(scipy.fft.fftshift(np.absolute(fft2(kernel))))

np.save("../simulate-data/img/kernel.npy", kernel)
plt.show()