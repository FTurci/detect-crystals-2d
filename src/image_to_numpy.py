import numpy as np
import matplotlib.pyplot as plt
import skimage
import scipy
import copy
from scipy import ndimage
data = skimage.io.imread("../exp-data/0.524squ.png").astype(float)

plt.imshow(data)

sx, sy = 30,70
kernel = copy.deepcopy(data[sx:sx+128,sy:sy+128])

kernel -= kernel.mean()
kernel = ndimage.gaussian_filter(kernel, 1.0)
kernel [kernel<kernel.mean()]=0
plt.figure()
plt.pcolormesh(scipy.fft.fftshift(np.absolute(scipy.fft.fft2(kernel))))

np.save("../exp-data/0.524squ.npy", data)
np.save("../exp-data/kernel.npy", kernel)

plt.show()