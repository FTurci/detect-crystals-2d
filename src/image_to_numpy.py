import numpy as np
import matplotlib.pyplot as plt
import skimage
import scipy
import copy
from scipy import ndimage

class Reader:

    def __init__(self,path = "../exp-data/0.524squ.png"):
        self.image = skimage.io.imread(path).astype(float)
        fig, ax = plt.subplots(1,2, figsize=(12,5))
        ax[0].matshow(self.image, cmap = plt.cm.binary)
        ax[1].hist(self.image.ravel(), bins=16)
        ax[1].set_xlabel('intensity')
      
        
    def get_kernel(self,si=30,sj=70, width=128, smoothing=1.0):
        """Extract a kernel, remove the average, compute the fourier transform and zero regions where the power spectrum is below some standard deviations from the average"""
        kernel = copy.deepcopy(self.image[si:si+width,sj:sj+width])
        self.width = width
        kernel -= kernel.mean()

        smooth_kernel = ndimage.gaussian_filter(kernel, smoothing)
        # smooth_kernel [smooth_kernel<(smooth_kernel.mean()-smooth_kernel.std()*zero_below)]=smooth_kernel.min()
        ftkernel = scipy.fft.fft2(smooth_kernel)

        
        fig, ax = plt.subplots(1,3,figsize=(12,5))
        ax[0].matshow(kernel, cmap = plt.cm.binary)
        ax[1].matshow(smooth_kernel, cmap = plt.cm.binary)
        ax[2].matshow(scipy.fft.fftshift(np.absolute(ftkernel)),cmap = plt.cm.binary)
        self.kernel = smooth_kernel
