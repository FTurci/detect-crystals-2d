import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from scipy import ndimage
from sklearn.mixture import GaussianMixture

# data= np.roll(np.load("../simulate-data/img/image.npy"),(100,0,0))
# kernel= np.load("../simulate-data/img/kernel.npy")




# for the experimentsm, try multiple rotations of the kernel
convolutions = []

data= np.load("../exp-data/0.524squ.npy")
kernel= np.load("../exp-data/kernel.npy")

nrotations = 12
nrotations += 1
dtheta = 360/nrotations
for i in range(nrotations):
    angle = i*dtheta
    k = ndimage.rotate(kernel,angle)

    conv = signal.fftconvolve(data, k, mode='same')
    convolutions.append(conv)
# plt.pcolormesh(data)
# plt.colorbar()
# plt.savefig("input.png")
# plt.figure()

convolutions = np.array(convolutions)
# print( conv.shape, data.shape)


poolmax = convolutions.max(axis=0)

b = int(kernel.shape[0]/2)
without_border = poolmax[b:-b, b:-b]

plt.hist(without_border.ravel(), bins=100)
plt.figure()

X =np.empty(  (len(without_border.ravel()),1) )
X[:,0] = without_border.ravel()
gm = GaussianMixture(n_components=3, random_state=0).fit(X)
print(gm.means_)

u = gm.predict(X)
print(u.shape, np.prod(without_border.shape))
plt.pcolormesh(u.reshape(without_border.shape))




# plt.pcolormesh(poolmax>poolmax.mean()+poolmax.std())#+poolmax.std()*.5)#>conv.max()#alpha=0.5, cmap = plt.cm.bone_r)
plt.savefig("output.png")
plt.show()