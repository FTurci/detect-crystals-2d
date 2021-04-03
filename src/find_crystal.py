import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from scipy import ndimage
from sklearn.mixture import GaussianMixture

# data= np.roll(np.load("../simulate-data/img/image.npy"),(100,0,0))
# kernel= np.load("../simulate-data/img/kernel.npy")



# for the experiments, try multiple rotations of the kernel
convolutions = []

data= np.load("../exp-data/0.524squ.npy")
kernel= np.load("../exp-data/kernel.npy")

nrotations = 1
for angle in np.linspace(0,360,nrotations):
    k = ndimage.rotate(kernel,angle, mode="mirror")
    conv = signal.fftconvolve(data, k, mode='same')
    convolutions.append(conv)


convolutions = np.array(convolutions)

# from every rotation, pick only the largest contribution
poolmax = convolutions.max(axis=0)

# remove a border of thinkess kernel/2
b = int(kernel.shape[0]/2)
without_border = poolmax[b:-b, b:-b]
# plot the intensity histogram
plt.hist(without_border.ravel(), bins=100)
# plt.figure()
# plt.matshow(without_border)
# plt.show()
# plt.figure()

#gaussian mixture model: decompose the intesnity histogram into 3 gaussians: liquid, interface,crystal
X =np.empty(  (len(without_border.ravel()),1) )
X[:,0] = without_border.ravel()
gm = GaussianMixture(n_components=3, random_state=0).fit(X)
mean = gm.means_
#  use the model to label the image
u = gm.predict(X)

crystal_label_1 = u[X[:,0].argmax()]
crystal_label_2 = u[u!=crystal_label_1][X[u!=crystal_label_1,0].argmax()]
crystal_label_3 = u[(u!=crystal_label_1)*(u!=crystal_label_2)][X[(u!=crystal_label_1)*(u!=crystal_label_2),0].argmax()]

labelled = u.reshape(without_border.shape)
plt.figure(figsize=(6,6))
plt.pcolormesh(data[b:-b,b:-b], cmap = plt.cm.hot)

plt.contourf((labelled==crystal_label_1)+(labelled==crystal_label_2), [.5, 1.5], alpha=0.25)#, alpha=0.03, cmap=plt.cm.binary)
plt.contour((labelled==crystal_label_1)+(labelled==crystal_label_2), [.5], colors="k")

plt.axis('equal')
plt.xlim(0, without_border.shape[1])
plt.savefig("output.png")
plt.show()