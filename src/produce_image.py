import atooms
from atooms.trajectory import TrajectoryLAMMPS 
import numpy as np
import  matplotlib.pyplot as plt
from scipy import ndimage


with TrajectoryLAMMPS("../simulate-data/dump.atom") as tj:
    s = tj[7]
    cell = s.cell
    side = cell.side
    print(cell,side)
    dl = 0.25
    img3d , edges= np.histogramdd(s.dump('pos'), bins=(np.arange(-side[0]/2, side[0]/2+0.5,dl),np.arange(-side[1]/2, side[1]/2+0.5,dl),np.arange(-side[2]/2, side[2]/2+0.5,dl)) )
    image_clean = ndimage.gaussian_filter(img3d[3]+img3d[4], 1.2)
    noise = np.random.uniform(0,image_clean.max()*0.5, image_clean.shape)
    image = image_clean+noise

    plt.pcolormesh(image)
    plt.colorbar()
    np.save("../simulate-data/img/image.npy", image)
    plt.show()