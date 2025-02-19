
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os,glob, socket
import tifffile as tif
from skimage.filters import rank
from skimage.morphology import disk
import time

class caso_fast_cam:
    def __init__(self,path):
        self.path = path
        self.files = np.sort(glob.glob(path+'/*.tif'))
        self.coords_img = 1
        self.mean = 1
        self.std = 1
        self.fftvec0 = 1
        self.fftvec1 = 1
        self.freqs = 1
fsampling = 250
lista_cam = np.sort(glob.glob('/media/juan/JUAN_PMMH/LFD/photron/dshape*'))
lista_cam = np.sort(glob.glob('/media/juan/juan_win/photron/feb2025/*'))
lista_casos = [s for s in lista_cam if "_v0_" not in s]

lista_write =  np.sort(glob.glob('/home/juan/Documents/script_python/2024_flapflexwake/fastcam_coords/*.csv'))


dirout = '/home/juan/Documents/script_python/2024_flapflexwake/fastcam_coords/'


fig,ax = plt.subplots()
ncomienzo = len(lista_write)
ncomienzo = 0
nfin = -1
for i,filei in enumerate(lista_casos[ncomienzo:]):
    casoi = caso_fast_cam(filei)
    fileout = dirout + filei.split('/')[-1]+'.csv'
    if os.path.isfile(fileout)==False:

        if len(casoi.files)>0:

            img = tif.imread(casoi.files[0])
            im1 = ax.imshow(img)
            ax.set_title(filei.split('/')[-1][:24])
            #plt.show()
            x0 = plt.ginput(-1,timeout=0)
            x0 = np.array(x0).astype('int32')
            casoi.coords_img = [x0[0][1],x0[2][1],x0[0][0],x0[1][0]]

            A = pd.DataFrame(data=x0)#index=['p0','p1','p2'])

        A.to_csv(fileout)
        print(x0)


