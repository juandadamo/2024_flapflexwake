
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
lista_casos = [s for s in lista_cam if "_v0_" not in s]
fig,ax = plt.subplots()

for i,filei in enumerate(lista_casos[:1]):
    casoi = caso_fast_cam(filei)
    img = tif.imread(casoi.files[0])
    im1 = ax.imshow(img)
    x0 = plt.ginput(-1,timeout=0)
    x0 = np.array(x0).astype('int32')
    casoi.coords_img = [x0[0][1],x0[2][1],x0[0][0],x0[1][0]]

    A = pd.DataFrame(data=x0,index=['p0','p1','p2'])

    ax.set_xlim([x0[0][0],x0[1][0]])
    ax.set_ylim([x0[0][1],x0[2][1]])
    img = img[x0[0][1]:x0[2][1],x0[0][0]:x0[1][0]]

    time0 = time.time()
    Nfiles = len(casoi.files)
    Nfiles = 100
    imgs = np.tile(np.zeros_like(img),[Nfiles,1,1])
    imgs_0 = np.zeros_like(imgs[:,:120,:140])
    imgs_1 = np.zeros_like(imgs[:,400:,:140])
    for j,file_img_j in enumerate(casoi.files[1:Nfiles]):
        timej = time.time() - time0
        im0 = tif.imread(file_img_j)
        im0 = im0[x0[0][1]:x0[2][1],x0[0][0]:x0[1][0]]
        #img += im0
        imgs[j] = im0
        imgs_0[j] = im0[:120,:140]
        imgs_1[j] = im0[400:,:140]
        time1 = time.time() - time0
        time_f_est = (time1-timej)*(Nfiles-j)
        print(f"\r Time: {time1:.1f} sec de {time_f_est:.1f}", end=" "*20)
    #img = img/Nfiles
    casoi.mean = imgs.mean(axis=0)
    casoi.std = imgs.std(axis=0)
    VEC0 = np.fft.fft(imgs_0,axis=0)
    VEC1 = np.fft.fft(imgs_1,axis=0)
    freqs = np.fft.fftfreq(Nfiles,1/fsampling)
    casoi.fftvec0 = VEC0
    casoi.fftvec0 = VEC0
    casoi.fftfreq = freqs
plt.close(fig)



#plt.imshow(imgs_1.mean(0))#+casoi.mean)

#Imgs = np.fft.fft(imgs_0,axis=0)

#fig,ax1 = plt.subplots(2,1)
#ax1[0].imshow(img)
#ax1[1].imshow(im0)
#plt.show()
