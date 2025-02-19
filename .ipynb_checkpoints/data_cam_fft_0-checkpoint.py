
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os,glob, socket
import tifffile as tif
from skimage.filters import rank
from skimage.morphology import disk
import time,pickle

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

lista_write =  np.sort(glob.glob('/home/juan/Documents/script_python/2024_flapflexwake/fastcam_coords/*.csv'))


dirout = '/home/juan/Documents/script_python/2024_flapflexwake/fastcam_coords/'


fig,ax = plt.subplots()
ncomienzo = 0
nfin = -1

for i,filei in enumerate(lista_casos[ncomienzo:nfin]):
    casoi = caso_fast_cam(filei)

    filein = dirout + filei.split('/')[-1]+'.csv'
    nameout = filein.replace('.csv','.pickle')
    if os.path.isfile(nameout) == False:
        img = tif.imread(casoi.files[0])
        A = pd.read_csv(filein)
        x0 = A.iloc[:,1:].to_numpy()
        img = img[x0[0][1]:x0[2][1],x0[0][0]:x0[1][0]]
    #
        time0 = time.time()
        Nfiles = len(casoi.files)

        imgs = np.tile(np.zeros_like(img),[Nfiles,1,1])
        imgs_0 = np.zeros_like(imgs[:,:120,:140])
        imgs_1 = np.zeros_like(imgs[:,400:,:140])
        for j,file_img_j in enumerate(casoi.files[1:Nfiles]):
            timej = time.time() - time0
            im0 = tif.imread(file_img_j)
            im0 = im0[x0[0][1]:x0[2][1],x0[0][0]:x0[1][0]]

            imgs[j] = im0
            imgs_0[j] = im0[:120,:140]
            imgs_1[j] = im0[400:,:140]
            time1 = time.time() - time0
            time_f_est = (time1-timej)*(Nfiles-j)
            print(f"\r Time: {time1:.1f} sec de {time_f_est:.1f}", end=" "*20)

        casoi.mean = imgs.mean(axis=0)
        casoi.std = imgs.std(axis=0)
        del (imgs)
        VEC0 = np.fft.fft(imgs_0,axis=0).sum(1).sum(1)
        VEC1 = np.fft.fft(imgs_1,axis=0).sum(1).sum(1)
        freqs = np.fft.fftfreq(Nfiles,1/fsampling)
        casoi.fftvec0 = VEC0
        casoi.fftvec1 = VEC1
        casoi.fftfreq = freqs
        nameout = filein.replace('.csv','.pickle')
        with open(nameout, 'wb') as handle:
            pickle.dump(casoi, handle, protocol=pickle.HIGHEST_PROTOCOL)

#
#
#
# #plt.imshow(imgs_1.mean(0))#+casoi.mean)
#
# #Imgs = np.fft.fft(imgs_0,axis=0)
#
# #fig,ax1 = plt.subplots(2,1)
#ax1[0].imshow(img)
#ax1[1].imshow(im0)
#plt.show()
#plt.close(fig)
print('fin')
