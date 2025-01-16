
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
        self.fmax0 = 1
        self.fmax1 = 1
        self.escala0 = 1
        self.escala1 = 1
        self.delta_0 = 1
        self.delta_1 = 1
        self.angulo_cuerpo0 = 1
        self.angulo_cuerpo1 = 1
        self.deltas = 1
    def img_manual(self):
        self.escala0 = 1
        self.escala1 = 1
        self.delta_0 = 1
        self.delta_1 = 1
        self.angulo_cuerpo0 = 1
        self.angulo_cuerpo1 = 1
        self.deltas = 1
        self.complex_num_0 = 1
        self.complex_num_1 = 1
fsampling = 250
lista_cam = np.sort(glob.glob('/media/juan/JUAN_PMMH/LFD/photron/dshape*'))
lista_casos = [s for s in lista_cam if "_v0_" not in s]

lista_write =  np.sort(glob.glob('/home/juan/Documents/script_python/2024_flapflexwake/fastcam_coords/*.csv'))


dirout = '/home/juan/Documents/script_python/2024_flapflexwake/fastcam_stats/'
lista_objs = np.sort(glob.glob('/home/juan/Documents/script_python/2024_flapflexwake/fastcam_coords/*.pickle'))



#ax0,ax1 = axs
for i,filei in enumerate(lista_objs[:]):

    nameout = filei.replace('fastcam_coords','fastcam_stats')
    fileout_mean = filei.replace('fastcam_coords','fastcam_stats').replace('.pickle','_mean.png')
    fileout_std = filei.replace('fastcam_coords','fastcam_stats').replace('.pickle','_std.png')
    fileout_fft1 = filei.replace('fastcam_coords','fastcam_stats').replace('.pickle','_fft1.png')
    fileout_fft2 = filei.replace('fastcam_coords','fastcam_stats').replace('.pickle','_fft2.png')

    if os.path.isfile(fileout_mean) == False:
        fig0,ax0 = plt.subplots(1,1,figsize=(20,10))
        file_aux = open(filei, 'rb')
        data = pickle.load(file_aux)
        data.img_manual()
        ax0.set_title(filei.split('/')[-1][:24])

        # determinacion de desplazamientos
        plt.sca(ax0)
        im1 = ax0.imshow(data.mean)
        x0 = plt.ginput(-1,timeout=0)
        for x0i in x0:
            ax0.plot(x0i[0],x0i[1],'wo')
        plt.draw()

        plt.savefig(fileout_mean)


        im1.remove()
        aux1 = ax0.get_lines()
        for ai in aux1: ai.remove()

        #print(control_x)
        #raise ValueError()
        im2 = ax0.imshow(data.std)


        x1 = plt.ginput(-1,timeout=0)
        for x1i in x1:
            ax0.plot(x1i[0],x1i[1],'wo')
        plt.draw()
        plt.savefig(fileout_std)


        im2.remove()
        aux1 = ax0.get_lines()
        for ai in aux1: ai.remove()

        plt.close(fig0)
        fig1,ax1 = plt.subplots(1,1,figsize=(20,10))
        plt.sca(ax1)
        a,b,c,d = x0

        za = a[0]+1j*a[1]
        zb = b[0]+1j*b[1]
        zc = c[0]+1j*c[1]
        zd = d[0]+1j*d[1]

        data.complex_num_0 = [za,zb,zc,zd]
        Dpix = np.abs(zb-za)
        escala0= 5e-2/Dpix
        angulo_cuerpo0 = np.angle(zb-zc)
        delta_0 = np.sin(np.angle(za-zb))*np.abs(za-zb)
        delta_1 = np.sin(np.angle(zc-zd))*np.abs(zc-zd)

        a1,a2,b,c,d1,d2 = x1
        za1 = a1[0]+1j*a1[1]
        za2 = a2[0]+1j*a2[1]
        zb = b[0]+1j*b[1]
        zc = c[0]+1j*c[1]
        zd1 = d1[0]+1j*d1[1]
        zd2 = d2[0]+1j*d2[1]

        data.complex_num_1 = [za1,za2,zb,zc,zd1,zd2]
        Dpix2 = np.abs(zb-za)
        escala1= 5e-2/Dpix2
        angulo_cuerpo1 = np.angle(zb-zc)
        delta_s0_0 = np.sin(np.angle(za1-zb))*np.abs(za1-zb)
        delta_s0_1 = np.sin(np.angle(za2-zb))*np.abs(za2-zb)
        delta_s1_0 = np.sin(np.angle(zc-zd1))*np.abs(zc-zd1)
        delta_s1_1 = np.sin(np.angle(zc-zd2))*np.abs(zc-zd1)

        # determinacion de frecuencia
        lin, = ax1.semilogy(data.fftfreq,np.abs(data.fftvec0))
        ax1.set_xlim([0,125])

        x1 = plt.ginput(-1,timeout=0)
        plt.savefig(fileout_fft1)

        frec_max0 = x1
        lin.remove()

        lin, = ax1.semilogy(data.fftfreq,np.abs(data.fftvec1))
        ax1.set_xlim([0,125])
        x1 = plt.ginput(-1,timeout=0)
        plt.savefig(fileout_fft2)
        frec_max1 = x1
        lin.remove()

        plt.close(fig1)

        data.frec_max0,data.frec_max1, data.escala0, data.escala1= [frec_max0,frec_max1,escala0,escala1]
        data.angulo_cuerpo0,data.angulo_cuerpo1 = [angulo_cuerpo0,angulo_cuerpo1]

        data.delta_0,data.delta_1 = [delta_0,delta_1]
        data.deltas = [delta_s0_0,delta_s0_1,delta_s1_0,delta_s1_1]
        with open(nameout, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


    #plt.show()


    # A.th = np.round(float(filei.split('_t')[-1][:3])*1e-6,6)
    # A.L = np.round(float(filei.split('_L')[-1][:3])/100*D,4)
    # A.deltaV = float(filei.split('_v')[-1].split('_')[0])
    # A.veloc = veloc_tunel(A.deltaV )
    # A.drag_i = []
    # A.label_drag_i = []
    # A.file_caso = filei

#
#     casoi = caso_fast_cam(filei)
#     img = tif.imread(casoi.files[0])
#
#
#
#     filein = dirout + filei.split('/')[-1]+'.csv'
#     A = pd.read_csv(filein)
#     x0 = A.iloc[:,1:].to_numpy()
#     img = img[x0[0][1]:x0[2][1],x0[0][0]:x0[1][0]]
# #
#     time0 = time.time()
#     Nfiles = len(casoi.files)
#     #Nfiles = 100
#     imgs = np.tile(np.zeros_like(img),[Nfiles,1,1])
#     imgs_0 = np.zeros_like(imgs[:,:120,:140])
#     imgs_1 = np.zeros_like(imgs[:,400:,:140])
#     for j,file_img_j in enumerate(casoi.files[1:Nfiles]):
#         timej = time.time() - time0
#         im0 = tif.imread(file_img_j)
#         im0 = im0[x0[0][1]:x0[2][1],x0[0][0]:x0[1][0]]
#         #img += im0
#         imgs[j] = im0
#         imgs_0[j] = im0[:120,:140]
#         imgs_1[j] = im0[400:,:140]
#         time1 = time.time() - time0
#         time_f_est = (time1-timej)*(Nfiles-j)
#         print(f"\r Time: {time1:.1f} sec de {time_f_est:.1f}", end=" "*20)
# #     #img = img/Nfiles
#     casoi.mean = imgs.mean(axis=0)
#     casoi.std = imgs.std(axis=0)
#     del (imgs)
#     VEC0 = np.fft.fft(imgs_0,axis=0).sum(1).sum(1)
#     VEC1 = np.fft.fft(imgs_1,axis=0).sum(1).sum(1)
#     freqs = np.fft.fftfreq(Nfiles,1/fsampling)
#     casoi.fftvec0 = VEC0
#     casoi.fftvec1 = VEC1
#     casoi.fftfreq = freqs
#     nameout = filein.replace('.csv','.pickle')
#     with open(nameout, 'wb') as handle:
#         pickle.dump(casoi, handle, protocol=pickle.HIGHEST_PROTOCOL)
# plt.close(fig)
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

plt.close('all')
