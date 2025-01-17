import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import pandas as pd
import os.path
import time
import scipy as sc
import sympy as sp
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import rotate
rho = 1.2
Lspan = 35e-2
D = 5e-2
Sf = Lspan  * D
scale_force = 1/241  # de gramo a newton  


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
        self.img2 = 1
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

class caso_drag():
    def __init__(self,lista_files):
        lista_files_temp  =  np.zeros_like(lista_files) 
        lista_files_tempn = np.zeros((len(lista_files),1))
        volt = np.zeros((len(lista_files),1))
        for i,filei in enumerate(lista_files):
            lista_files_temp[i],lista_files_tempn[i]  = get_modified_date(filei)
            volt[i] = float(filei.split('V')[-1].split('_')[0])
        dict_files1 = {'file': lista_files}
        data_files= pd.DataFrame.from_dict(dict_files1)  
        data_files['time creation'] = lista_files_temp
        data_files['time_ind'] = lista_files_tempn
        data_files['volt'] = volt
        data_files['velocity'] = veloc_tunel(volt)
        index_casos = ~data_files['volt'].isin([0.0])
        index_casos_0 =  data_files['volt'].isin([0.0])
        index_file_ref = np.zeros((len(lista_files),1)) 
        file_ref = np.zeros_like(lista_files)
        data_files['index_file_ref'] = index_file_ref
        data_files['file_ref'] = file_ref
        data_files['FD_m'] = index_file_ref*0.0
        lista_files_0 = data_files[index_casos_0]['time_ind'].to_numpy()
        for i, casoi in enumerate(data_files.itertuples()):
            filei = casoi.file
            index_file_ref_i = np.argmin(np.abs(casoi.time_ind-lista_files_0))
            data_files.at[i,'index_file_ref'] = index_file_ref_i
            filei_ref = data_files['file'][index_casos_0].to_numpy()[index_file_ref_i]
            data_files.at[i,'file_ref'] = filei_ref
            FD_raw = read_cd_csv(filei)
            FD_raw_ref = read_cd_csv(filei_ref)
            data_files.at[i,'FD_m'] = (FD_raw -FD_raw_ref)*scale_force
        self.df = data_files
        self.index_casos = index_casos
        self.index_casos0 = index_casos_0
    def fft_filei(self,filei):
        FD_raw = pd.read_csv(filei).iloc[:,3].to_numpy()
        FF_FD = np.fft.fft(FD_raw)
        freq = np.fft.fftfreq(len(FD_raw),1/80)
        self.fft_freq = freq
        self.fft_FD = FF_FD
    def freq_strouhal(self):
        data_files = self.df[self.index_casos]
        f_peaks = []
        for i, casoi in enumerate(data_files.itertuples()):
            filei = casoi.file
            self.fft_filei(filei)

            #FD_raw = pd.read_csv(filei).iloc[:,3].to_numpy()
            FF_FD = self.fft_FD
            freq = self.fft_freq
            
            filei_im = filei.replace('medidas','medidas_im')
            dir_im = filei_im.split('/')[0]
            if os.path.isdir(dir_im) == False:
                os.mkdir(os.getcwd()+'/'+dir_im)

            #x1s  = find_peaks(np.abs(FF_FD),prominence=1000)
            #index_peak = np.nonzero(freq[x1s[0]]>10)[0]
            #f_peak = freq[x1s[0]][index_peak]

            f_peak = np.array([])
            prom_0 = 1e4
            while f_peak.size<1:
                x1s  = find_peaks(np.abs(FF_FD),prominence=prom_0)
                index_peak = np.nonzero(freq[x1s[0]]>10)[0]
                f_peak = freq[x1s[0]][index_peak]
                prom_0 = prom_0*.75
            f_peaks.append(f_peak)

            if os.path.isfile(filei_im+'.png') == False:
            
                fig,ax = plt.subplots()
                ax.semilogy(freq,np.abs(FF_FD))

                ax.plot(f_peak,np.abs(FF_FD)[x1s[0][index_peak]],'ro')
                ax.set_xlim([1,40])
                ax.grid()

                plt.savefig(filei_im+'.png')
                plt.close(fig)

        self.strouhal_freq = f_peaks
        
        
        
        
    

def veloc_tunel(V):
    tunel_v = pd.read_csv('tunel_viento.csv',header=None)
    deltaV, U = tunel_v.to_numpy().T
    p1 = np.polyfit(deltaV,U,1)
    UdeltaV = np.poly1d(p1)
    if np.isscalar(V)==False:
        velocidad = np.zeros_like(V)
        for i,Vi in enumerate(V):
            velocidad[i] = UdeltaV(Vi)
            if Vi == 0: 
                velocidad[i] = 0
        
    elif V==0:
        velocidad = 0
    else:
        velocidad = UdeltaV(V)
    return velocidad

def get_modified_date(file_name):
    file_path=f"./{file_name}"
    # more info about os.path.getmtime function here: 
    # https://docs.python.org/3/library/os.path.html#os.path.getmtime
    modified_time = os.path.getmtime(file_path)
    return time.ctime(modified_time), modified_time


x, y, z = sp.symbols('x y z')
eq1 = sp.diff(sp.cos(x)*sp.cosh(x)+1, x)
def fun_modn (x):
    return np.cosh(x)*np.cos(x)+1
    
callable_fct = sp.lambdify(x, eq1)

x_s = np.linspace(1,15,200)
y_s = np.zeros_like(x_s)
for i,xi in enumerate(x_s):
    y_s[i] = fun_modn(xi)
signo_s = np.sign(y_s)
diff_signo_s = np.diff(signo_s)
x0s = x_s[np.nonzero(diff_signo_s)]

BnL = np.zeros_like(x0s)
for i,x0i in enumerate(x0s):
    BnL[i] = sc.optimize.fsolve(fun_modn,x0i,fprime=callable_fct)[0]
    
def beta_nL (i):
    return sc.optimize.fsolve(fun_modn,x0s[i],fprime=callable_fct)[0]

def freq_elast (betaL,L,B,rho,th):
    beta = betaL/L
    mu = rho*th*1e-6
    f_n = beta**2*(B/mu)**0.5/2/np.pi
    return f_n
def B_flexion (lstuart,rho,th):
    g = 9.8
    self.B =  ((lstuart*1e-3*1.103)**3)*rho*th*1e-6*g 

def read_cd_csv(file):
    Cd_t = pd.read_csv(file).iloc[:,3].to_numpy()
    return Cd_t.mean()


### la rotacion de la imagen es desde el centro de coordenadas
### asi podemos recuperar la rotacion de puntos xy
def rot(image, xy, angle):
    im_rot = rotate(image,angle)
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
    org = xy-org_center
    a = np.deg2rad(angle)
    new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
            -org[0]*np.sin(a) + org[1]*np.cos(a) ])
    return im_rot, new+rot_center
def rot_point(image, xy, angle):
    im_rot = rotate(image,angle)
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
    org = xy-org_center
    a = np.deg2rad(angle)
    new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
            -org[0]*np.sin(a) + org[1]*np.cos(a) ])
    return new+rot_center
