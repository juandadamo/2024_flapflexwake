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
import time,pickle,glob
rho = 1.2
Lspan = 35e-2
D = 5e-2
Sf = Lspan  * D
scale_force = 1/241  # de gramo a newton  

rho_50 = 1305
rho_75 = 1902
y1_50 = 37
y1_75 = 50
nu = 15e-6

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
    def wakeflex(self):
        self.t = float(self.path.split('_t')[1][:3])
        self.L = float(self.path.split('_L')[1][:3])/100
        self.volt = float(self.path.split('_v')[1].split('_')[0])
        self.U = veloc_tunel(self.volt)
        t = self.t
        L = self.L
        rho  = 1.2
        nu = 15e-6
        D = 5e-2
        Lspan = 35e-2
        Sf = Lspan*D

        scale_force = 1/241  # de gramo a newton  

        rho_50 = 1305 # valores en kg/m3
        rho_75 = 1902
        y1_50 = 37  #valores en mm
        y1_75 = 50 
        D = 5e-2

        if t == 50:
            self.y1 = y1_50
            self.rhom = rho_50
        elif t == 75:
            self.y1 = y1_75
            self.rhom = rho_75
        self.t = t
        self.L = L
        self.Lr = L*D
        self.B = B_flexion(self.y1,self.rhom,self.t)
        if self.L >=1.25:
            self.modo_elastico_i(1)
        elif np.logical_and(self.L==1,self.t==50):
            self.modo_elastico_i(1)  
        else:
            self.modo_elastico_i(0)
        
        #self.modo_elastico = 0
        self.BnL  = beta_nL(self.modo_elastico)
        self.freqn = freq_elast (self.BnL,self.Lr,self.B,self.rhom,self.t)

    def modo_elastico_i(self,i):
        self.modo_elastico = i
        self.BnL  = beta_nL(self.modo_elastico)
        self.freqn = freq_elast(self.BnL,self.Lr,self.B,self.rhom,self.t)
    def caso_drag(self):
        
        lista_files_total = np.sort(glob.glob(f'medidas/Dshape_e*')) 
        matching_v0 = np.array([s for s in lista_files_total if "deltaV0" in s])
        Fds,n_1,L_0,e_0 = np.tile(np.zeros((len(matching_v0))),[4,1])
        time_1= np.zeros_like(matching_v0)

        for i,filei_1 in enumerate(matching_v0):
            time_1[i], n_1[i] = get_modified_date(filei_1)
            Fds[i] = read_cd_csv(filei_1)
            L_0 [i] = float(filei_1.split('L')[1][:3])/100
            e_0 [i] = float(filei_1.split('_e')[1].split('_')[0])
        dict_time_0 = {'files':matching_v0,'times':time_1,'ntimes':n_1,'Val':Fds,'L':L_0,'e':e_0,
                       'clust':e_0*0,'val_m':e_0*0,'time_m':e_0*0}
        dftime0 = pd.DataFrame.from_dict(dict_time_0)  
        vals_t0 = dftime0['ntimes'].to_numpy()
        
        self.files_drag = np.sort(glob.glob(f'medidas/Dshape_e{int(self.t):02d}_L{int(self.L*100):03d}_deltaV{int(self.volt):02d}*'))
            
        FD0,FD_raw = np.tile(np.zeros(len(self.files_drag)),[2,1])
        for i,filei in enumerate(self.files_drag):
            FD_raw[i] = read_cd_csv(filei)
            time_str_i, time_filei = get_modified_date(filei)
            n_val = np.argmin(np.abs(time_filei-vals_t0))
            FD0[i] = dftime0['Val'][n_val]
        self.FD0,self.FD_raw = [FD0,FD_raw]
        
        self.FD = (self.FD_raw-self.FD0)*scale_force
    def caso_drag_fft(self):
        freq_s = []
        FF_FD_s = []
        for i,filei in enumerate(self.files_drag):
            FD_raw = pd.read_csv(filei).iloc[:,3].to_numpy()
            FF_FD = np.fft.fft(FD_raw)
            freq = np.fft.fftfreq(len(FD_raw),1/80)
            freq_s.append(freq)
            FF_FD_s.append(FF_FD)
        self.drag_freq_s = np.asarray(freq_s)
        self.drag_FF_FD_s = np.asarray(FF_FD_s)
    def caso_drag_strouhal(self):

        f_peaks = []
        for i, casoi in enumerate(self.files_drag):

            #FD_raw = pd.read_csv(filei).iloc[:,3].to_numpy()
            FF_FD = self.drag_FF_FD_s[i]
            freq = self.drag_freq_s[i]
            

            f_peak = np.array([])
            prom_0 = 1e4
            while f_peak.size<1:
                x1s  = find_peaks(np.abs(FF_FD),prominence=prom_0)
                index_peak = np.nonzero(freq[x1s[0]]>10)[0]
                f_peak = freq[x1s[0]][index_peak]
                prom_0 = prom_0*.75
            f_peaks.append(f_peak)



        self.drag_strouhal_freq = f_peaks

        
        
   
        
        
        
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
    B =  ((lstuart*1e-3*1.103)**3)*rho*th*1e-6*g 
    return B
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

class wakeflex():
    def __init__(self,t,L):
        rho  = 1.2
        nu = 15e-6
        D = 5e-2
        Lspan = 35e-2
        Sf = Lspan*D

        scale_force = 1/241  # de gramo a newton  

        rho_50 = 1305 # valores en kg/m3
        rho_75 = 1902
        y1_50 = 37  #valores en mm
        y1_75 = 50 
        D = 5e-2
        
        if t == 50:
            self.y1 = y1_50
            self.rhom = rho_50
        elif t == 75:
            self.y1 = y1_75
            self.rhom = rho_75
        self.t = t
        self.L = L
        self.Lr = L*D
        self.B = B_flexion(self.y1,self.rhom,self.t)
        
 
        self.BnL  = beta_nL(self.modo_elastico)
        self.freqn = freq_elast (self.BnL,self.Lr,self.B,self.rhom,self.t)
        
    def modo_elastico_i(self,i):
        self.modo_elastico = i
        self.BnL  = beta_nL(self.modo_elastico)
        self.freqn = freq_elast (self.BnL,self.Lr,self.B,self.rhom,self.t)
    


def plot_displacement(filei):
    file_aux = open(filei, 'rb')
    data = pickle.load(file_aux)
    fig,ax = plt.subplots(1,2)
    ax0,ax1 = ax
    ax0.contourf(data.std)


    ax1.contourf(data.std_rot)


    for i,xi in enumerate(data.complex_num_0):
        xi0,xi1 = [np.real(xi),np.imag(xi)]
        ax0.plot(xi0,xi1,'wo')
        xi2 = [np.real(data.complex_num_0_rot[i]),np.imag(data.complex_num_0_rot[i])]
        ax1.plot(xi2[0],xi2[1],'ys')
        data.complex_num_0_rot[i]  = xi2[0] + xi2[1]*1j
    for i,xi in enumerate(data.complex_num_1):
        xi0,xi1 = [np.real(xi),np.imag(xi)]
        ax0.plot(xi0,xi1,'wo',fillstyle='none')
        xi2 = [np.real(data.complex_num_1_rot[i]),np.imag(data.complex_num_1_rot[i])]
        ax1.plot(xi2[0],xi2[1],'rs')   
        data.complex_num_1_rot[i]  = xi2[0] + xi2[1]*1j  

def corrige_rotacion(lista_objs):
    
    for i,filei in enumerate(lista_objs):

        file_aux = open(filei, 'rb')
        data = pickle.load(file_aux)
        za1,za2,zb,zc,zd1,zd2 = data.complex_num_1
        za_m,zb_m,zc_m,zd_m = data.complex_num_0
        data.angulo_cuerpo1 = np.angle(zb-zc)
        #data.angulo_cuerpo1 = -np.pi/50+np.pi/2
        zrot = np.exp(1j*(np.pi/2-np.abs(data.angulo_cuerpo1)))
        angle = 90-np.abs(data.angulo_cuerpo1*90/np.pi*2)

        data_rot, (x1,y1) = rot(data.std, np.array([0,0]), angle)
        data.std_rot = data_rot
        data_rot, (x1,y1) = rot(data.mean, np.array([0,0]), angle)
        data.mean_rot = data_rot
        data.complex_num_0_rot = np.zeros_like(data.complex_num_0)
        data.complex_num_1_rot = np.zeros_like(data.complex_num_1)
        for i,xi in enumerate(data.complex_num_0):
            xi0,xi1 = [np.real(xi),np.imag(xi)]
            xi2 = rot_point(data.std,np.array([xi0,xi1]),angle)
            data.complex_num_0_rot[i]  = xi2[0] + xi2[1]*1j
        for i,xi in enumerate(data.complex_num_1):
            xi0,xi1 = [np.real(xi),np.imag(xi)]
            xi2 = rot_point(data.std,np.array([xi0,xi1]),angle)
            data.complex_num_1_rot[i]  = xi2[0] + xi2[1]*1j  
        a,b,c,d = data.complex_num_0_rot
        a1,a2,b_,c_,d1,d2 = data.complex_num_1_rot

        data.delta_0 = np.imag(a)-np.imag(b)
        data.delta_1 = np.imag(c)-np.imag(d)
        data.delta_0_rms = np.abs(np.imag(a1)-np.imag(a2))
        data.delta_1_rms = np.abs(np.imag(d1)-np.imag(d2))
        with open(filei, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)



def plot_displac(lista_objs,espesor=50,Largo=1.5):
    plt.close('all')
    matching_1 = [s for s in lista_objs if f"t0{espesor:0d}" in s]
    matching_caso = [s for s in matching_1 
                          if f"L{int(Largo*100):03d}" in s]
    volts = [float(s.split('_v')[1][:2]) for s in matching_caso]
    velocidad = veloc_tunel(volts)
    delta_0, delta_1,Re,delta_0_rms,delta_1_rms = np.zeros((5,len(matching_caso)))
    Uinf, B_s,Largo_r,fnatural,CD = np.zeros((5,len(matching_caso)))

    for i,filei in enumerate(matching_caso):
        file_aux = open(filei, 'rb')
        data = pickle.load(file_aux)
        data.wakeflex()
        delta_0[i],delta_1[i] = [data.delta_0,data.delta_1]
        delta_0_rms[i],delta_1_rms[i] = [data.delta_0_rms,data.delta_1_rms]
        Re[i] = data.U*D/nu
        Uinf[i] = data.U
        B_s [i] = data.B
        Largo_r [i] = data.Lr
        
    fig,ax = plt.subplots()
    lin0, = ax.plot(Re,delta_0*data.escala0/D,'s',fillstyle='none',markersize=15)
    lin1, = ax.plot(Re,delta_1*data.escala0/D,'o',fillstyle='none',markersize=15)
    ax.errorbar(Re,delta_0*data.escala0/D,delta_0_rms*data.escala0/D,
                linestyle='none',color=lin0.get_color(),capsize=8)
    ax.errorbar(Re,delta_1*data.escala0/D,delta_1_rms*data.escala0/D,
                linestyle='none',color=lin1.get_color(),capsize=8)
    ax.grid()
    ax.set_ylim(top=0.5,bottom=-0.1)
    

def get_dataframe(lista_objs,espesor=50,Largo=1.5):
    plt.close('all')
    matching_1 = [s for s in lista_objs if f"t0{espesor:0d}" in s]
    matching_caso = [s for s in matching_1 
                          if f"L{int(Largo*100):03d}" in s]
    volts = [float(s.split('_v')[1][:2]) for s in matching_caso]
    velocidad = veloc_tunel(volts)
    delta_0, delta_1,Re,delta_0_rms,delta_1_rms = np.zeros((5,len(matching_caso)))
    Uinf, B_s,Largo_r,fnatural,CD,espesor,fmax0,fmax1 = np.zeros((8,len(matching_caso)))
    
    

    for i,filei in enumerate(matching_caso):
        file_aux = open(filei, 'rb')
        data = pickle.load(file_aux)
        data.wakeflex()
        if data.L>=1.25:
            data.modo_elastico_i(1)
        elif np.logical_and(data.L==1,data.t==50):
            data.modo_elastico_i(1)         
        
        
        delta_0[i],delta_1[i] = [data.delta_0,data.delta_1]
        delta_0_rms[i],delta_1_rms[i] = [data.delta_0_rms,data.delta_1_rms]
        Re[i] = data.U*D/nu
        Uinf[i] = data.U
        B_s [i] = data.B
        Largo_r [i] = data.Lr
        espesor [i] = data.t
        fnatural [i] = data.freqn
        fmax0 [i] = data.frec_max0[0][0]
        fmax1 [i] = data.frec_max1[0][0]
        
        
    dict_df = {'espesor':espesor,'Largo':Largo_r,'$U_\inf$':Uinf,'Re':Re,
'B':B_s,'$\delta_0$':delta_0,'$\delta_1$':delta_1,'${\delta_0}^\prime$':delta_0_rms,'${\delta_1}^\prime$':delta_1_rms,'$f_n$':fnatural,'$f_{\max 0}$':fmax0,'$f_{\max 1}$':fmax1,'$C_D$':CD}
    out_df = pd.DataFrame.from_dict(dict_df)
    return out_df


class caso_drag_f():
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