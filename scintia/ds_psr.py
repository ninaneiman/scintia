import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os, sys, math, time
from scipy import interpolate
from astropy.time import Time
from astropy.visualization import quantity_support
from astropy import constants as const
import astropy
from matplotlib.colors import LogNorm

import matplotlib as mpl
from scipy.sparse.linalg import eigsh
from scipy.optimize import curve_fit
from . import load_data as ld
from . import ththmod as THTH
#np.seterr(divide='ignore', invalid='ignore')


def load_triple_spectrum(name='/mnt/scratch-lustre/gusinskaia/triple_system/5602579_AO_1400_ds.npz',factor=[1,1], tel='Undefined', psr='PSRJ0337+1715', pad_it=True, npad=3, mean0=True, wnoise=True):
    ''' load  data from npz file into Spec object using load_data module
    Takes:
    filenpz - name of the npz file to load data from
    factor - [#int, #int], where #int is always a factor of 2. 
    factor difined the number by which to downsample 2D ds data (see shrink_2 function)
    tel - Observatpry on which the data was taken (if known)
    psr - pulsar name of the object (if known)
    pad_it  - whether to pad the secondary spectra with zeros
    npad - (int) number of paded zeros 
    mean0 - set mean of the data to 0 (i.e. whether to subtract mean)
    wnoise - whether to load noise data (which is present or not)
    Returns:
    Spec object'''

    triple_ds=np.load(name)
    ds,a_t,a_f, ns=ld.load_triple(name, factor=factor, mean0=mean0, wnoise=wnoise)
    if tel == 'Undefined':
        spt_tel=name.split('_')
        if "AO" in spt_tel:
            tel="AO"
        if "WSRT" in spt_tel:
            tel="WSRT"
        if "GBT" in spt_tel:
            tel="GBT"
    if wnoise is False:
        ns_info='no noise'
    else:
        ns_info='w noise'
    my_spec=Spec(I=ds,t=a_t,f=a_f,stend=triple_ds['mjd'], nI=ns, tel=tel, psr=psr,pad_it=pad_it, npad=npad, ns_info=ns_info)
    return my_spec


##Function for making extent tuple for plotting
def ext_find(x,y):
    '''function from Daniels code'''
    dx=np.diff(x).mean()
    dy=np.diff(y).mean()
    ext=[(x[0]-dx/2).value,(x[-1]+dx/2).value,(y[0]-dy/2).value,(y[-1]+dy/2).value]
    return(ext)


def fun_shrink_ds(ds,t,f,factor=[1,1]):
    '''Downsamples ds, t, and f by ant integer factors'''
    new_ds=ld.shrink_any_2(ds, factor=factor)
    new_t=ld.shrink_any_1(t, factor[0])
    new_f=ld.shrink_any_1(f, factor[1])
    return new_ds, new_t, new_f


def fun_plot_ds_woaxis(ds, t, f, fig=plt.figure(figsize=(3,8), dpi=150), rect=[0.0,0.0,1.0,1.0], inpt='none'):
    '''function that plots ds inside a given axis. The purpose is to avoid wasting time setting parameters right.
    it takes:
    ds - dynamic spectrum (2D array)
    t, f - time and frequency axes (astropy quantity with units)
    fig - figure to which to add an axis
    rect - rectangular parameters of the axis'''

    ax=fig.add_axes(rect)
    vmin,vmax = np.percentile(ds,[1,99])
    ax.imshow(ds.T,extent=(0,(t[-1].value-t[0].value)/3600.,np.amin(f).value,np.amax(f).value),
           vmin=vmin, vmax=vmax, aspect='auto', origin='lower', interpolation=intp)
    ax.set_xlabel('Time (hr)')
    ax.set_ylabel('Frequency (%s)'%f.unit.to_string('latex'))

def fun_plot_ds(ds, t, f, new_fig=True, figsize=(3,8), dpi=150, vmin=None, vmax=None, cmap='viridis', lab_mfr=False, intp='none'):
    '''function that plots ds. The purpose is to avoid wasting time setting parameters right.
    it takes:
    ds - dynamic spectrum (2D array)
    t, f - time and frequency axes (astropy quantity with units)
    newfig - whether or not one wishes/needs to create new plt.figure()
    dpi - resolution of a newly created figure
    vmin, vmax - value limits of ds for plt.imshow()'''

    if new_fig is True:
    	plt.figure(figsize=figsize, dpi=dpi)
    if (vmin is None) and (vmax is None):
        vmin,vmax = np.percentile(ds,[1,99])
    plt.imshow(ds.T,extent=(0,(t[-1].to(u.hr).value-t[0].to(u.hr).value),np.amin(f).value,np.amax(f).value),
           vmin=vmin, vmax=vmax, aspect='auto', origin='lower', cmap=cmap, interpolation=intp)
    plt.xlabel('Time (hr)')
    if lab_mfr is True:
        plt.ylabel('%.2f (%s)'%(np.mean(f).value, f.unit.to_string('latex')))
    else:
        plt.ylabel('Frequency (%s)'%f.unit.to_string('latex'))

def fun_make_ss(ds, t, f, pad_it=True, npad=3):
    '''Makes secondary spectrun from dynamic spectrum using Daniels funtions
    Takes:
    ds - dynamic spectrum (2D array)
    t, f - time and frequency axes (astropy quantity with units)
    pad_it  - whether to pad the secondary spectra with zeros
    npad - (int) number of paded zeros
    Returns:
    Is - secodary specrum array
    tau, fd  - delay and dopler shift
    '''
    if pad_it is True:
        ds_pad=np.pad(ds.T,((0,npad*ds.T.shape[0]),(0,npad*ds.T.shape[1])),mode='constant',
                        constant_values=ds.T.mean())
        ds=ds_pad.T
        fd=THTH.fft_axis(t,u.mHz,npad)
        tau=THTH.fft_axis(f,u.us,npad)
        Is=np.fft.fftshift(np.fft.fft2(ds.T))
    else:
        fd=THTH.fft_axis(t,u.mHz)
        tau=THTH.fft_axis(f,u.us)
        Is=np.fft.fftshift(np.fft.fft2(ds.T))    
    return Is, tau, fd


def fun_plot_ss(Is, tau, fd, fd_lim=[-1.5, 1.5], tau_lim=[0.0,1.4], vmin=None, vmax=None,
           new_fig=True, figsize=(3,2), dpi=150, cb=True, cmap='viridis',
           plot_parabola=False, eta=1.0):
    ''' Plots secondary spectrum
    Takes:
    Is  - complex array- secodary specrum array
    tau, df - delay and dopler shift
    fd_lim, tau_lim - [float, float]  - limit of dopler shift and delay in plot
    vmin, vmax - value limits of ds for plt.imshow()

    new_fig - whether or not one wishes/needs to create new plt.figure()
    figsize - (float, float) - size of the newly created figure
    dpi - resolution of a newly created figure
    cb - whether or not to show the colorbar
    '''
    Is2=np.abs(Is**2)
    SS_ext=ext_find(fd,tau)
    if (vmin is None) or (vmax is None):
        vmin_pc,vmax_pc=np.percentile(Is2,[10,100-5e-2])
        #vmin=np.median(np.abs(Is[(Is!=0)])**2)*2
    if vmin is None:
        vmin=vmin_pc
    if vmax is None:
        vmax=vmax_pc
    if tau_lim is None:
        tau_lim=np.array([tau[0].value,tau[-1].value])
    if fd_lim is None:
        fd_lim=np.array([fd[0].value,fd[-1].value])
    if new_fig is True:
        fig=plt.figure(figsize=figsize, dpi=dpi, facecolor='w', edgecolor='k')
    plt.imshow(Is2,norm=LogNorm(vmax=vmax, vmin=vmin),aspect='auto',extent=SS_ext, origin='lower', cmap=cmap)
    if cb is True:
        plt.colorbar()

    plt.xlabel(fd.unit.to_string('latex'))
    plt.ylabel(tau.unit.to_string('latex'))
    plt.xlim(fd_lim[0], fd_lim[1])
    plt.ylim(tau_lim[0], tau_lim[1])
    if plot_parabola is True:
        plt.plot(fd,eta*(fd**2),'r',lw=1, ls='--')

def find_nearest(array, value):
    '''Finds index of the nearest to value number in an array
    Takes: 
    array - (np.array(dtype=float)) array with numbers
    value - (float) - number to which the find the index of the closes number in an array
    Returns:
    idx (int)''' 
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

#not sure this function is in use - could delete 
def find_edges(I_cross, fd_cross, tau_cross, fd_lim, tau_lim):
    fd_l=find_nearest(fd_cross, fd_lim[0])
    fd_r=find_nearest(fd_cross, fd_lim[1])
    tau_b=np.amax([find_nearest(tau_cross, tau_lim[0]),find_nearest(tau_cross, tau_lim[1])])-1
    tau_t=np.amin([find_nearest(tau_cross, tau_lim[0]),find_nearest(tau_cross, tau_lim[1])])+1
    return fd_l, fd_r, tau_b, tau_t


def crop_array(my_array, idx=None, values=None):
    '''Crops array in time and frequency (given by values of time and/or frequency) or by indexes of the array
    Takes:
    array -  (1D array), (astropy quantity with units)
    values - (astropy units quantites) [q,q] - min and max of values to cut the array with. 
    idx - [int,int] - index edges to crop array to
    Returns:
    res_array - cropped 1D array
    res_idx - indexes to which it was cropped
    '''
    if idx is not None and values is not None:
        raise ValueError("Both selection chosen! Choose one: based either on index or value!")
    elif values is not None:
        if (values[0].unit != my_array.unit) or (values[0].unit != my_array.unit):
            raise ValueError("Unit of an array and selection do not match!")
        res_i0=find_nearest(my_array.value, values[0].value)
        res_i1=find_nearest(my_array.value, values[1].value)
        res_idx=np.sort(np.array([res_i0,res_i1]))
    elif idx is not None:
        res_idx=np.sort(np.array([idx[0],idx[1]]))
    else:
        res_idx=np.array([0,len(my_array)])
    res_array=my_array[res_idx[0]:res_idx[1]]
    res_idx=[res_idx[0], res_idx[1]]
    return res_array, res_idx
    
def fun_select(ds,t,f, time_sel=None, freq_sel=None, freq_idx=None, time_idx=None, ns=None):
    '''Crops array in time and frequency (given by values of time and/or frequency) or by indexes of the array
    Takes:
    ds - dynamic spectrum (2D array)
    t, f -  time and frequency axes (astropy quantity with units)
    time_sel, freq_sel - (astropy units quantites) [q,q] - min and max of time and frequency to cut the array with. 
    time_idx, freq_idx - [int,int] - index edges in time and freq axes to crop array to
    ns - noise spectrum (if present)
    Returns:
    I_sel, t_sel, f_sel - cropped ds, time and frequency
    t_idx, f_idx - what?
    nI_sel - cropped ns
    '''
    if (freq_sel is not None) or (freq_idx is not None):
        f_sel, f_idx = crop_array(f, idx=freq_idx, values=freq_sel)
    else:
        f_idx=[0,len(f)]
        f_sel=f[f_idx[0]:f_idx[1]]
    if (time_sel is not None) or (time_idx is not None):
        t_sel, t_idx = crop_array(t, idx=time_idx, values=time_sel)
    else:
        t_idx=[0,len(t)]
        t_sel=t[t_idx[0]:t_idx[1]]
    I_sel = ds[t_idx[0]:t_idx[1],f_idx[0]:f_idx[1]]
    if ns is None:
        nI_sel=np.random.normal(size=np.shape(I_sel))*np.std(I_sel)/6
    else:
        nI_sel=ns[t_idx[0]:t_idx[1],f_idx[0]:f_idx[1]]
    return I_sel, t_sel, f_sel, t_idx, f_idx, nI_sel

def fun_interp(ds, mjd, f, t_ed, f_ed, t_len, f_len, ns=None):
    '''Interpolates ds to a new time and frequency axis
    Takes:
    ds - dynamic spectrum (2D array)
    mjd - time axis in mjd (not quantities!)
    f - frequency axis (astropy quantity with units)
    t_ed - (q -astropy units quantites) [q,q] - edges of the desired time axis to be interpolated to
    t_len - (int) - size of the desired time axis to be interpolated to 
    f_ed - (q -astropy units quantites) [q,q] - edges of the desired frequency axis to be interpolated to
    f_len - (int) - size of the desired frequency axis to be interpolated to 

    ns - noise spectrum (if present)
    Returns:
    I_new - interpolated ds
    t_sec_new - new time axis (q)
    f_new - new frequency axis
    t_int - new time axis (float)
    nI_new - interpolated ns
    '''
    fun=interpolate.RegularGridInterpolator((mjd,f), ds, method='linear')
    new_ntbin=(t_ed[1].value-t_ed[0].value)/(t_len)
    t_int = np.arange(t_ed[0].value,t_ed[1].value, new_ntbin)
    new_nfbin=(f_ed[1].value-f_ed[0].value)/(f_len-1)
    f_new = np.arange(f_ed[0].value,f_ed[1].value+new_nfbin, new_nfbin)

    new_t_grid, new_f_grid = np.meshgrid(t_int, f_new, indexing="ij")
    new_points = np.array([new_t_grid.ravel(), new_f_grid.ravel()]).T
    I_new = fun(new_points).reshape(len(t_int), len(f_new))
    #I_new = fun((t_int,f_new))
    if ns is None:
        nI_new=np.random.normal(size=np.shape(I_new))*np.std(I_new)/6
    else:
        nfun=interpolate.interp2d(f, mjd, ns, kind='linear')
        nI_new=nfun(f_new,t_int)
    t_sec_new = (np.arange(t_len) * new_ntbin * 24.*3600. *u.s)
    return I_new, f_new, t_sec_new, t_int, nI_new


def get_lt_lf(spec):
    Nf = len(spec.f)
    df = spec.bw / (Nf - 1)
    lag_chanf = np.fft.fftshift(np.fft.fftfreq(Nf))*Nf
    lags_freq = lag_chanf *df
    Nt = len(spec.t)
    dt = (spec.t[-1] - spec.t[0]) / (Nt - 1)
    lag_chant = np.fft.fftshift(np.fft.fftfreq(Nt))*Nt
    lags_time = lag_chant *dt
    return lags_time, lags_freq

def get_acf(spec):
    ac_ss=np.abs((np.fft.fft2(spec.I)))**2
    acf=(np.fft.fftshift(np.fft.ifft2(ac_ss))).real.T
    lags_time, lags_freq=get_lt_lf(spec)
    return acf, lags_time, lags_freq

def cut_acf(acf, lags_time, lags_freq, lt_cutoff=4000,lf_cutoff=40, ii=0):
    
    ltc=(lags_time[(lags_time.value>-lt_cutoff) & (lags_time.value<lt_cutoff) ]).to(u.min)
    lfc=lags_freq[(lags_freq.value>-lf_cutoff) & (lags_freq.value<lf_cutoff) ]

    acfc=acf[(lags_freq.value>-lf_cutoff) & (lags_freq.value<lf_cutoff), :]
    acfc=acfc[:, (lags_time.value>-lt_cutoff) & (lags_time.value<lt_cutoff)]

    dtau0=np.argmin(np.abs(ltc))
    dnu0=np.argmin(np.abs(lfc))
    ii=ii
    acfc[dnu0,dtau0]=np.nan
    acfc[dnu0-ii:dnu0+ii,dtau0-ii:dtau0+ii]=np.nan
    dfreq_prof=acfc[:,dtau0]-np.nanmedian(acfc[:,dtau0])
    dtime_prof=acfc[dnu0,:]-np.nanmedian(acfc[dnu0,:])
    return acfc, ltc, lfc, dtime_prof, dfreq_prof

def plot_acf(acf, lt, lf, new_fig=True, figsize=(3,3), dpi=150):
    vmin, vmax=np.percentile(acf, [0.1,99.9])
    if new_fig is True:
        fig=plt.figure(figsize=figsize, dpi=dpi, facecolor='w', edgecolor='k')
    plt.imshow(acf, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax, origin='lower',
               extent=[lt[0].value,lt[-1].value, lf[0].value,lf[-1].value])
    plt.xlabel('Time lag, sec')
    plt.ylabel('Freq lag, MHz')


##Curvature evolution with freuquency
def eta_func(f0,A):
    '''Daniels function'''
    return(A/(f0**2))

class SecSpec(object):
    '''Seconday spectum object
    Is - 2d fourier transform of dynamic spectrum (I or ds); type: 2D complex numpy array
    tau - doppler delay (fourier tranform axis in frequency domain); type: 1D astropy quantity (mus)
    fd - frindge frequency (fourier transform axis in time domain); type: 1D astropy quentity (mHz)'''
    def __init__(self, ds, Is, tau, fd):
        self.ds = ds
        self.Is = Is
        self.tau = tau
        self.fd = fd
    def __repr__(self):
        return "<Secondary spectrum>"

class Acf(object):
    '''Autocorrelation of the dynamic spectrum and its axes
    acf - autocorrelation function itself (2D complex np array)
    lt - time lag; (acf axis in time domain); type: 1D astropy quantity (min)
    lf - frequency lag (acf axis in frequency domain); type: 1D astropy quantity (MHz)'''
    def __init__(self, acf, lt, lf):
        self.acf=acf
        self.lt=lt
        self.lf=lf
    def __repr__(self):
        return "<ACF>"


class Spec(object):
    '''Spectrum object'''
    def __init__(self, I, t, f, stend=[0.0,1.0], nI=None, tel='Unknown', psr='PSRJ0337+1715',pad_it=True, npad=3, ns_info='no noise', subbands=None):
        '''class to manipulate dynamic spectra
        Usually initiated from npz file using function load_triple_spectra,
        but it can aslo be cropped from other Spec object
        or constracted manualy. While initiated it automatically creates secondary spectra, thus requiring decision about padding.
  
        I - dynamic spectra (2D float array)
        t, f - time and frequency axes (astropy.units quantities)
        stend - start and end of the observation in mjd (astropy.Time object)
        nI - array of noise in dynamic spectra

        tel - Observatory with which the data was obtained (if known)
        psr - pulsar name of the object (if known)
        pad_it  - whether to pad the secondary spectra with zeros
        npad - (int) number of paded zeros 
        ns_info - whether noise data is present or not
        '''
        self.I = I
        self.t = t
        self.f = f
        self.nsinfo=ns_info
        self.bw=np.ptp(f)
        self.cf=np.mean(f)
        self.dur=np.diff(stend)[0]*24*60*u.min
        if nI is None:
            self.nI=np.random.normal(size=np.shape(self.I))*np.std(self.I)/6
            self.nsinfo='no noise'
        else:
            self.nI=nI
        self.stend=stend
        
        if subbands is None:
            self.subbands=np.array([[1301,1317],[1321,1337],[1341,1357],[1361,1377],[1381,1397],
                      [1401,1417],[1421,1437],[1441,1457]])
        else:
            self.subbands=subbands

        full_time=(self.stend[1]-self.stend[0])
        ntbin=full_time/len(self.t)
        a_t_mjd = (np.arange(len(self.t)) * ntbin * u.d)+self.stend[0]*u.d
        self.mjd=Time(a_t_mjd, format='mjd', scale='utc')
                   
        self.tel=tel
        self.psr=psr
        self.ss = self.make_ss(pad_it, npad)
        self.acf = self.make_acf()
    def __repr__(self):
        times=(self.stend[1]-self.stend[0])*24.
        
        mjd_c=(self.stend[1]+self.stend[0])/2.
        return "<Dynamic spectrum: Dur: %.2f hr, Freq: %.2f - %2.f MHz, MJD: %.2f, PSR: %s, Tel: %s, %s>"%(times,np.amin(self.f).value, np.amax(self.f).value, mjd_c, self.psr, self.tel, self.nsinfo)
    def plot_ds(self, new_fig=True, figsize=(3,8), dpi=150, vmin=None, vmax=None, cmap='viridis', lab_mfr=False, intp='none'):
        '''Plots ds with pre-defined settings. (see fun_plot_ds)'''
        fun_plot_ds(self.I, self.t, self.f,new_fig=new_fig,figsize=figsize, dpi= dpi, vmin=vmin, vmax=vmax, cmap=cmap, lab_mfr=lab_mfr, intp=intp)
    def plot_nds(self, new_fig=True, figsize=(3,8), dpi=150, vmin=None, vmax=None, cmap='viridis'):
        '''Plots noise of the dynamic spectra same way as plot_ds. (see plot_ds)'''
        fun_plot_ds(self.nI, self.t, self.f,new_fig=new_fig,figsize=figsize, dpi= dpi, vmin=vmin, vmax=vmax,cmap=cmap)
        
    def make_ss(self, pad_it=True, npad=3):
        '''Makes secondary spectra and  loads it to SecSpec object'''
        Is, tau, fd=fun_make_ss(self.I, self.t, self.f, pad_it=pad_it, npad=npad)
        return SecSpec(self, Is, tau, fd)   

 
    def plot_ss(self,fd_lim=[-1.5, 1.5], tau_lim=[0.0,1.4], vmin=None, vmax=None, new_fig=True, figsize=(3,2), dpi=150, cb=True,cmap='viridis', plot_parabola=False, eta=1):
        '''Plots secondary spectra with pre-defined plotting settings'''
        fun_plot_ss(self.ss.Is, self.ss.tau, self.ss.fd, fd_lim=fd_lim, tau_lim=tau_lim, vmin=vmin, vmax=vmax,
                new_fig=new_fig,figsize=figsize, dpi=dpi, cb=cb, cmap=cmap, plot_parabola=plot_parabola, eta=eta)

    def make_acf(self):
        '''Makes autocorrlation function and loads it to Acf object'''
        acf, lt, lf=get_acf(self)
        return Acf(acf=acf, lt=lt, lf=lf)

    def plot_acf(self, new_fig=True, figsize=(3,3), dpi=150):
        '''Plots acf'''
        plot_acf(self.acf.acf, self.acf.lt, self.acf.lf, new_fig=new_fig, figsize=figsize, dpi=dpi)

    def select(self, time_sel=None, freq_sel=None, freq_idx=None, time_idx=None, pad_it=True, npad=3):
        '''Crops a piece of ds as a sepatate Spec object'''
        if (time_sel is not None) and (time_sel[0].unit == u.s):
            time_axis=self.t
            I_sel, sec_sel, f_sel, t_idx, f_idx, nI_sel=fun_select(self.I,time_axis,self.f, time_sel=time_sel,
                              freq_sel=freq_sel, freq_idx=freq_idx,time_idx=time_idx, ns=self.nI)
            sec_sel=sec_sel-sec_sel[0]
            print ('it is happening')
            mjd_sel=self.mjd.mjd[t_idx[0]:t_idx[1]]*u.d
            mjd_bin=mjd_sel[1]-mjd_sel[0]
            my_stend=[mjd_sel[0].value,mjd_sel[-1].value+mjd_bin.value]
            I_sel = self.I[t_idx[0]:t_idx[1],f_idx[0]:f_idx[1]]
            nI_sel = self.nI[t_idx[0]:t_idx[1],f_idx[0]:f_idx[1]]

        if (time_sel is not None) and (time_sel[0].unit == u.d):
            time_axis=self.mjd.mjd * u.d
            I_sel, mjd_sel, f_sel, t_idx, f_idx, nI_sel=fun_select(self.I,time_axis,self.f, time_sel=time_sel,
                              freq_sel=freq_sel, freq_idx=freq_idx,time_idx=time_idx, ns=self.nI)
            sec_sel=self.t[t_idx[0]:t_idx[1]]
            mjd_bin=mjd_sel[1]-mjd_sel[0]
            my_stend=[mjd_sel[0].value,mjd_sel[-1].value+mjd_bin.value]
            I_sel = self.I[t_idx[0]:t_idx[1],f_idx[0]:f_idx[1]]
            nI_sel = self.nI[t_idx[0]:t_idx[1],f_idx[0]:f_idx[1]]

        if (time_sel is None) and (time_idx is None):
            time_axis=self.t
            I_sel, mjd_sel, f_sel, t_idx, f_idx, nI_sel=fun_select(self.I,time_axis,self.f, time_sel=time_sel,
                              freq_sel=freq_sel, freq_idx=freq_idx,time_idx=time_idx, ns=self.nI)
            I_sel = self.I[:,f_idx[0]:f_idx[1]]
            nI_sel = self.nI[:,f_idx[0]:f_idx[1]]
            my_stend=self.stend
            sec_sel=self.t

        return Spec(I=I_sel, t=sec_sel, f=f_sel, stend=my_stend, nI=nI_sel,tel=self.tel, psr=self.psr, pad_it=pad_it, npad=npad, ns_info=self.nsinfo, subbands=self.subbands)

    def shrink(self, factor=[1,1], pad_it=True, npad=3):
        ds, t, f=fun_shrink_ds(self.I, self.t, self.f, factor=factor)
        ns=ld.shrink_any_2(self.nI, factor=factor)
        mjd=ld.shrink_any_1(self.mjd.mjd, factor=factor[0])
        mjd_bin=mjd[1]-mjd[0]
        return Spec(I=ds, t=t, f=f, stend=[mjd[0],mjd[-1]+mjd_bin], nI=ns,tel=self.tel, psr=self.psr, pad_it=pad_it, npad=npad, ns_info=self.nsinfo, subbands=self.subbands)

    def interp(self, t_ed, f_ed, t_len, f_len, pad_it=True, npad=3):
        '''Interpolates given ds to a new grid in time and frequency as well as new range'''
        I_new, f_new, t_sec_new, t_int, nI_new=fun_interp(self.I, self.mjd.mjd, self.f,
                                     t_ed=t_ed, f_ed=f_ed, t_len=t_len, f_len=f_len, ns=self.nI)
        new_spec=Spec(I=I_new, t=t_sec_new, f=f_new*u.MHz, stend=[t_ed[0].value,t_ed[1].value], nI=nI_new,
                        tel=self.tel, psr=self.psr, pad_it=pad_it, npad=npad)
        return new_spec

    def get_noise(self):
        '''Calculates noise level in ds using Daniel method'''
        temp=np.fft.fftshift(np.abs(np.fft.fft2(self.I.T)/np.sqrt(self.I.T.shape[0]*self.I.T.shape[1]))**2)
        N=np.sqrt(temp[:temp.shape[0]//6,:temp.shape[1]//6].mean())
        return N 

    def __add__(self, other, pad_it=True, npad=3):
        if len(self.f) != len(other.f):
            raise ValueError("ds objects have different frequency axis. Currently not supported") 

        if np.mean(self.mjd.mjd) < np.mean(other.mjd.mjd):
            st=self
            fin=other
        else:
            st=other
            fin=self

        if st.mjd[-1] > fin.mjd[0]:
            raise ValueError("ds objects overlap in time. Currently not supported")

        if (fin.mjd[0] - st.mjd[-1]) >0.5:
            raise ValueError("gap between data is too big")


        dt_st=(st.t[1]-fin.t[0]).to(u.d).value
        dt_fin=(fin.t[1]-fin.t[0]).to(u.d).value

        len_gap_fin=(fin.mjd.mjd[0]-st.mjd.mjd[-1])/dt_fin
        len_gap_st=(fin.mjd.mjd[0]-st.mjd.mjd[-1])/dt_st
        len_st=len(st.t)

        desired_len=int((fin.mjd.mjd[0]-st.mjd.mjd[0])/dt_fin)

        mjd_end=fin.mjd.mjd[0]-dt_fin
        mjd_st=fin.mjd.mjd[0]-dt_fin*desired_len

        gap_I=np.ones((int(len_gap_st), len(st.f)))*np.mean(st.I)
        #gap_mjd=np.arange(st.mjd.mjd[-1]+dt_st, st.mjd.mjd[-1]+dt_st*(int(len_gap_st)), dt_st)
        gap_mjd=np.linspace(st.mjd.mjd[-1]+dt_st, st.mjd.mjd[-1]+dt_st*(int(len_gap_st)), int(len_gap_st))
        #print (dt_st*24*3600, (gap_mjd[1]-gap_mjd[0])*24*3600, dt_fin*24*3600)
        #print ((dt_st-(gap_mjd[1]-gap_mjd[0]))*24*3600, (dt_st-dt_fin)*24*3600)

        st_gap_ds=np.concatenate((st.I, gap_I), axis=0)
        st_gap_mjd=np.concatenate((st.mjd.mjd, gap_mjd), axis=0)

        t_ed=[mjd_st*u.d, mjd_end*u.d+dt_fin*u.d]
        t_len=desired_len

        f_ed=[st.f[0], st.f[-1]]
        f_len=len(st.f)
        print (np.shape(st_gap_ds),np.shape(st_gap_mjd))
        I_new, f_new, t_sec_new, t_int, nI_new=fun_interp(st_gap_ds, st_gap_mjd, st.f,
                                                          t_ed, f_ed, t_len, f_len, ns=None)


        all_ds=np.concatenate((I_new, fin.I), axis=0)
        time_new=np.concatenate((t_sec_new.value,(fin.t+t_sec_new[-1]+(fin.t[1]-fin.t[0])).value), axis=0)

        new_spec=Spec(I=all_ds, t=time_new*u.s, f=fin.f, stend=[t_int[0],fin.mjd.mjd[-1]],
                                tel=fin.tel, psr=fin.psr, pad_it=pad_it, npad=npad)
        return new_spec
