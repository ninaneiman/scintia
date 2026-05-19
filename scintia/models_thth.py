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

sys.path.insert(1, '/home/gusinskaia/scintools/scintools')
from . import ththmod as THTH
np.seterr(divide='ignore', invalid='ignore')


##Function for making extent tuple for plotting
def ext_find(x,y):
    '''function from Daniels code'''
    dx=np.diff(x).mean()
    dy=np.diff(y).mean()
    ext=[(x[0]-dx/2).value,(x[-1]+dx/2).value,(y[0]-dy/2).value,(y[-1]+dy/2).value]
    return(ext)

def recover_phases(dspec,time,freq,SS,fd,tau,edges,eta_fit):
    thth_red, thth2_red, recov, model, edges_red,w,V = THTH.modeler(SS, tau, fd, eta_fit, edges)
    eta=eta_fit
    ththE_red=thth_red*0
    ththE_red[ththE_red.shape[0]//2,:]=np.conjugate(V)*np.sqrt(w)
    ##Map back to time/frequency space
    recov_E=THTH.rev_map(ththE_red,tau,fd,eta,edges_red,isdspec = False)
    model_E=np.fft.ifft2(np.fft.ifftshift(recov_E))[:dspec.shape[0],:dspec.shape[1]]
    model_E_ev=model_E[:dspec.shape[0],:dspec.shape[1]]
    model_E_ev=np.pad(model_E_ev,
                    (   (0,SS.shape[0]-model_E_ev.shape[0]),
                        (0,SS.shape[1]-model_E_ev.shape[1])),
                    mode='constant',
                    constant_values=model_E_ev.mean())
    model_field_ev=np.abs(np.fft.fftshift(np.fft.fft2(model_E_ev)))**2
    model_E_ev=model_E_ev[:dspec.shape[0],:dspec.shape[1]].T

    model_E*=(dspec.shape[0]*dspec.shape[1]/4)
    model_E[dspec>0]=np.sqrt(dspec[dspec>0])*np.exp(1j*np.angle(model_E[dspec>0]))
    model_E=np.pad(model_E,
                    (   (0,SS.shape[0]-model_E.shape[0]),
                        (0,SS.shape[1]-model_E.shape[1])),
                    mode='constant',
                    constant_values=model_E.mean())
    recov_E=np.abs(np.fft.fftshift(np.fft.fft2(model_E)))**2
    model_E=model_E[:dspec.shape[0],:dspec.shape[1]]
    N_E=recov_E[:recov_E.shape[0]//4,:].mean()
    model_ds=model[:dspec.shape[0],:dspec.shape[1]]
    model_ss=recov
    model_field=recov_E
    return model_E, model_ds, model_ss, model_field, model_E_ev, model_field_ev

def get_models(sp_part, eta_fit=1.6*u.us/(u.mHz**2), edge=1.0,ntau=512):
    #need to update the chi2
    edges=np.linspace(-edge,edge,ntau)
    n_ds,n_t,n_f,p_sec,p_fd,p_tau=sp_part.I, sp_part.t, sp_part.f, sp_part.ss.Is, sp_part.ss.fd, sp_part.ss.tau
    model_E, model_ds,model_ss, model_field, model_E_ev, model_field_ev=recover_phases(n_ds.T,n_t,n_f,p_sec,p_fd,p_tau, edges,eta_fit)
    N=np.mean(sp_part.nI)
    chi2=((sp_part.I-model_ds.T)**2).mean()/N**2
    return model_E.T, model_ds.T, model_ss, model_field, model_E_ev, model_field_ev, chi2

def get_models_spec(sp_part, eta_fit=1.6*u.us/(u.mHz**2), edge=1.0,ntau=512, model_ev=False):
    model_E, model_ds, model_ss, model_field, model_E_ev, model_field_ev, chi2 = get_models(sp_part, eta_fit=eta_fit, edge=edge,ntau=ntau)
    if model_ev is True:
        model_E=model_E_ev
        model_field=model_field_ev
    model_spec=ModelSpec(eta=eta_fit, mI=model_ds, mIs=model_ss, mE=model_E, mEs=model_field,
                                     spec=sp_part)
    return model_spec

class ModelSpec(object):
    def __init__(self, eta, mI, mIs, mE, mEs, spec):
        self.eta = eta
        self.mI = mI
        self.t=spec.t
        self.f=spec.f
        self.mIs=mIs
        self.tau = spec.ss.tau
        self.fd = spec.ss.fd
        self.mE=mE
        self.mEs=mEs
        self.spec=spec
    def __repr__(self):
        return "<Model spectrum for eta=%.1f %s>"%(self.eta.value,self.eta.unit.to_string())
    def plot_mds(self, new_fig=True, figsize=(3,3), dpi=150, vmin=None, vmax=None):
        if new_fig is True:
            plt.figure(figsize=figsize, dpi=dpi)
        if (vmin is None) and (vmax is None):
            vmin,vmax = np.percentile(self.mI,[1,99])
        plt.imshow(self.mI.T,extent=(0,(self.spec.stend[1]-self.spec.stend[0])*24,
                                     np.amin(self.f).value,np.amax(self.f).value),
                   vmin=vmin, vmax=vmax, aspect='auto', origin='lower')
        plt.xlabel('Time (hr)')
        plt.ylabel('Frequency (%s)'%self.f.unit.to_string())
    def plot_mss(self,fd_lim=[-1.5,1.5], tau_lim=[0,1.4], vmin=None, vmax=None,cb=True, new_fig=True, figsize=(3,3), dpi=150):
        if new_fig is True:
            plt.figure(figsize=figsize, dpi=dpi)
        SS_ext=ext_find(self.fd,self.tau)
        if vmin is None:
            vmin=np.median(np.abs(self.mIs[(self.mIs!=0)])**2)/10
        if vmax is None:
            vmax=np.max(np.abs(self.mIs[(self.mIs!=0)])**2)*np.exp(-2.7)*10
        if tau_lim is None:
            tau_lim=np.array([self.tau[0].value,self.tau[-1].value])
        if fd_lim is None:
            fd_lim=np.array([self.fd[0].value,self.fd[-1].value])

        plt.imshow(np.abs(self.mIs)**2.,norm=LogNorm(vmax=vmax, vmin=vmin),aspect='auto',extent=SS_ext, origin='lower')
        plt.xlabel(self.fd.unit.to_string('latex'))
        plt.ylabel(self.tau.unit.to_string('latex'))
        if cb is True:
            plt.colorbar()
        plt.xlim(fd_lim)
        plt.ylim(tau_lim)
    def plot_me(self,cmap='seismic', new_fig=True, figsize=(3,3), dpi=150):
        if new_fig is True:
            plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(np.angle(self.mE).T,aspect='auto', origin='lower',interpolation='none',cmap=cmap,
                  extent=(0,(self.spec.stend[1]-self.spec.stend[0])*24,
                                     np.amin(self.f).value,np.amax(self.f).value))
        plt.xlabel('Time (hr)')
        plt.ylabel('Frequency (%s)'%self.f.unit.to_string())
    def plot_mes(self,fd_lim=[-1.5,1.5],tau_lim=[0.0,1.4],vmin=None,vmax=None,vv_default=True, cb=True,new_fig=True, figsize=(3,3),dpi=150):
        fun_plot_mes(self.mEs, self.fd, self.tau, fd_lim=fd_lim, tau_lim=tau_lim, vmin=vmin, vmax=vmax, vv_default=vv_default, cb=cb,new_fig=new_fig,figsize=figsize, dpi=dpi)


def fun_plot_mes(mEs,fd,tau,fd_lim=[-1.5,1.5],tau_lim=[0.0,1.4],vmin=None,vmax=None,vv_default=True, cb=True,new_fig=True, figsize=(3,3),dpi=150, cmap='viridis'):
    if new_fig is True:
        plt.figure(figsize=figsize, dpi=dpi)
    SS_ext=ext_find(fd,tau)
    if vmin is None and vv_default is True:
        vmin=np.median(np.abs(mEs[(mEs!=0)]))*5
    if vmax is None and vv_default is True:
        vmax=np.max(np.abs(mEs[(mEs!=0)]))*np.exp(-2.7)*5
    if tau_lim is None:
        tau_lim=np.array([tau[0].value,tau[-1].value])
    if fd_lim is None:
        fd_lim=np.array([fd[0].value,fd[-1].value])

    plt.imshow(mEs,norm=LogNorm(vmax=vmax, vmin=vmin),aspect='auto',extent=SS_ext, origin='lower', cmap=cmap)
    plt.xlabel(fd.unit.to_string('latex'))
    plt.ylabel(tau.unit.to_string('latex'))
    if cb is True:
        plt.colorbar()
    plt.xlim(fd_lim)
    plt.ylim(tau_lim)


def plot_etas(dic, new_fig=True, figsize=(4,3), dpi=150):
    if new_fig is True:
        plt.figure(figsize=figsize, dpi=dpi)
    etas=dic['par_array']
    measure=dic['chi2']
    eta_fit=dic['par_fit']
    fit_res=dic['fit_res']
    etas_fit=dic['fit_array']
    eta_sig=dic['par_sig']

    plt.plot(etas,measure)
    if not np.isnan(eta_fit):
        plt.plot(etas_fit,
            THTH.chi_par(etas_fit.value, *fit_res),
            label=r'x = %.1f $\pm$ %.1f' % (eta_fit.value, eta_sig.value))
        plt.legend(fontsize=8, loc='upper center')

