from glob import glob
import os, math, time
import sys

import numpy as np
import astropy
from astropy import units as u, constants as const
from astropy.time import Time
from astropy.visualization import quantity_support

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl

import psrchive

#load functions from scintia:
from . import load_data as ld
from . import archive_functions as af
from . import ds_psr as dsa

from numpy.fft import rfft, irfft, fft, ifft
from scipy.optimize import curve_fit


def Gauss(x, A, B): 
    y = A*np.exp(-1*B*x**2) 
    return y 


def plot_pbf_from_ss(spec, tau_noise=0.075, plot_it=True):
    spec.ss=spec.make_ss(pad_it=False)
    fd_res=spec.ss.fd[1]-spec.ss.fd[0]
    tau_res=spec.ss.tau[1]-spec.ss.tau[0]

    rec_mEs=abs(spec.ss.Is)**2
    delay=rec_mEs.sum(1)
    if plot_it is True:
        plt.plot(spec.ss.tau, delay, alpha=0.5, color='blue')
        plt.yscale('log')
        plt.xlim(-0.01,0.125)
        plt.xlabel(r'delay ($\mu$s)')
        plt.ylabel('Field intensity')

    tau_axis=spec.ss.tau.value
    fd_axis=spec.ss.fd.value
    tau_int=rec_mEs.sum(1)
    noise_part=tau_int[(tau_axis>tau_noise)]
    med_pos=np.median(noise_part)
    std_pos=np.std(noise_part)
    min_pos_value=med_pos+7*std_pos
    tau_axis_crop=tau_axis[(tau_int>min_pos_value)]
    max_tau=0.1# now empirically defined; used to be: #np.amax(tau_axis_crop) 
    pos_int=tau_int[(tau_axis>0.0)&(tau_axis<max_tau)]
    tau_axis_pos=tau_axis[(tau_axis>0.0)&(tau_axis<max_tau)]
    if plot_it is True:
        plt.plot(tau_axis_pos, pos_int, color='grey')
    try:
        delay_ss=np.average(tau_axis_pos, weights=pos_int)
        delay_ss_err=np.abs(np.average(tau_axis, weights=tau_int))
        if plot_it is True:
            plt.axvline(delay_ss, color='r', alpha=0.5)
    except:
        print ("Spectrum too noisy")
        delay_ss=0.0
        delay_ss_err=0.0
    if plot_it is True:
        plt.title(r"scatteting delay = %.3f $\mu$s"%delay_ss)
    return delay_ss, delay_ss_err


def plot_lf_fit(spec, lf_cutoff=None, plot_it=True):
    acf, lt, lf = dsa.get_acf(spec)
    if lf_cutoff is not None:
        acf, lt, lf, dtime_prof, dfreq_prof=dsa.cut_acf(acf, lt, lf, lt_cutoff=4000,lf_cutoff=lf_cutoff, ii=0)
        if plot_it is True:
            plt.plot(lf,dfreq_prof/np.nanmax(dfreq_prof), color='grey')
    else:
        dtau0=np.argmin(np.abs(lt))
        dnu0=np.argmin(np.abs(lf))
        acf[dnu0,dtau0]=np.nan
        dfreq_prof=acf[:,dtau0]-np.nanmedian(acf[:,dtau0])
        if plot_it is True:
            plt.plot(lf,dfreq_prof/np.nanmax(dfreq_prof), color='k')

    parameters, covariance = curve_fit(Gauss, lf[~np.isnan(dfreq_prof)].value, dfreq_prof[~np.isnan(dfreq_prof)]/np.nanmax(dfreq_prof))
    fit_A=parameters[0]
    fit_B=parameters[1]
    perr = np.sqrt(np.diag(covariance)) #uncertainity in the model curve
    error_B = perr[1] #1 standard deviation error of the width of the Gaussian

    fit_df=np.sqrt(np.log(2)/fit_B)
    fit_df_p=np.sqrt(np.log(2)/(fit_B+error_B))
    fit_df_m=np.sqrt(np.log(2)/(fit_B-error_B))
    error_fit_df=np.abs(fit_df_p-fit_df_m)/2
    
    if plot_it is True:
        fit_y = Gauss(lf.value, fit_A, fit_B)
        plt.title("Scint. bandwidth = %.3f MHz"%fit_df)
        if lf_cutoff is not None:
            plt.plot(lf,fit_y, color='red', ls='--', lw=2)
        else:
            plt.plot(lf,fit_y, color='dodgerblue', lw=2)
        plt.xlabel('Frequency lag, MHz')
    return fit_df, error_fit_df, lf, dfreq_prof


def summary_plot(portrait, spec, DM):
    fig=plt.figure(figsize=(10,10))
    rect=[0.0, 0.63, 0.25,0.25]
    fig.add_axes(rect)
    frame1=plt.gca()
    af.plot_portrait(portrait, spec)

    rect=[0.35, 0.63, 0.25,0.25]
    fig.add_axes(rect)
    spec.plot_acf(new_fig=False)
    plt.title("ACF")

    rect=[0.0, 0.3, 0.25,0.25]
    fig.add_axes(rect)
    spec.plot_ds(new_fig=False)
    plt.title("Dynamic spectrum")
    
    rect=[0.35, 0.3, 0.25,0.25]
    fig.add_axes(rect)
    spec.plot_ss(tau_lim=[0.0,0.125], fd_lim=[-2,2],new_fig=False)
    plt.title("Secondary spectrum")

    rect=[0.7, 0.3, 0.25,0.25]
    fig.add_axes(rect)
    tau_ss, tau_ss_err=plot_pbf_from_ss(spec)
    df_ss=1/(2*math.pi*tau_ss*u.us)


    rect=[0.7, 0.63, 0.25,0.25]
    fig.add_axes(rect)
    fit_df, fit_df_err,lf,dfreq_prof=plot_lf_fit(spec)
    scintl_size=df_ss.to(u.MHz).value
    fit_df, fit_df_err, lf,dfreq_prof=plot_lf_fit(spec, lf_cutoff=scintl_size*10)
    tau_df=(1/(2*math.pi*fit_df*u.MHz)).to(u.ns)
    
    plt.figtext(0.0,0.22, 'tau_s from acf: %.2f ns'%tau_df.value)
    plt.figtext(0.0,0.18, 'tau_s from ss: %.2f ns'%(tau_ss*1000))

    plt.figtext(0.19,0.22, 'nu_s from acf: %.2f MHz'%fit_df)
    plt.figtext(0.19,0.18, 'nu_s from ss: %.2f MHz'%(scintl_size*2))

    nu_s=np.mean([fit_df,scintl_size*2])
    nscin=np.ptp(spec.f).value/nu_s
    
    tau_s=np.mean([tau_df.value,tau_ss*1000])
    plt.figtext(0.0,0.14, 'nu_s=%.2f MHz, tau_s=%.2f ns; nscin=%.1f'%(nu_s, tau_s, nscin), fontsize=12)
    return fit_df, fit_df_err, tau_ss, tau_ss_err



