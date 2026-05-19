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
#import scintools.ththmod as THTH
from scipy.sparse.linalg import eigsh
from scipy.optimize import curve_fit

sys.path.insert(1, '/home/gusinskaia/scintools/scintools')
from . import ththmod as THTH
np.seterr(divide='ignore', invalid='ignore')


def eta_to_mu(eta, spec, deff):
    '''converts eta to mu_eff for a specific ds chunk
    eta - with units us/mHz**2 
    spec - Spec object. required just for central frequency.
    deff - effective distance
    returns: mu_eff in units of mas/yr'''
    c_f=spec.f.mean()
    vel=(const.c/(2.*c_f**2.*deff*eta))**0.5*deff
    mu=(vel.decompose().to('km/s').value)*1e3/(4.74*deff.to(u.pc).value)
    return mu*u.mas/u.yr

def mu_to_eta(mu, spec, deff):
    '''converts mu_eff to eta for a specific ds chunk
    mu - mu_eff in units mas/yr
    spec - Spec object. required just for central frequency.
    deff - effective distance
    returns: eta in units us/mHz**2'''
    c_f=spec.f.mean()
    vel=mu.value*4.74*deff.to(u.pc).value*1e-3
    vel=vel*u.km/u.s
    eta=deff*const.c/(2*c_f**2*vel**2)
    return eta.to(u.us/u.mHz**2)

def eta_to_dveff_cf(eta, c_f):
    '''converts eta to v_eff/sqrt(deff)
    eta - with units us/mHZ**2 
    c_f - central frequency
    returns: dveff=v_eff/sqrt(deff) in units km/s/pc**0.5'''
    dveff=(const.c/(2.*c_f**2.*eta))**0.5
    return dveff.decompose().to(u.km/(u.pc**0.5 *u.s))

def eta_to_dveff(eta, spec):
    '''converts eta to v_eff/sqrt(deff) for a specific ds chunk
    eta - with units us/mHZ**2 
    spec - Spec object. required just for central frequency
    returns: dveff=v_eff/sqrt(deff) in units km/s/pc**0.5'''
    c_f=spec.f.mean()
    dveff=eta_to_dveff_cf(eta, c_f)
    return dveff

def dveff_to_eta_cf(dveff, c_f):
    '''converts v_eff/sqrt(deff) to eta
    dveff-v_eff/aqrt(deff) - with units km/s/pc**0.5 
    c_f - central frequency
    returns: eta in units us/mHz**2'''
    eta=const.c/(2*c_f**2*dveff**2)
    return eta.to(u.us/u.mHz**2)

def dveff_to_eta(dveff, spec):
    '''converts v_eff/sqrt(deff) to eta for a specific ds chunk
    dveff-v_eff/aqrt(deff) - with units km/s/pc**0.5 
    spec - Spec object. required just for central frequency
    returns: eta in units us/mHz**2'''
    c_f=spec.f.mean()
    eta=dveff_to_eta_cf(dveff, c_f)
    return eta


def parabola_fit(par_array, chisq):
    '''Fit for a parabola around the minimum. Finds the parameter and its errorbar - i.e. your solution.
    par_array - array of values that corresponds to the chi2 curve
    chisq - array of chisq values for these parameters
    returns:
    par_fit  - the solution
    par_sig - the errorbar
    popt - curve fit output
    fit_range_array - array of par values for which parabola fit was performed'''
    #plt.plot(par_array, chisq)
    p_min=par_array[chisq==chisq.min()][0]
    fit_results=chisq[np.abs(par_array-p_min)<.1*p_min]
    C=fit_results.min()

    fit_range_array=par_array[np.abs(par_array-p_min)<.1*p_min]
    x0=fit_range_array[fit_results==C][0].value
    A=(fit_results[0]-C)/((fit_range_array[0].value-x0)**2)
    try:
        popt,pcov=curve_fit(THTH.chi_par,fit_range_array.value,fit_results,p0=np.array([A,x0,C]))
        par_fit=popt[1]*par_array.unit
        par_sig=np.sqrt((fit_results-THTH.chi_par(fit_range_array.value,*popt)).std()/popt[0])*par_array.unit

    except RuntimeError:
        print('Fit curve didnt converge')
        par_fit, par_sig, popt=float('nan') , 0.0*par_array.unit,  0.0
    except TypeError:
        print ('Parabola is being fitted to only two points. The chi2 curve is not smooth - probably bad ds chunk')
        par_fit, par_sig, popt=float('nan') , 0.0*par_array.unit,  0.0
    return par_fit, par_sig, popt, fit_range_array


def daniel_pars_fit(spec, curv_par='eta', par_lims=[0.25,5.5], edge=1.4,ntau=512, d_eff=0.325*u.kpc, npoints=100, thth_method='coherent', chi2_method='Nina', reduced=True, edge_threshold=False, tau_ed=0.25, chi2svd=False, cen0=0.1, eta2=None, edge2=None):
    '''Performes th-th fit for a given ds chunk.
    spec - Spec. object. contains ds,f,t its ss, tau, fd and other info about the observation
    curv_par - which parameter to use for the chisq curve fitting: eta, mueff or dveff = veff/sqrt(deff). Note that eta is non-linear while two others are
    par_lims - range of par values for which perform thth and calculate chisq
    edge - limit of fd in ss until which to make thth decomposition and model
    ntau - number of points to which split the thth decomposition and model
    d_eff - effective distance. Needed to fit in mueff space. Use guessed value if not known
    npoints - number of values of parameter to perform thth and calculte chisq. Number of points to split the range (par_lims) into
    chi2_method - which formula to use for chisq calculation. Daniel - ss based. Nina - ds based
    reduced - whether or not to use reduced chisq, i.e. divide each value by ndof. (Not reccommeneded - introduces bias)
    ---
    edge_threshold, tau_ed, chi2svd, cen0, eta2 and edge2 - for 2-screen thth. Will comment on that later. Rarely in use 
    -------
    Returns:
    fitdic - dictionary with the parameter results: eta, mueff, dveff and its errors, as well as frequency and mjd
    mean frequecy of the ds chunck
    mean mjd of the ds chunck
    res_dic - disctionary with curve fit inputs and outputs, mostly for plotting and checking 
    '''
    print ('fit:', spec)
    edges=np.linspace(-edge,edge,ntau)    
   
    if curv_par=='eta': 
        print ('Warning! Curretly the combined chi2 does not support eta as initial parameter')
        eta_low=np.amin(par_lims)*u.us/(u.mHz**2)
        eta_high=np.amax(par_lims)*u.us/(u.mHz**2)
        pars2=np.linspace(eta_low.value,eta_high.value,npoints)*eta_low.unit
        etas2=pars2
        
    if curv_par=='mueff':
        mu_low=np.amin(par_lims)*u.mas/u.yr
        mu_high=np.amax(par_lims)*u.mas/u.yr
        pars2=np.linspace(mu_high.value,mu_low.value,npoints)*mu_low.unit
        etas2=mu_to_eta(pars2, spec, d_eff)

    if curv_par=='dveff':
        dveff_low=np.amin(par_lims)*(u.km/(u.pc**0.5 *u.s))
        dveff_high=np.amax(par_lims)*(u.km/(u.pc**0.5 *u.s))
        pars2=np.linspace(dveff_high.value,dveff_low.value,npoints)*dveff_low.unit
        etas2=dveff_to_eta(pars2, spec)
    
    chisq=np.zeros(pars2.shape[0])
    ntheta_reds=np.zeros(pars2.shape[0])
    N=spec.get_noise()
    edges=np.linspace(-edge,edge,ntau)
    mask=np.ones(spec.I.T.shape,dtype=bool)
    for i in range(pars2.shape[0]):
        eta=etas2[i]
        if edge_threshold is True:
            fd_ed=np.sqrt(tau_ed/eta.value)
            fd_ed=np.amin([1.0,fd_ed])
            edges=np.linspace(-fd_ed,fd_ed,ntau)
        ##Fits a model generated by taking the outer product of the dominate eigenvector
        ##(in theta-theta space) and then mapping back to the dynamic spectrum
        if thth_method == 'coherent':
            if chi2_method == 'Daniel':
                chisq[i]=THTH.chisq_calc(spec.I.T,spec.ss.Is, spec.ss.tau, spec.ss.fd, eta, edges,mask,N)
            if chi2_method == 'Nina':
                chisq[i] = nina_get_chi2_spec(spec, eta, edge, ntau, reduced=reduced, chi2svd=chi2svd, cen0=cen0, eta2=eta2, edge2=edge2)
            if chi2_method == 'Eigen':
                chisq[i] = -THTH.Eval_calc(spec.ss.Is, spec.ss.tau, spec.ss.fd, eta, edges)
        if thth_method == 'incoherent':
            chisq[i] = -THTH.Eval_calc(abs(spec.ss.Is)**2, spec.ss.tau, spec.ss.fd, eta, edges)
    ##Fit for a parabola around the minimum
    par_fit, par_sig, popt, fit_range_array = parabola_fit(pars2,chisq)
    measure=chisq

    if np.isnan(par_fit):
        eta_fit, mueff_fit, dveff_fit=par_fit,par_fit,par_fit
        eta_sig, mueff_sig, dveff_sig=0.0*u.us/(u.mHz**2), 0.0*u.mas/u.yr, 0.0*u.km/(u.pc**0.5 *u.s)
    else:
        if curv_par=='mueff':
            mueff_fit=par_fit
            mueff_sig=par_sig
            eta_fit=mu_to_eta(par_fit, spec, d_eff)
            eta_sig=(2.*eta_fit/mueff_fit)*mueff_sig
            dveff_fit=eta_to_dveff(eta_fit, spec)
            dveff_sig=(0.5*dveff_fit/eta_fit)*eta_sig

        if curv_par=='dveff':
            dveff_fit=par_fit
            dveff_sig=par_sig
            eta_fit=dveff_to_eta(par_fit, spec)
            eta_sig=(2.*eta_fit/dveff_fit)*dveff_sig
            mueff_fit=eta_to_mu(eta_fit, spec, d_eff)
            mueff_sig=(0.5*mueff_fit/eta_fit)*eta_sig

        if curv_par=='eta':
            eta_fit, eta_sig= par_fit, par_sig
            mueff_fit=eta_to_mu(eta_fit, spec, d_eff)
            mueff_sig=(0.5*mueff_fit/eta_fit)*eta_sig
            dveff_fit=eta_to_dveff(eta_fit, spec)
            dveff_sig=(0.5*dveff_fit/eta_fit)*eta_sig
     
    fitdic={'eta':eta_fit, 'mueff':mueff_fit, 'dveff':dveff_fit, 'eta_err':eta_sig, 'mueff_err':mueff_sig,
              'dveff_err':dveff_sig, 'mean_f':np.mean(spec.f), 'mean_t':np.mean(spec.mjd.mjd)}
    
    res_dic={'par_array':pars2, 'fit_array':fit_range_array, 'chi2':measure, 'fit_res':popt,
              'par_fit':par_fit, 'par_sig':par_sig, 'mean_f':np.mean(spec.f), 'mean_t':np.mean(spec.mjd.mjd)}
    return fitdic, np.mean(spec.f), np.mean(spec.mjd.mjd), res_dic
    

def nina_get_chi2_spec(spec, eta, edge=1.4, ntau=512, fd2=None,tau2=None, plot_mds=False, reduced=True, chi2svd=False, cen0=0.1, eta2=None, edge2=None):
    '''Calculate chisq for each thth decomposion of a given Spec object (i.e. ds chunck).
    See nina_get_chi2 for more details
    ---
    spec - Spec. object. contains ds,f,t its ss, tau, fd and other info about the observation
    eta - curvature of the parabola in ss
    edge - limit of fd in ss until which to make thth decomposition and model
    ntau - number of points to which split the thth decomposition and model
    -- 
    the rest of the parameters are only used for 2 screen thth - comment on that later
    -----
    Returns:
    chisq - value of calculated chisq of thth model for a given curvature (eta)
    '''
    if chi2svd is True:
        chisq=-nina_get_svd(spec.I, spec.ss.Is, spec.ss.tau, spec.ss.fd, eta, edge, ntau, cen0=cen0, eta2=eta2, edge2=edge2)
    else:
        chisq=nina_get_chi2(spec.I, spec.ss.Is, spec.ss.tau, spec.ss.fd, eta, edge, ntau, fd2, tau2, plot_mds, spec.nI, reduced=reduced)
    return chisq

def nina_get_chi2(ds, SS,tau,fd, eta, edge=1.4, ntau=512, fd2=None,tau2=None, plot_mds=False, ns=None,
reduced=True):
    '''Calculate chisq for each thth decomposition of given ds and its noise
    ds - dynamic spectrum
    SS - intensity of secondary spectrum (ss)
    tau - delay of ss
    fd - doppler shift of ss
    eta - curvature of the parabola in ss
    edge - limit of fd in ss until which to make thth decomposition and model
    ntau - number of points to which split the thth decomposition and model
    ---
    fd2, tau2 - for 2 screen thth. Comment that later. Rarely in use
    ---
    plot_mds - whether to plot resulted model ds. Dunno why it was needed here
    ns - noise of the ds. For triple system data I produce noise real noise array together with ds. If you don't have one, it will automatically generate one
    reduced - whether to make reduced or not chisq. i.e. include or not ndof
    -------
    Returns:
    chisq - value of calculated chisq of thth model for a given curvature (eta)
    '''
    edges=np.linspace(-edge,edge,ntau) 
    thth_red,thth2_red,recov,model,edges_red,w,V=THTH.modeler(SS,tau,fd, eta, edges, fd2, tau2)
    emin, emax=np.amin(edges_red), np.amax(edges_red)
    c=((tau.value > emin)&(tau.value < emax) )
    tau_red=tau[c]
    #ndof is number of degrees of freedom - i.e. number of points minus number of fitting parameters:
    ndof=ds.size-tau_red.shape[0]*2-2#/(SS.shape[0]/ds.shape[1]) - 2
    if ns is None:
        #Devided by - 6 because this is roughly the S/N ratio of my observations. Guess I can make it a parameter.
        ns=np.random.normal(size=np.shape(ds))*np.std(ds)/6
    if plot_mds is True:
        vmin, vmax=np.percentile(model, [1,99])
        plt.imshow(model, aspect='auto', vmin=vmin, vmax=vmax, origin='lower')
        plt.show()
    chisq=((model[:ds.T.shape[0],:ds.T.shape[1]]-ds.T)**2).sum()/np.mean(ns)**2
    if reduced is True:
        chisq=chisq/ndof
    else:
        #actually this is some sort of scaled chisq... values don't matter anyway...
        chisq=chisq/ds.size
    return chisq

def nina_get_svd(ds, SS,tau,fd, eta, edge=1.4, ntau=512, cen0=0.1, eta2=None, edge2=None):
    '''This function only used for 2 screen thth - comment on that later'''
    if eta2 == None:
        eta2=eta
    edges=np.linspace(-edge,edge,ntau)
    if edge2 == None:
        edges2=edges
    else:
        edges2=np.linspace(-edge2,edge2,ntau)
    thth_red, edges_red1, edges_red2 = THTH.two_curve_map(SS, tau, fd, eta,
                                                edges, eta2, edges2)
    cents1=(edges_red1[1:]+edges_red1[:-1])/2
    thth_red[:,np.abs(cents1)<=cen0]=0
    U,S,W=np.linalg.svd(thth_red)
    chisq_svd=S[0]
    return chisq_svd


def nina_two_screen_chi2(ds, SS,tau,fd, eta, edge=1.4, ntau=512, eta2=None, edge2=None):
    '''This function is only used for 2 screen thth - comment on that later'''
    if eta2 == None:
        eta2=eta
    edges=np.linspace(-edge,edge,ntau)
    if edge2 == None:
        edges2=edges
    
    thth=THTH.thth_map2(SS.T, tau, fd, eta1, edges1,eta2,edges2)
    recov=THTH.rev_map2(thth, tau, fd, eta1, edges1, eta2, edges2,isdspec=True)
    model_modeler=np.fft.ifft2(np.fft.ifftshift(recov)).real
    chi2_val=np.sum(((model_modeler-ds.T)**2)/ds.size)
