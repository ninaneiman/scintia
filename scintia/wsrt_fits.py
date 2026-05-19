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

from . import ththmod as THTH
np.seterr(divide='ignore', invalid='ignore')

from . import load_data as ld
from . import ds_psr as dsa
from . import fit_thth as fth
from . import models_thth as mth

def fit_wsrt_spec(my_spec, figsize=(5,7.5), spec_pieces='Default', par_lims=[0.5,5.5],
                      pc7=False, pc_overlap=False,load_model=False, eta_ref=None, ref_freq=None, edge=1.4, ntau=512,time_lim=2.0,save_models=False, wnoise=False, d_eff=0.325*u.pc, mean0=True, ind_mean0=True, curv_par='dveff', saveauxname='test_wnoise', chi2_method='Nina', reduced=True, edge_threshold=False, tau_ed=0.25, model_ev=False):
    if spec_pieces=='Manual':
        spec_pieces=np.array([[1301,1317],[1321,1337],[1341,1357],[1361,1377],[1381,1397],
                      [1401,1417],[1421,1437],[1441,1457]])
    else:
        spec_pieces=my_spec.subbands
    if pc_overlap is True:
        new_sp_pieces=np.empty((spec_pieces.shape[0]-1, 2), dtype=int)
        for i in range(0,new_sp_pieces.shape[0]):
            new_sp_pieces[i,0]=spec_pieces[i,0]
            new_sp_pieces[i,1]=spec_pieces[i+1,1]
        spec_pieces=new_sp_pieces
    res_fit, res_f, res_t, dics_res, all_models =[],[],[],[],[]
    if load_model is True:
        aux_name='load'
    else:
        aux_name='fit'
        f = open("thth_results_%s_%.2f_%s.txt"%(aux_name,my_spec.stend[0],saveauxname), "a")
    dics_res_a=[]
    res_f_a=[]
    models_e_a=[]
    vmin_ss,vmax_ss=np.percentile(np.abs(my_spec.ss.Is)**2, [10,100-5e-2])
    fig=plt.figure(figsize=figsize, dpi= 70, facecolor='w', edgecolor='k')
    fig.add_axes([0.0,0.0,0.25,1.0])
    shr_spec=my_spec.shrink(factor=[16,1])
    plt.gca()
    shr_spec.plot_ds(new_fig=False)
    for i in range(0,spec_pieces.shape[0]):
        spec_sel=my_spec.select(freq_sel=[spec_pieces[i,0]*u.MHz,spec_pieces[i,1]*u.MHz])

        if ind_mean0 is True:
            spec_sel.I=spec_sel.I-np.mean(spec_sel.I)
            spec_sel.ss=spec_sel.make_ss(pad_it=True, npad=3)
        fig.add_axes([0.9,0.006+0.125*i,0.25, 0.105])
        frame1=plt.gca()
        spec_sel.plot_ss(new_fig=False, cb=False,fd_lim=[-2.0,2.0],tau_lim=[0.0,1.3],vmin=vmin_ss,vmax=vmax_ss)
        frame1.axes.get_xaxis().set_ticks([])
        frame1.axes.get_yaxis().set_ticks([])
        spec_sel.get_noise()
        if load_model is False:
            fitdic, fit_f, fit_t, dic_res=fth.daniel_pars_fit(spec_sel, par_lims=par_lims,edge=edge,ntau=ntau,curv_par=curv_par, chi2_method=chi2_method, reduced=reduced, edge_threshold=edge_threshold, tau_ed=tau_ed)
            if np.isnan(fitdic['eta']):
                print ('the fit did not converge, not correct eta, skip this spw')
            else:
                f.write('%.3f, %.3f, %.3f, %.3f, %.3f, %.3f  \n'%(fitdic['eta'].value,
                            fitdic['eta_err'].value,fitdic['dveff'].value, fitdic['dveff_err'].value,
                                                     fit_f.value,fit_t))
                dics_res.append(dic_res)
                dics_res_a.append(dic_res)

            if np.isfinite(fitdic['eta']):
                res_fit.append(fitdic)
                res_f.append(fit_f)
                res_f_a.append(fit_f)
                res_t.append(fit_t)
  
  
                print (fit_f, fit_t)
                print ('eta  : %.3f'%fitdic['eta'].value, fitdic['eta'].unit,
                       'mueff: %.3f'%fitdic['mueff'].value, fitdic['mueff'].unit, 
                       'dveff: %.3f'%fitdic['dveff'].value, fitdic['dveff'].unit)
                print ('e_err: %.3f'%fitdic['eta_err'].value, fitdic['eta'].unit,
                       'm_err: %.3f'%fitdic['mueff_err'].value, fitdic['mueff'].unit, 
                       'v_err: %.3f'%fitdic['dveff_err'].value, fitdic['dveff'].unit)
                eta_load=fitdic['eta']

        else:
            eta=eta_ref*(ref_freq/spec_sel.f.mean())**2
            eta_err=0.0*eta.unit
            eta_f=np.mean(spec_sel.f)
            eta_t=np.mean(spec_sel.mjd.mjd)
            eta_load=eta
            print ('%.2f MHz'%eta_f.value, 'eta: %.3f'%eta_load.value, eta_load.unit)

        if np.isfinite(eta_load):
            model_spec=mth.get_models_spec(spec_sel,eta_load,edge=edge,ntau=ntau,model_ev=model_ev)
            models_e_a.append(model_spec.mE.T)
            h_ds=0.105
            if pc_overlap is True:
                h_ds=0.23
            fig.add_axes([0.3,0.006+0.125*i,0.25, h_ds])
            frame1=plt.gca()
            model_spec.plot_mds(new_fig=False)
            frame1.axes.get_xaxis().set_ticks([])
            frame1.axes.get_yaxis().set_ticks([])
            plt.xlabel('')
            fig.add_axes([0.6,0.006+0.125*i,0.25, 0.105])
            frame1=plt.gca()
            model_spec.plot_me(new_fig=False)
            frame1.axes.get_xaxis().set_ticks([])
            frame1.axes.get_yaxis().set_ticks([])
            fig.add_axes([1.2,0.006+0.125*i,0.25, 0.105])
            frame1=plt.gca()
            model_spec.plot_mss(new_fig=False, cb=False)
            frame1.axes.get_xaxis().set_ticks([])
            frame1.axes.get_yaxis().set_ticks([])
            fig.add_axes([1.5,0.006+0.125*i,0.25, 0.105])
            frame1=plt.gca()
            model_spec.plot_mes(new_fig=False, cb=False)
            frame1.axes.get_xaxis().set_ticks([])
            frame1.axes.get_yaxis().set_ticks([])
            fig.add_axes([1.85,0.006+0.125*i,0.2, 0.105])
            if save_models is True:
                all_models.append(model_spec)
            if load_model is False:
                frame1=plt.gca()
                mth.plot_etas(dic_res,new_fig=False)
                frame1.axes.get_xaxis().set_ticks([])
            else:
                aux_name=aux_name+'%.1f'%eta_load.value
        print ('----------')

    if load_model is False:
        fig.add_axes([2.2,0.625,0.25, 0.125])
        frame1=plt.gca()
        chi2sg=np.empty((len(dics_res_a),100))
        for g in range(0,len(dics_res_a)):
            chi2sg[g,:]=dics_res_a[g]['chi2']
            plt.plot(dics_res_a[g]['par_array'],dics_res_a[g]['chi2'], label ='%.2f MHz'%res_f_a[g].value)
            plt.legend(loc=(0.0,1.1))
        fig.add_axes([2.2,0.45,0.25, 0.125])
        frame1=plt.gca()
        sim_fit = fth.parabola_fit(dics_res_a[0]['par_array'],chi2sg.mean(0))
        plt.plot(dics_res_a[0]['par_array'],chi2sg.mean(0), label='x=%.1f pm %.1f'%(sim_fit[0].value, sim_fit[1].value))
        plt.legend(fontsize=8, loc='upper center')
    fig.add_axes([2.2,0.26,0.25, 0.15])
    frame1=plt.gca()
    my_spec.plot_ss(new_fig=False, cb=False,vmin=vmin_ss,vmax=vmax_ss)
    #this makes full model via Daniel's mozaicking
    #---------------------------------------------
    #if pc_overlap is True:
    #    chunks=np.zeros((len(models_e_a),1,models_e_a[0].shape[0],models_e_a[0].shape[1]),dtype=complex)
    #    for h in range(0,len(models_e_a)):
    #        chunks[h,0,:,:]=models_e_a[h]
    #    full_mE=THTH.mosaic(chunks)
    #    me_f= np.linspace(spec_pieces[0,0],spec_pieces[-1,-1], full_mE.shape[0]) * u.MHz
    #    me_t= my_spec.t
    #    fig.add_axes([0.6,0.0,0.25,1.0])
    #    frame1=plt.gca()
    #    plt.imshow(np.angle(full_mE), aspect='auto', origin='lower', cmap='seismic', interpolation='none') 
    #    npad=3
    #    full_mE_pad=np.pad(full_mE.T,((0,npad*full_mE.T.shape[0]),(0,npad*full_mE.T.shape[1])),mode='constant',
    #                    constant_values=full_mE.T.mean())
    #    full_mE_pad=full_mE_pad.T
    #    me_fd=THTH.fft_axis(me_t,u.mHz,npad)
    #    me_tau=THTH.fft_axis(me_f,u.us,npad)
    #    full_mEs=np.fft.fftshift(np.fft.fft2(full_mE_pad))
    #    fig.add_axes([2.2,0.026,0.25, 0.15])
    #    frame1=plt.gca()
    #    mth.fun_plot_mes(np.abs(full_mEs)**2, me_fd, me_tau, new_fig=False, cb=False, vmin=1e7, vmax=5e8)
    #    if load_model is True:
    #        sim_fit={'mE':full_mE, 't':me_t, 'f':me_f, 'mEs':full_mEs, 'fd':me_fd, 'tau':me_tau}

    plt.savefig('triple_%.2f_%s_%s_%s.png'%(my_spec.mjd.mjd.mean(),my_spec.tel,aux_name,saveauxname),
                format='png',bbox_inches='tight',dpi=90)
    plt.show()
    if load_model is False:
        f.close()
    return res_fit, res_f, res_t, dics_res, all_models, sim_fit





def load_wsrt_spec(my_spec, eta_ref=None, ref_freq=None,figsize=(5,7.5), spec_pieces='Default', pc_overlap=False, edge=1.4,ntau=512, save_models=False, wnoise=False, d_eff=0.325*u.kpc, saveauxname='test_wnoise', ind_mean0=True, model_ev=False):
    if spec_pieces=='Manual':
        spec_pieces=np.array([[1301,1317],[1321,1337],[1341,1357],[1361,1377],[1381,1397],
                      [1401,1417],[1421,1437],[1441,1457]])
    else:
        spec_pieces=my_spec.subbands
    if pc_overlap is True:
        new_sp_pieces=np.empty((spec_pieces.shape[0]-1, 2), dtype=int)
        ch_sz=int(my_spec.f.size/spec_pieces.shape[0])
        for i in range(0,new_sp_pieces.shape[0]):
            new_sp_pieces[i,:]=[i*ch_sz,i*ch_sz+ch_sz*2]
        spec_pieces=new_sp_pieces
    models_e_a=[]
    models_ds_a=[]
    vmin_ss,vmax_ss=np.percentile(np.abs(my_spec.ss.Is)**2, [10,100-5e-2])
    fig=plt.figure(figsize=figsize, dpi= 70, facecolor='w', edgecolor='k')
    plt.figtext(0.9,0.95,'MJD: %.2f'%my_spec.mjd.mjd.mean(), fontsize=15)
    dveff_here=fth.eta_to_dveff_cf(eta_ref, ref_freq)
    plt.figtext(1.25,0.95,r'$\eta$: %.2f $s^3$, $\nu_c$: %.1f MHz'%(eta_ref.value, ref_freq.value), fontsize=15)
    plt.figtext(1.9,0.95,'dveff:%.2f'%dveff_here.value, fontsize=15)
    fig.add_axes([0.0,0.0,0.25,1.0])
    plt.gca()
    my_spec.plot_ds(new_fig=False)
    for i in range(0,spec_pieces.shape[0]):
        spec_sel=my_spec.select(freq_idx=[spec_pieces[i,0],spec_pieces[i,1]])

        if ind_mean0 is True:
            spec_sel.I=spec_sel.I-np.mean(spec_sel.I)
        
        spec_sel.ss=spec_sel.make_ss(pad_it=True, npad=3)
        eta=eta_ref*(ref_freq/spec_sel.f.mean())**2
        eta_err=0.0*eta.unit
        eta_f=np.mean(spec_sel.f)
        eta_t=np.mean(spec_sel.mjd.mjd)
        eta_load=eta
        print ('%.2f MHz'%eta_f.value, 'eta: %.3f'%eta_load.value, eta_load.unit)

        model_spec=mth.get_models_spec(spec_sel, eta_load, edge=edge,ntau=ntau, model_ev=model_ev)
        models_e_a.append(model_spec.mE.T)
        models_ds_a.append(model_spec.mI.T)
        h_ds=0.105
        if pc_overlap is True:
            h_ds=0.23
        fig.add_axes([0.3,0.006+0.125*i,0.25, h_ds])
        frame1=plt.gca()
        model_spec.plot_mds(new_fig=False)
        frame1.axes.get_xaxis().set_ticks([])
        frame1.axes.get_yaxis().set_ticks([])
        plt.xlabel('')
        fig.add_axes([0.6,0.006+0.125*i,0.25, 0.105])
        frame1=plt.gca()
        model_spec.plot_me(new_fig=False)
        frame1.axes.get_xaxis().set_ticks([])
        frame1.axes.get_yaxis().set_ticks([])
        if save_models is True:
            all_models.append(model_spec)
        print ('----------')

    fig.add_axes([1.0,0.5,0.35,0.35])
    frame1=plt.gca()
    plt.title(r'Secondary spectrum of data')
    my_spec.plot_ss(new_fig=False, cb=False,vmin=vmin_ss, vmax=vmax_ss)#,vmin=7e6,vmax=2e8)
    plt.plot(my_spec.ss.fd,eta_load*(my_spec.ss.fd**2),'r',lw=2)
    
    fig.add_axes([1.4,0.5,0.35,0.35])
    frame1=plt.gca()
    my_spec.plot_ss(new_fig=False, cb=False,vmin=vmin_ss, vmax=vmax_ss)
    
    
    if pc_overlap is True:
        chunks_e=np.zeros((len(models_e_a),1,models_e_a[0].shape[0],models_e_a[0].shape[1]),dtype=complex)
        chunks_ds=np.zeros((len(models_ds_a),1,models_ds_a[0].shape[0],models_ds_a[0].shape[1]),dtype=float)
        
        for h in range(0,len(models_e_a)):
            chunks_e[h,0,:,:]=models_e_a[h]
            chunks_ds[h,0,:,:]=models_ds_a[h]
        full_ds=THTH.mosaic_ds(chunks_ds)    
        full_mE=THTH.mosaic(chunks_e)
        
        
        me_f= np.linspace(my_spec.f[0],my_spec.f[-1], full_mE.shape[0])
        me_t= my_spec.t
        fig.add_axes([0.6,0.006,0.25,0.986])
        frame1=plt.gca()
        plt.imshow(np.angle(full_mE), aspect='auto', origin='lower', cmap='seismic', interpolation='none')
        fig.add_axes([0.3,0.006,0.25,0.986])
        frame1=plt.gca()
        dsa.fun_plot_ds(full_ds.T,me_t, me_f, new_fig=False)
        frame1.axes.get_xaxis().set_ticks([])
        frame1.axes.get_yaxis().set_ticks([])
        
        npad=3
        full_mE_pad=np.pad(full_mE.T,((0,npad*full_mE.T.shape[0]),(0,npad*full_mE.T.shape[1])),mode='constant',
                        constant_values=full_mE.T.mean())
        full_mE_pad=full_mE_pad.T
        me_fd=THTH.fft_axis(me_t,u.mHz,npad)
        me_tau=THTH.fft_axis(me_f,u.us,npad)

        full_mEs=np.fft.fftshift(np.fft.fft2(full_mE_pad))
        fig.add_axes([1.8,0.5,0.35, 0.35])
        frame1=plt.gca()
        plt.title(r'recovered wavefield')
        mth.fun_plot_mes(np.abs(full_mEs)**2, me_fd, me_tau, new_fig=False, cb=True)
        full_models={'mE':full_mE, 't':me_t, 'f':me_f, 'mEs':full_mEs, 'fd':me_fd, 'tau':me_tau,
                    'mds':full_ds, 'mjd':my_spec.mjd.mjd.mean()}

    edges=np.linspace(-edge,edge,ntau)
    
    eta_thth=eta_ref*(ref_freq/me_f.mean())**2
    print ('eta_thth:', eta_thth)
    thth_red, thth2_red, recov, model, edges_red,w,V=THTH.modeler(full_models['mEs'],full_models['tau'],
                                                                  full_models['fd'],eta=eta_thth, edges=edges)

    fig.add_axes([0.95,0.01,0.4,0.4])
    frame1=plt.gca()
    vmin,vmax=np.percentile(np.abs(thth_red)**2, [10,100-1e-2])
    plt.title(r'$\theta$-$\theta$ of wavefield')
    plt.imshow(np.abs(thth_red)**2,norm=LogNorm(vmin=vmin, vmax=vmax))
    frame1.axes.get_xaxis().set_ticks([])
    frame1.axes.get_yaxis().set_ticks([])
    fig.add_axes([1.5,0.01,0.65, 0.35])
    frame1=plt.gca()
    plt.title('abs ( magnifications )^2')
    ang_sep=(edges_red[:-1]*my_spec.ss.fd.unit*(const.c/ref_freq)/(dveff_here*d_eff**0.5)).decompose()*u.rad
    magns=np.abs(V*w)**2
    
    thth_dic={'ang_sep':ang_sep.to(u.mas), 'magn':magns, 'fd':edges_red[:-1], 'mjd':my_spec.mjd.mjd.mean()}
    plt.plot(ang_sep.to(u.mas),magns)
    plt.xlabel(r'maybe $\Delta \theta$ (mas)')
    plt.xlim(-2.1, 2.1)
    if model_ev is True:
        plt.ylim(1e-4, 1e4)
    else:
        plt.ylim(1e4,1e12)
    plt.yscale('log')
    plt.savefig('load_model_%.2f_%s_eta%.1f_%s.png'%(my_spec.mjd.mjd.mean(),my_spec.tel,eta_load.value,saveauxname),
                format='png',bbox_inches='tight',dpi=90)
    plt.show()
    return (full_models, thth_dic)
