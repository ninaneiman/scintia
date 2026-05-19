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

def shrink_any_2(array, factor=[1,1]):
    '''downsamples 2D-data
    It takes 2D array of any size, finds the closes factorable number to desired number of averaged pixels
    and creates new array factor of times averaged.
    
    it throws away pixels that is mod of array.shape[i]/factor[i]
    takes: 2D-array (array) and factors (factor) by which to downsample that array in each axis
    factor -(int) number of pixels to average
    returns: downsampled 2D array'''
    factor=np.array(factor, dtype=int)
    
    size0=array.shape[0]//factor[0]
    size1=array.shape[1]//factor[1]
    
    new_shape0=array.shape[0]-array.shape[0]%factor[0]
    new_shape1=array.shape[1]-array.shape[1]%factor[1]

    new_data=np.zeros((size0,array.shape[1]))
    for i in range(0,new_shape0,factor[0]):
        k=i//factor[0]
        new_data[k,:]=np.mean(array[i:i+factor[0],:],axis=0)
    array=new_data

    nnew_data=np.zeros((array.shape[0],size1))
    for i in range(0,new_shape1,factor[1]):
        k=i//factor[1]
        nnew_data[:,k]=np.mean(array[:,i:i+factor[1]],axis=1)
    array=nnew_data
    return array


def shrink_any_1(axis, factor):
    '''Downsamples 1-D array same way as shrink_any_2'''
    factor=int(factor)
    new_len=len(axis)-len(axis)%factor
    new_size=len(axis)//factor
    cr_axis=axis[:new_len]
    new_axis = np.linspace(np.amin(cr_axis),np.amax(cr_axis), new_size)
    return new_axis


def load_triple(filenpz='/mnt/scratch-lustre/gusinskaia/triple_system/5602579_AO_1400_ds.npz',factor=[1,1], mean0=True, wnoise=True):
    '''loads ds data from npz file, downsamples it, subtructs mean (if applied) and loads noise data if present (and applied)
    Takes:
    filenpz - name of the npz file to load data from
    factor - [#int, #int] 
    factor difines the number by which to downsample 2D ds data (see shrink_2 function) for 2 axis (separately); 0-time, 1-freq.
    mean0 - set mean of the data to 0 (i.e. whether to subtract mean)
    wnoise - whether to load noise data (which is present or not)
    Returns:
    ds - dynamic spectra (2D array)
    t - time axis (astropy units quantity)
    f - frequency axis (astropy units quantity)
    ns - noise of dynamic spectra (2D array)'''
    triple_ds=np.load(filenpz)
    print (triple_ds['ds'].shape)
    ds=triple_ds['ds']
    if wnoise is True:
        ns=triple_ds['noise']
    else:
        ns=np.random.normal(size=np.shape(ds))*np.std(ds)/6
    if 'WSRT' in filenpz:
        ds=shrink_any_2(ds, factor=factor)
        ns=shrink_any_2(ns, factor=factor)/np.sqrt(factor[0]*factor[1])
    if mean0 is True:
        ds=ds-ds.mean()
    end_mjd=triple_ds['mjd'][1]
    start_mjd=triple_ds['mjd'][0]
    center_frequency=triple_ds['c_fr']
    bw=triple_ds['bw_fr']

    full_time=(end_mjd-start_mjd)*(24.*3600.)
    ntbin=full_time/ds.shape[0]
    a_t = (np.arange(ds.shape[0]) * ntbin * u.s)
    if 'AO' in filenpz:
        a_f = np.linspace(center_frequency+bw/2,center_frequency-bw/2, ds.shape[1]) * u.MHz
        ds=np.flip(ds, axis=1)
        ns=np.flip(ns, axis=1)
    else:
        a_f = np.linspace(center_frequency-bw/2,center_frequency+bw/2, ds.shape[1]) * u.MHz
    return ds, a_t, a_f, ns
