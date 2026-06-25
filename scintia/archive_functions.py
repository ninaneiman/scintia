import psrchive
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.linalg
from numpy.fft import rfft, irfft, fft, ifft

from scipy.optimize import curve_fit
from . import ds_psr as dsa
from . import load_data as ld

import astropy
from astropy import units as u, constants as const

def rotate_phase(prof, phase):
    """Rotate phase of profile earlier"""
    fprof = fft(prof)
    # top coefficient is special in even-length real FFTs
    # save it so phase shifting is non-destructive
    #topco = fprof[-1]
    n = len(fprof)
    fprof[:n//2] *= np.exp(2.j*np.pi*np.arange(n//2)*phase)
    fprof[-(n//2)+1:] *= np.exp(2.j*np.pi*np.arange(-(n//2)+1,0)*phase)
    #fprof[-1] = topco
    return ifft(fprof).real

def align_profile(template, prof):
    """Use cross-correlation to align template optimally with prof

    Return phase so that prof is approximately equal to
    rotate_phase(template,phase)*amp + bg
    (actually this should be a least-squares minimization).

    Note that swapping template and prof simply changes the sign of
    the resulting phase; the code is otherwise symmetrical.

    The code requires the template to have the same length as the
    profile.

    FIXME: can fail if the shift is exactly a half-bin.
    """
    ftemplate = fft(template)
    fprof = fft(prof)
    fcorr = ftemplate*np.conj(fprof)
    fcorr[0] = 0 # Ignore the constant
    fcorr[len(fcorr)//2] = 0 # Ignore the annoying middle component
    corr = ifft(fcorr)
    i = np.argmax(np.abs(corr))
    iphase = float(i)/len(corr)
    n = len(fcorr)
    def peak(p):
        return -np.abs(np.sum(fcorr[1:n//2]
                                  *np.exp(2.j*np.pi*np.arange(1,n//2)*p))
                 +np.sum(fcorr[-(n//2)+1:]
                             *np.exp(2.j*np.pi*np.arange(-(n//2)+1,0)*p)))
    r = scipy.optimize.minimize_scalar(peak,
                    bracket=(iphase-2./len(corr),
                             iphase,
                             iphase+2./len(corr)))
    phase = (r.x+0.5)%1-0.5
    return phase

def align_scale_profile(template, prof):
    """Use cross-correlation to align template optimally with prof

    Return phase, amp, bg so that prof is approximately equal to
    rotate_phase(template,phase)*amp + bg
    (actually this should be a least-squares minimization).
    """
    phase = align_profile(template, prof)
    rtemp = rotate_phase(template,phase)
    tz = rtemp - np.mean(rtemp)
    pz = prof - np.mean(prof)
    amp = np.dot(tz, pz)/np.dot(tz,tz)
    bg = np.mean(prof)-np.mean(rtemp)*amp
    return phase, amp, bg



def get_all_pars(f):
    print (f)
    psr_name=f.split('/')[4]
    F = psrchive.Archive_load(f)
    F.pscrunch()
    F.dedisperse()
    F.remove_baseline()
    d = F.get_data()[:,0,:,:]
    w = F.get_weights()[:,:]

    mjd_end=F.end_time().in_days()
    mjd_start=F.start_time().in_days()
    nchan = d.shape[1]
    bw = F.get_bandwidth()
    center_frequency = F.get_centre_frequency()
    print (center_frequency, bw, d.shape)
    
    full_time=(mjd_end-mjd_start)*(24.*3600.)
    ntbin=full_time/d.shape[0]
    a_t = (np.arange(d.shape[0]) * ntbin * u.s)
    a_f = np.linspace(center_frequency-bw/2,center_frequency+bw/2, d.shape[1])*u.MHz

    wdata=d*(w[...,None])
    #print (wdata)

    portrait=wdata.mean(axis=0)
    dprof = wdata.mean(axis=(0,1))

    template=dprof
    t_values = template/np.amax(template)
    t_phases = np.linspace(0,1,len(t_values),endpoint=False)

    phase, amp, bg = align_scale_profile(t_values, dprof)# t_values is template profile (1-D array)
    tz = rotate_phase(t_values,phase)
    tz -= tz.mean()
    d -= d.mean(axis=-1, keepdims=True)
    tz_sc=tz*amp+bg
    variance=np.var(d-tz_sc, axis=-1, keepdims=True)
    j = np.sum(d*tz_sc, axis=-1)
    j = np.array(j)
    all_data=j[w!=0]
    j[w==0] = np.mean(all_data)

    ns = np.sqrt(np.sum(variance*tz_sc**2, axis=-1))
    all_noise=ns[w!=0]
    ns[w==0] = np.mean(all_noise)

    if j.shape[0] <2:
        j=np.concatenate((j,j*0.9), axis=0)
        ns = np.concatenate((ns,ns*0.9), axis=0)
        a_t=np.arange(j.shape[0]) * ntbin * u.s
        print ('this spec has only one time bin, appending extra bin to be able to plot ss and acf')

    spec=dsa.Spec(I=j, t=a_t, f=a_f, stend=np.array([mjd_start,mjd_end+a_t[-1].to(u.d).value]),
                  nI=ns, tel='Nancay', psr=psr_name, pad_it=True, npad=3, ns_info='with noise')

    DM=F.get_dispersion_measure()
    my_coord=F.get_coordinates()
    radec=my_coord.getRaDec()
    dec=radec.angle2.getDegrees()
    ra=radec.angle1.getDegrees()
    coord=np.array([ra,dec])
    psr_name=F.get_source()

    return portrait, spec, DM, coord, psr_name

def plot_portrait(portrait, spec):
    vmin,vmax=np.percentile(portrait, [0.1,99.9])
    plt.imshow(portrait, vmin=vmin,vmax=vmax, origin='lower', aspect='auto',
              extent=[0,1,spec.f[0].value, spec.f[-1].value])
    plt.xlim(0.2,0.8)
    plt.ylabel('freq. (MHz)')
    plt.xlabel('Pulse phase')

def shift_to_middle(portrait):
    profile=np.mean(portrait, axis=0)
    peak_arg=np.argmax(profile)
    print (peak_arg)
    shift=int(portrait.shape[1]/2)-peak_arg
    return np.roll(portrait, shift, axis=1)


