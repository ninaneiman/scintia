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







