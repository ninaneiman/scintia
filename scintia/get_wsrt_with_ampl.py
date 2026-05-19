import psrchive
import numpy as np
import template_match
from template_match import rotate_phase, align_profile, convert_template
import matplotlib.pyplot as plt
from glob import glob
from os.path import join
#generating dynamic spectra from psr archive files. This one also includes extra pulsar phase arrays. 

templ='/export/astron/archibald/projects/triplesystem/processing/template-work/56412.76_GBT_1400.rmset.scrunch.sm'


#obs_dir='/data/archibald/0337+1715/obs/56412.76_GBT_1400/singlefrequency/zap_*.ar'
obs_dir='/data/archibald/0337+1715/obs/'

obs=['56071.30_WSRT_1400','56087.23_WSRT_1400','56087.33_WSRT_1400','56091.21_WSRT_1400','56091.35_WSRT_1400','56098.46_WSRT_1400','56098.59_WSRT_1400','56105.32_WSRT_1400','56105.41_WSRT_1400','56106.18_WSRT_1400','56110.21_WSRT_1400','56110.31_WSRT_1400']

obss=np.load('onetotwo_obs.npz')
#observations=[obss['name'][0]]
observations=obs
#observations=['56025.74_WSRT_1400', '56025.79_AO_1400','56039.69_GBT_1400', '56039.72_AO_1400']  
print observations
processing_name='tuned2'

T=psrchive.Archive_load(templ)
T.dedisperse()
T.pscrunch()
T.remove_baseline()

template=T.get_data()[0,0,0,:]

print (np.shape(T.get_data()[0,0,0,:]))


array=np.arange(10)
#plt.plot(array)
#plt.show()

print ('glob for loop:',sorted(glob(join(obs_dir,obs[0],processing_name,'zap_*.ar'))))

for obs in observations:
    newname=obs.split('.')[0]+obs.split('.')[1]
    #newname=obs
    tel=obs.split('_')[1]
    print (tel, newname)
    #obs=obs[:5]+'.'+obs[5:]
    print ('observation:',obs)
   
    if tel== 'WSRT':
        t_values = convert_template(T.get_data()[0,0,0,:], 512)

        t_values /= np.amax(t_values)
        print ('len of t_values', len(t_values)) 
        a = np.angle(np.fft.fft(t_values)[1])/(2*np.pi)
        
        t_values = template_match.rotate_phase(t_values, -a)
        t_phases = np.linspace(0,1,len(t_values),endpoint=False)
        
        #plt.plot(t_phases,t_values)
        #plt.show()
        
        
        start_mjd = None
        dynamic_spectrum = []
        noise_spectrum = []
        amps=[]
        
        for f in sorted(glob(join(obs_dir,obs,processing_name,'zap_*.ar'))):
            print (f)
            F = psrchive.Archive_load(f)
            F.pscrunch()
            F.dedisperse()
            F.remove_baseline()
            d = F.get_data()[:,0,:,:]
            w = F.get_weights()[:,:]
            if start_mjd is None:
                start_mjd = F.start_time().in_days()
            nchan = d.shape[1]
            bw = F.get_bandwidth()
            center_frequency = F.get_centre_frequency()
            
            mjd_end=F.end_time().in_days()
            mjd_start=F.start_time().in_days()

            full_time=(mjd_end-mjd_start)*(24.*3600.)
            ntbin=full_time/d.shape[0]
            a_t = (np.arange(d.shape[0]) * ntbin)
            a_f = np.linspace(center_frequency-bw/2,center_frequency+bw/2, d.shape[1])

            dprof = (d*w[...,None]).mean(axis=(0,1))#here used to be sum instead of mean
            #phase = template_match.align_profile(t_values, dprof)
            phase, amp, bg = template_match.align_scale_profile(t_values, dprof)

            tz = template_match.rotate_phase(t_values,phase)
            my_templ=np.copy(tz)
            my_data=np.copy(d)
            
            #tz -= tz.mean()
            #d -= d.mean(axis=-1, keepdims=True)

            tz_sc=tz*amp+bg
            variance=np.var(d-tz_sc, axis=-1, keepdims=True)

            j = np.sum(d*tz_sc, axis=-1)

            #j = np.ma.array(j)
            #j[w==0] = np.ma.masked
            #dynamic_spectrum.append(j)
            j = np.array(j)
            all_data=j[w!=0]
            j[w==0] = np.mean(all_data)
            dynamic_spectrum.append(j)
            
            ns = np.sqrt(np.sum(variance*tz_sc**2, axis=-1))
            all_noise=ns[w!=0]
            ns[w==0] = np.mean(all_noise)
            noise_spectrum.append(ns)
            print ('this is shape of noise:',np.shape(ns))
            print ('and this is shape pf data:', np.shape(j))
 
        end_mjd = F.end_time().in_days()
 
        ds = np.concatenate(dynamic_spectrum)
        noise_sp=np.concatenate(noise_spectrum)
        #vmin,vmax = np.percentile(ds,[1,99])
        #plt.imshow(ds.T,extent=(0,(end_mjd-start_mjd)*24,center_frequency-bw/2,center_frequency+bw/2),vmin=vmin, vmax=vmax, aspect='auto')
        #plt.xlabel("t (hours)")
        #plt.ylabel("f (MHz)")
        #plt.show()
 
        np.savez(newname+'_wnoise', ds=np.array(ds), mjd=np.array([start_mjd,end_mjd]),c_fr=center_frequency, bw_fr=bw, noise=np.array(noise_sp), templ=my_templ, dprof=dprof, phase=phase, amp=amp, bg=bg)

#
