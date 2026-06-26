# psr_tools
Useful tools to manage pulsar scintillometry data and models


## Installation guide:

```
git clone git@github.com:ninaneiman/scintia.git
cd scintia
pip install -e .
```
## Brief description of functions and utils

ds_psr.py - functions to manipulate dynamic spectra. In contains classes: "Spec" and "SecSpec"

fit_thth.py - functions to fit curvatures to secondary spectra products (Based on Daniel Baker's version on scintools https://github.com/DanielTBaker/scintools/tree/master/scintools)

models_thth.py - functions to manipulate models of dynamic and secondary spectra (e.g. electric field). Contains "Model" class.

scint_scales.py - functions to measure scintillation bandwidth and timescale from the dynamic spectrum or pulse portrait

archive_functions.py - functions to produce dynamic spectrum from raw pulsar archive files

Load_data.py  - functions to manipulate observed data, mostly to load dynamic spectra data from .npz to useful formats

gbt_fits.py and wsrt_fits.py - functions to simplify fitting curvatures into GBT and WSRT data specifially
