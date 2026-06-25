# psr_tools
some tools to manage pulsar scintillometry data and models


## Installation guide:

```
git clone git@github.com:ninaneiman/scintia.git
cd scintia
pip install -e .
```
## Brief description of functions and utils

Load_data.py  - functions to manipulate observed data, mostly to load data from .npz to useful formats

ds_psr.py - functions to manipulate dynamic spectra. In contains class "Spec" and "SecSpec"

fit_thth.py - functions to fit curvatures to dynamic spectra (Based on Daniel Baker's version on scintools https://github.com/DanielTBaker/scintools/tree/master/scintools)

models_thth.py - functions to manipulate models of dynamic spectra (including electric field). Contains "Model" class.
