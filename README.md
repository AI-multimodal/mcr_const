Prior Knowledge Embedding Constraints to Help the Deconvolution of Spectra Sequence 
using Multivariate Curve Resolution (MCR).

For a time resolved spectra sequence D, it deconvolutes to concentration time evolution 
profile C and individual spectra S<sup>T</sup> for each components.

D = CS<sup>T</sup>                            | 
:--------------------------------------------:|
![](images/equation/Equation.png) | 

# QuickStart
The notebooks in [Examples](./examples) demonstates the usage of the constraints, including a rank selectivity example [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AI-multimodal/mcr_const/blob/master/examples/rank_selectivity.ipynb), a smooth constraint example [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AI-multimodal/mcr_const/blob/master/examples/smooth_concentration.ipynb) and a more complex multimodal example [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AI-multimodal/mcr_const/blob/master/examples/TiZnK_XANES_MultiModal_MCR_Example.ipynb). 


# Installation
```console
git clone git@github.com:xhqu1981/mcr_const.git
cd mcr_const
conda create --name mcr_py3 python=3
source activate mcr_py3
python setup.py develop
```

# Usage
The constraints in this repository is designed to be compatible with [pyMCR](https://github.com/usnistgov/pyMCR) 
from NIST. Every MCR can be initiated in the same way they would do in pyMCR.

E.g. To decompose a XANES sequence, we can use pyMCR only code:
```python
from pymcr.mcr import McrAR
from pymcr.regressors import NNLS
from pymcr.constraints import ConstraintNorm

mcrar = McrAR(c_regr=NNLS(), st_regr=NNLS(), c_constraints=[ConstraintNorm()])
spec_guess = spectra_sequence[[5, 20, 40, -3]]
mcrar.fit(spectra_sequence, ST=spec_guess)
resolved_conc = mcrar.C_opt_
resolved_spec = mcrar.ST_opt_

# Make plots for visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure()
sns.set_palette('bright', resolved_conc.shape[0])
for i, conc in enumerate(resolved_conc.T):
    plt.plot(conc, label=f'X Material {i+1}')
plt.legend()
plt.yticks([])
title = "Resolved Concentration"
plt.title(title)
plt.savefig(f'{title}.pdf', dpi=300)

plt.figure()
colors = sns.color_palette('bright', true_specs.shape[0])
for i, (tspec, rspec, color) in enumerate(zip(true_specs, resolved_spec, colors)):
    plt.plot(tspec, c=color, lw=1.2, label=f'X Material {i+1}') # True Spectra
    plt.plot(rspec, "--", lw=2.0, c=color) # Resolved spectra
plt.legend()
plt.xticks([])
plt.yticks([])
title = "Resolved Spectra"
plt.title(title)
plt.savefig(f'{title}.pdf', dpi=300)
```
Resolved Concentration                             |  Resolved Spectra
:-------------------------------------------------:|:-------------------------------------------:
![](images/closure_4/Resolved%20Concentration.png) | ![](images/closure_4/Resolved%20Spectra.png)

For this specific dataset, the spectra is almost perfect (solid line are resolved value, dashed line 
are true value). However, the red and green concentration exhibit a wierd trend of first going to 
zero and then going up again. In addition, there are more than 2 phases in the same point which is 
against phase law. To solve this problem, we can apply the rank selectivity constraint.

### Rank Selectivity Constraint
This constraint is available via module ```mcr_const.constraints.nist``` in this repository and comes with a constructor 
for flexible targeting region definition using numpy advanced indexing. In parallel, we have provided a helper function 
```from_xxx()``` as a ```classmethod``` to make ease usage of the constraint for simple questions. Inspired by the 
previous result without rank selectivity, the phases might separate at 19 & 37 since the trend in concentration profile 
changes at these two points. We can construct a rank selectivity constraint with phase separators at these two points, 
assuming phase law holds for this dataset. The following code is an example on how to use this constraint:
```python
from pymcr.mcr import McrAR
from pymcr.regressors import NNLS
from pymcr.constraints import ConstraintNorm
from mcr_const import ConstraintPointBelow

rank_selectivity = ConstraintPointBelow.from_phase_law(
    n_species=4,
    sequence_length=spectra_sequence.shape[0],
    interface_positions=[19, 37],
    threshold=1.0E-5
)

mcrar = McrAR(c_regr=NNLS(), st_regr=NNLS(),  c_constraints=[ConstraintNorm(), rank_selectivity])
spec_guess = spectra_sequence[[5, 20, 40, -3]]
mcrar.fit(spectra_sequence, ST=spec_guess)
resolved_conc = mcrar.C_opt_
resolved_spec = mcrar.ST_opt_

# Make plots for visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure()
sns.set_palette('bright', resolved_conc.shape[0])
for i, conc in enumerate(resolved_conc.T):
    plt.plot(conc, label=f'X Material {i+1}')
plt.legend()
plt.yticks([])
title = "Resolved Concentration"
plt.title(title)
plt.savefig(f'{title}.pdf', dpi=300)


plt.figure()
colors = sns.color_palette('bright', true_specs.shape[0])
for i, (tspec, rspec, color) in enumerate(zip(true_specs, resolved_spec, colors)):
    plt.plot(tspec, c=color, lw=1.2, label=f'X Material {i+1}') # True Spectra
    plt.plot(rspec, "--", c=color, lw=2.0) # Resolved spectra
plt.legend()
plt.xticks([])
plt.yticks([])
title = "Resolved Spectra"
plt.title(title)
plt.savefig(f'{title}.pdf', dpi=300)
```
Resolved Concentration                             |  Resolved Spectra
:-------------------------------------------------:|:-------------------------------------------:
![](images/rank_selectivity/Resolved%20Concentration.png) | ![](images/rank_selectivity/Resolved%20Spectra.png)

It is intuitive that not only the concentrations conforms to physical law now, but also spectra quality shows 
improvement. The underlying ConstraintPointBelow class pushes concentration to zero at certain regions, which makes use of
the information obtained from prior knowledge.

### Smoothing Constraint
The smoothing constraint is designed to smooth noisy profile, for concentration or/and spectra. Similarly, it comes with
a helper function for simplified construction. It is worth note that we need to specify ```var_type``` parameter to
specify whether concentration or spectra is the target of constraint to be created:
```python
from pymcr.mcr import McrAR
from pymcr.regressors import NNLS
from mcr_const import ConstraintSmooth, VarType


smooth_conc_1 = ConstraintSmooth.from_range(
    i_specie=0, 
    i_range=(0, spectra_sequence.shape[0]),
    exponent=5,
    smoothing_factor=0.02
)
smooth_conc_2 = ConstraintSmooth.from_range(
    i_specie=1, 
    i_range=(0, spectra_sequence.shape[0]),
    exponent=5,
    smoothing_factor=0.02
)

smooth_spec_1 = ConstraintSmooth.from_range(
    i_specie=0, 
    i_range=(0, spectra_sequence.shape[1]),
    exponent=2,
    smoothing_factor=0.1,
    var_type=VarType.SPECTRA
)
smooth_spec_2 = ConstraintSmooth.from_range(
    i_specie=1, 
    i_range=(0, spectra_sequence.shape[1]),
    exponent=2,
    smoothing_factor=0.1,
    var_type=VarType.SPECTRA
)
mcrar = McrAR(c_regr=NNLS(), st_regr=NNLS(), tol_increase=1.0,
              c_constraints=[smooth_conc_1, smooth_conc_2],
              st_constraints=[smooth_spec_1, smooth_spec_2])
spec_guess = spectra_sequence[[5, -5]]
mcrar.fit(spectra_sequence, ST=spec_guess)
resolved_conc = mcrar.C_opt_
resolved_spec = mcrar.ST_opt_

# Make plots for visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure()
colors = sns.color_palette('bright', resolved_conc.shape[0])
for i, (conc, nc, color) in enumerate(zip(resolved_conc.T, noisy_conc.T, colors)):
    plt.plot(conc, c=color, label=f'X Material {i+1}')
    plt.plot(nc, lw=0.5, c=color)
plt.legend()
plt.yticks([])
title = "Resolved Concentration"
plt.title(title)
plt.savefig(f'{title}.pdf', dpi=300)

plt.figure()
colors = sns.color_palette('bright', resolved_spec.shape[0])
for i, (spec, ns, color) in enumerate(zip(resolved_spec, noisy_spec, colors)):
    plt.plot(spec, lw=1.2, c=color, label=f'X Material {i+1}')
    plt.plot(ns, lw=0.5, c=color)
plt.legend()
plt.xticks([])
plt.yticks([])
title = "Resolved Spectra"
plt.title(title)
plt.savefig(f'{title}.pdf', dpi=300)
```
Resolved Concentration                             |  Resolved Spectra
:-------------------------------------------------:|:-------------------------------------------:
![](images/smooth/Resolved%20Concentration.png) | ![](images/smooth/Resolved%20Spectra.png)


# Funding acknowledgement

This research is based upon work supported by the U.S. Department of Energy, Office of Science, Office Basic Energy Sciences, under Award Number FWP PS-030. This research used resources of the Center for Functional Nanomaterials (CFN), which is a U.S. Department of Energy Office of Science User Facility, at Brookhaven National Laboratory under Contract No. DE-SC0012704.

## Disclaimer

The Software resulted from work developed under a U.S. Government Contract No. DE-SC0012704 and are subject to the following terms: the U.S. Government is granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable worldwide license in this computer software and data to reproduce, prepare derivative works, and perform publicly and display publicly.

THE SOFTWARE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF ANY KIND. THE UNITED STATES, THE UNITED STATES DEPARTMENT OF ENERGY, AND THEIR EMPLOYEES: (1) DISCLAIM ANY WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT, (2) DO NOT ASSUME ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF THE SOFTWARE, (3) DO NOT REPRESENT THAT USE OF THE SOFTWARE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS, (4) DO NOT WARRANT THAT THE SOFTWARE WILL FUNCTION UNINTERRUPTED, THAT IT IS ERROR-FREE OR THAT ANY ERRORS WILL BE CORRECTED.

IN NO EVENT SHALL THE UNITED STATES, THE UNITED STATES DEPARTMENT OF ENERGY, OR THEIR EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, CONSEQUENTIAL, SPECIAL OR PUNITIVE DAMAGES OF ANY KIND OR NATURE RESULTING FROM EXERCISE OF THIS LICENSE AGREEMENT OR THE USE OF THE SOFTWARE.

# Developers
[Xiaohui Qu](mailto:xiaqu@bnl.gov), [Mingzhao Liu](mailto:mzliu@bnl.gov), [Mark Hybertsen](mailto:mhyberts@bnl.gov), [Ruoshui Li](mailto:rli1@bnl.gov), [Deyu Lu](mailto:dlu@bnl.gov), [Xuance Jiang](mailto:xuance.jiang@stonybrook.edu), [Dario Stacchiola](mailto:djs@bnl.gov) at BNL/CFN.

# Get help
Send an email to [Xiaohui Qu](mailto:xiaqu@bnl.gov) at BNL/CFN.

# Citation
1. Qu X, Yan D, Li R, Cen J, Zhou C, Zhang W, Lu D, Attenkofer K, Stacchiola DJ, Hybertsen MS,* Stavitski E,* Liu* M Resolving the Evolution of Atomic Layer-Deposited Thin-Film Growth by Continuous In Situ X-Ray Absorption Spectroscopy. Chemistry of Materials 2021, 33, 1740–1751. 
2. Li R, Jiang X, Zhou C, Topsakal M, Nykypanchuk D, Attenkofer K, Stacchiola D, Hybertsen MS, Stavitski E,* Qu X,* Lu D,* Liu M* Deciphering Phase Evolution in Complex Metal Oxide Thin Films via High-Throughput Materials Synthesis and Characterization. Nanotechnology 2023, 34, 125701.
