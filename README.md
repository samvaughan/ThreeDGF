# ThreeDGF- Three D Galaxy Fitting

[![Build Status](https://travis-ci.com/samvaughan/ThreeDGF.svg?token=egMGmTiKxkq3cq2CjC71&branch=master)](https://travis-ci.com/samvaughan/ThreeDGF)

Checklist:

* Take a model velocity field and sigma profile and create a cube of gaussians at appropriate wavelengths
* Multiply that cube by a light profile (which we won't vary during the fitting)
* Bin that cube in the same way as the data (so need the voronoi bins)
* Compare to the data!

ToDo:

* Continuum subtraction? Fit Legendre polynomials to the ratio of the models and the data before finding the goodness of fit
* Plotting- nice way to plot the results
* Write the fitting code. Pymc3 with NUTS? 
* Unittests. Add coverage with coveralls?