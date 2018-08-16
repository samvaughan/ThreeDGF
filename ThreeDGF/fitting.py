import numpy as np 

from . import disk_model as DM, convolutions as C, binning as B 



def make_final_model(disk_params, shape, oversample, Ha_lam, logLamdas, light_profile, PSF_FFT, bins, x, y):


    deconvolved_model=DM.make_deconvolved_model(disk_params, shape, oversample, Ha_lam, logLamdas, light_profile)

    convolved_model, _, _=C.convolve_3d_same(cube, psf, compute_fourier=False)

    binned_convolved_model=B.bin_cube(x, y, bins, convolved_cube)

    return binned_convolved_model