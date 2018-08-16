import numpy as np 
from scipy.special import iv, kv
import scipy.interpolate as si

def interpolators_for_bessel_functions():

    #The Bessel Function part is sloooooow. Interpolate a look up table instead of calling iv and kv each time
    #I've tested this against the true arrays of I0K0 and I1K1. The median difference is 10^(-10) and 10^(-12) 
    #respectively. Largest difference is 10^-9.

    x=np.logspace(-4.0, 2.0, 1000000)
    
    I0K0 = iv(0,x)*kv(0,x)
    I1K1 = iv(1,x)*kv(1,x)

    interpI0K0 = si.interp1d(x, I0K0, bounds_error=True, assume_sorted=True)
    interpI1K1 = si.interp1d(x, I1K1, bounds_error=True, assume_sorted=True)

    return interpI0K0, interpI1K1

interpI0K0, interpI1K1=interpolators_for_bessel_functions()


#Settings
oversample=5
seeing=0.5 #In arcseconds

#maximum shift of the centre of our map in pixels
#Making this too large slows down the velfield function. Taking 5 instead of 30 speeds it up by a factor of 4
max_centre_shift=5

#The fraction of the central peak at which we trim away the outskirts of the cube. Any values less than fraction_of_peak*peak_lightprofile_value are 
#excluded from the kinematic fitting
fraction_of_peak=0.1

