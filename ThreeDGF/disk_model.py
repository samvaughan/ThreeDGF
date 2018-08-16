import numpy as np
from . import settings
import scipy.ndimage as ndi
import scipy.constants as const
#import matplotlib.pyplot as plt 


from . import gaussians as G

def velfield(params, shape, oversample=1):

    """
    Make a 2d array containing a velocity field. This Velocity field can be  oversampled compared to the KMOS spaxel resolution.
    
    * Take a set of X,Y coordinates. These are _larger_ than the data we're trying to fit- we pad them such that we can shift the velocity map
    to the centre at the end.
    * Rotate these coordinates by angle PA degrees. 
    * Make a velocity map in terms of these rotated coordinates, with the centre at data.shape/2! NOT at the required xc, yc coordinates yet.
    * Finally, shift using ndi.shift
    * Crop away the extra padded values

    Args:
        params (dict): A dictionary containing the keys 'PA' (position angle
            of velocity map), 'xc' and 'yc' (map centres), 'theta' (intrinsic inclination of disk), 'log_r0' and 'log_s0' (disk scale and surface brightness parameters) and 'v0' (velocity offset of central pixel)
        shape (tuple): The maximum x and y values of the disk. Note that a 
            shape of (30, 30) can be oversampled to larger than (30, 30) output

        oversample (int, optional): Integer amount to oversample the array by.

    Returns:
        array: An array containing the velocity map, of shape (shape)*oversample

    """

    #This is the 'angular eccentricity'
    #Shapes the flattening of the elliptical coordinates
    #cos(theta) is just b/a for the ellipse
    #sin(theta) is sqrt(1-b**2/a**2), or the eccentricity e
    #Should limit a=5b for reasonable galaxies

    assert type(oversample)==int, 'Oversample must be an integer'


    PA=params['PA']
    xc=params['xc']
    yc=params['yc']
    v0=params['v0']
    PA_rad=PA*np.pi/180.


    #Get coordinate axes
    max_shift=settings.max_centre_shift
    ys=np.linspace(0-max_shift, shape[0]+max_shift, oversample*(shape[0]+2*max_shift))
    xs=np.linspace(0-max_shift, shape[1]+max_shift, oversample*(shape[1]+2*max_shift))
    X_0, Y_0=np.meshgrid(xs, ys)


    #Shift things to the centre, rotate them by PA, then shift back
    centre_x=shape[0]/2.0
    centre_y=shape[1]/2.0

    X_r, Y_r=rotate_coordinates(X_0-centre_x, Y_0-centre_y, PA_rad)

    X=X_r+centre_x
    Y=Y_r+centre_y

    #Intrinisc viewing angle of the disk
    theta=params['theta']
    theta_rad=theta*np.pi/180.

    

    #Get the simple axisymetric velfield, then scale by (X-centre_of_array)/R)
    R = np.sqrt((X-centre_x)**2 + ((Y-centre_y)/np.cos(theta_rad))**2)
    velfield= v_circ_exp_quick(R, params)*(X-centre_x)/(R*np.sin(theta_rad))

    #Shift the velfield to where it should be
    shift=np.array([yc-centre_y, xc-centre_x])*oversample
    velfield=ndi.shift(velfield, shift, order=0, mode='nearest', cval=np.nan)

    #Crop away everything which we don't need- larger than the original data
    m_x=(X_0>0)&(X_0<shape[1])
    m_y=(Y_0>0)&(Y_0<shape[0])
    m_t=(m_x)&(m_y)

   

    velfield_final=velfield[m_t].reshape(oversample*shape[0], oversample*shape[1])+v0

    # import matplotlib.pyplot as plt 
    # import ipdb; ipdb.set_trace()

    return velfield_final


def shift_rotate_velfield(velfield, shift, PA,**kwargs):

    """
    Take a velocity field and shift by some offset and rotate by an angle, using the scipy ndimage library

    Args:
        velfield (array): Velocity field to shift
        shift (array-like): two component array containing [y_shift, x_shift]. 
            Note the order! 
        PA (float): Angle by which to rotate
        **kwargs: keyword arguments to pass to ndimage.rotate and 
            ndimage.shift. Defaults are order=0, reshape=False, mode='constant' and cval=np.nan
    Returns: 
        array_like: Shifted, rotatated velocity field
    """

    order=kwargs.pop('order', 0)
    reshape=kwargs.pop('reshape', False)
    mode=kwargs.pop('mode', 'constant')
    cval=kwargs.pop('cval', np.nan)


    #Rotate to the right PA
    velfield_shifted = ndi.shift(velfield, shift, order=order, mode=mode, cval=cval)
    velfield_final = ndi.rotate(velfield_shifted,PA, order=order, mode=mode, cval=cval, reshape=reshape, **kwargs)
    

    return velfield_final




def v_circ_exp_quick(R,params):

    """
    Make a rotation curve, following:
    # exponential disk model velocity curve (Freeman 1970; Equation 10)
    # v^2 = R^2*!PI*G*nu0*a*(I0K0-I1K1)
    """
    
   

    # G

    G = 6.67408e-11 #m*kg^-1*(m/s)^2
    G = G*1.989e30  #m*Msol^-1*(m/s)^2
    G = G/3.0857e19 #kpc*Msol^-1(m/s)^2
    G = G/1000./1000.

    

    # parameters
    log_R0=params['log_r0']
    R0 = 10**log_R0
    log_s0=params['log_s0']
    s0  = 10**log_s0
    # evaluate bessel functions (evaluated at 0.5aR; see Freeman70)

    half_a_R=(0.5*(R)/R0)

    #temp[temp>709.]=709.

    #Bessel Functions
    #Interpolate to speed up!
    I0K0 = settings.interpI0K0(half_a_R)
    I1K1 = settings.interpI1K1(half_a_R)

    #bsl  = I0K0 - I1K1

    

    # velocity curve
    V_squared  =  R*((np.pi*G*s0)*(I0K0 - I1K1)/R0)
    V=np.sqrt(V_squared)   

    return V


def rotate_coordinates(x, y, theta):

    """
    Rotate a series of x, y coordinates using a rotation matrix
    """

    X=np.cos(theta)*x-np.sin(theta)*y
    Y=np.sin(theta)*x+np.cos(theta)*y

    return X, Y



def _make_velocity_cube(velfield, sigma_profile, lam0, logLamdas):

    """
    Turn a 2D velocity map into a 3D cube
    """

    
    vel_cube=np.outer(np.ones_like(logLamdas), velfield).reshape(-1, *velfield.shape)

    sig_cube=np.outer(np.ones_like(logLamdas), sigma_profile).reshape(-1, *velfield.shape)

    vels=(logLamdas*const.c/1000.0)-np.log(lam0)*const.c/1000.0
    v_start=vels[0]
    v_stop=vels[-1]
    dv_kms=(logLamdas[1]-logLamdas[0])*const.c/1000.0

    # Create grid of velocities
    vgrid, y, x = np.mgrid[
        v_start: v_stop + 0.9 * dv_kms: dv_kms,
        0: np.shape(velfield)[0],
        0: np.shape(velfield)[1]
    ]


    vel_cube= np.exp(-0.5 * (vgrid - vel_cube) ** 2 / (scube ** 2))

    return vel_cube

def _make_gaussian_light_profile(light_params, shape, oversample=1):

    """
    Make a simple Gaussian light profile
    """


    y, x=np.indices(shape)
    light_profile=G.twoD_Gaussian(light_params, x, y)

    return light_profile


def make_deconvolved_model(params, shape, oversample, Ha_lam, logLamdas, light_profile):

    vfield=velfield(params, shape, oversample)
    model=_make_velocity_cube(vfield, Ha_lam, logLamdas)

    deconvolved_model=model*light_profile


    return deconvolved_model