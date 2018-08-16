import numpy as np 
from scipy import fftpack



def seeing(FWHM, shape):

    """
    A symmetric Gaussian seeing disk
    """

    y, x=np.indices(shape)

    sig=FWHM / (2*np.sqrt(2*np.log(2)))

    params={
    'X':shape[1]/2.,
    'Y':shape[0]/2.,
    'ROTATION':0.0,
    'XWIDTH':sig,
    'YWIDTH':sig,
    'OFFSET':0.0,
    'Amp':1.0
    }

    gaussian=twoD_Gaussian(params, x, y)

    return gaussian/gaussian.sum()


def twoD_Gaussian(params, X, Y):

    xo = params['X']
    yo = params['Y']
    theta = params['ROTATION']
    sigma_x = params['XWIDTH']
    sigma_y = params['YWIDTH']
    offset = params['OFFSET']
    amplitude = params['Amp']

    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    

    g = offset + amplitude*np.exp( - (a*((X-xo)**2) + 2*b*(X-xo)*(Y-yo) 
                            + c*((Y-yo)**2)))

    return g


def make_3d_PSF(FWHM_seeing, FWHM_LSF, shape_2D, logLamdas):

    line_wave=np.median(np.exp(logLamdas))

    PSF_image=seeing(FWHM_seeing, shape_2D)
    LSF_spectrum=gaussian(logLamdas, line_wave, FWHM_LSF, pixel=True)

    PSF_3d=np.rollaxis(PSF_image[..., None]*LSF_spectrum.squeeze(), -1)

    return PSF_3d
#From Michele Cappellari's ppxf

###############################################################################
# NAME:
#   GAUSSIAN
#
# MODIFICATION HISTORY:
#   V1.0.0: Written using analytic pixel integration.
#       Michele Cappellari, Oxford, 10 August 2016
#   V2.0.0: Define lines in frequency domain for a rigorous
#       convolution within pPXF at any sigma, including sigma=0.
#       Introduced `pixel` keyword for optional pixel convolution.
#       MC, Oxford, 26 May 2017

def gaussian(logLam_temp, line_wave, FWHM_gal, pixel=True):
    """
    Instrumental Gaussian line spread function (LSF), optionally integrated
    within the pixels. The function is normalized in such a way that
    
            line.sum(0) = 1
    
    When the LSF is not severey undersampled, and when pixel=False, the output
    of this function is nearly indistinguishable from a normalized Gaussian:
    
      x = (logLam_temp[:, None] - np.log(line_wave))/dx
      gauss = np.exp(-0.5*(x/xsig)**2)
      gauss /= np.sqrt(2*np.pi)*xsig

    However, to deal rigorously with the possibility of severe undersampling,
    this Gaussian is defined analytically in frequency domain and transformed
    numerically to time domain. This makes the convolution within pPXF exact
    to machine precision regardless of sigma (including sigma=0).
    
    :param logLam_temp: np.log(wavelength) in Angstrom
    :param line_wave: Vector of lines wavelength in Angstrom
    :param FWHM_gal: FWHM in Angstrom. This can be a scalar or the name of
        a function wich returns the instrumental FWHM for given wavelength.
        In this case the sigma returned by pPXF will be the intrinsic one,
        namely the one corrected for instrumental dispersion, in the same
        way as the stellar kinematics is returned.
      - To measure the *observed* dispersion, ignoring the instrumental
        dispersison, one can set FWHM_gal=0. In this case the Gaussian
        line templates reduce to Dirac delta functions. The sigma returned
        by pPXF will be the same one would measure by fitting a Gaussian
        to the observed spectrum (exept for the fact that this function
        accurately deals with pixel integration).
    :param pixel: set to True to perform integration over the pixels.
    :return: LSF computed for every logLam_temp

    """
    line_wave = np.asarray(line_wave)

    if callable(FWHM_gal):
        FWHM_gal = FWHM_gal(line_wave)

    n = logLam_temp.size
    npad = fftpack.next_fast_len(n)
    nl = npad//2 + 1  # Expected length of rfft

    dx = (logLam_temp[-1] - logLam_temp[0])/(n - 1)
    x0 = (np.log(line_wave) - logLam_temp[0])/dx
    xsig = FWHM_gal/2.355/line_wave/dx    # sigma in pixels units
    w = np.linspace(0, np.pi, nl)[:, None]

    # Gaussian with sigma=xsig and center=x0,
    # optionally convolved with an unitary pixel UnitBox[]
    # analytically defined in frequency domain
    # and numerically transformed to time domain
    rfft = np.exp(-0.5*(w*xsig)**2 - 1j*w*x0)
    if pixel:
        rfft *= np.sinc(w/(2*np.pi))
    line = np.fft.irfft(rfft, n=npad, axis=0)

    return line[:n, :]


