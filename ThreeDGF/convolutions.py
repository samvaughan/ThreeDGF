import numpy as np 
from numpy.fft import rfftn, fftshift, irfftn
from numpy.fft import rfft, fftshift, irfft


def convolve_3d_same(cube, psf, compute_fourier=True):
    """
    Convolve a 3D cube with PSF & LSF.
    PSF can be the PSF data or its Fourier transform.
    if compute_fourier then compute the fft transform of the PSF.
    if False then assumes that the fft is given.

    This convolution has edge effects (and is slower when using numpy than pyfftw).

    cube: The cube we want to convolve
    psf: The Point Spread Function or its Fast Fourier Transform
    """

    size = np.array(np.shape(cube)[slice(0, 3)])

    #import ipdb; ipdb.set_trace()

    if compute_fourier:
        fft_psf = rfftn(psf, axes=[0, 1, 2])
    else:
        fft_psf = psf

    fft_img = rfftn(cube,  axes=[0, 1, 2])

    # Convolution
    #fft_cube = np.real(fftshift(irfftn(fft_img * fft_psf, size=size, axes=[0, 1, 2]), axes=[0, 1, 2]))

    convolved_cube = np.real(fftshift(irfftn(fft_img * fft_psf, axes=[0, 1, 2]), axes=[0, 1, 2]))


    return convolved_cube, fft_psf, fft_img


def convolve_1d_same(spec, LSF, compute_fourier=True):
    """
    Convolve a 3D cube with PSF & LSF.
    PSF can be the PSF data or its Fourier transform.
    if compute_fourier then compute the fft transform of the PSF.
    if False then assumes that the fft is given.

    This convolution has edge effects (and is slower when using numpy than pyfftw).

    cube: The cube we want to convolve
    psf: The Point Spread Function or its Fast Fourier Transform
    """

    #size = np.array(np.shape(cube)[slice(0, 3)])

    #import ipdb; ipdb.set_trace()

    if compute_fourier:
        fft_LSF = rfft(LSF)
    else:
        fft_LSF = LSF

    fft_spec = rfft(spec)

    # Convolution
    #fft_cube = np.real(fftshift(irfftn(fft_img * fft_LSF, size=size)))

    convolved_spec = np.real(fftshift(irfft(fft_spec * fft_LSF)))


    return convolved_spec, fft_LSF, fft_spec