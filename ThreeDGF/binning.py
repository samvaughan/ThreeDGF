import numpy as np



def bin_cube(X, Y, bins, modelcube):

    """
    Take a list of bins for each spaxel in a datacube and bin a model cube the same way. The spectra in each bin are *summed*, not averaged or medianed. This should be the same as the data are binned!

    Args:
        x (array_like): x position of each spaxel
        y (array_like): y position of each spaxel
        bins (array_like): the bin number of spaxel (x, y)
        modelcube (array): the cube of model spectra. Must be wavelength axis first
        
    Returns:
        array: a 2D array of shape (n_lamdas, n_unique_bins) of spectra
    """

    assert len(x)==len(y)==len(bins), 'The lists X, Y and bins must be the same length!'

    n_lamdas, ny, nx=modelcube.shape
    assert (n_lamdas>nx) & (n_lamdas>ny), 'The wavelength axis must be first. Change this if you have a very large cube, where n_lamdas<nx or ny!'
    
    assert len(bins)==ny*nx, 'We must have the same number of bins as pixels in the datacube'

    spectra=np.empty(n_lamdas, len(np.unique(bins)))

    for i, b in enumerate(np.unique(bins)):

        bin_mask=bins==b
        spec=modelcube[:, Y[bin_mask], X[bin_mask]].sum(axis=1).sum(axis=1)

        spectra[:, i]=spec

    return spectra

