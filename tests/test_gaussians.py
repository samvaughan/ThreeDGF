import unittest
import numpy as np


from ThreeDGF import gaussians as G

class Test_Gaussian_functions(unittest.TestCase):

    def setUp(self):

        self.FWHM=5.0
        self.shape=(30, 30)
        #Not exactly identical to a logrebinned lamda array, but very close
        self.logLamdas=np.linspace(np.log(0.779999971389771), np.log(1.0898486661608342), 2048)

    def test_shape_of_seeing(self):

        seeing_disk=G.seeing(self.FWHM, self.shape)

        self.assertEqual(self.shape, seeing_disk.shape)

    def test_normalisation_of_seeing(self):

        seeing_disk=G.seeing(self.FWHM, self.shape)

        self.assertTrue(np.allclose(seeing_disk.sum(), [1.0]))

    def test_centre_of_seeing(self):

        seeing_disk=G.seeing(self.FWHM, self.shape)

        y, x=np.indices(self.shape)

        xbar=np.sum(x*seeing_disk)/np.sum(seeing_disk)
        ybar=np.sum(y*seeing_disk)/np.sum(seeing_disk)

        self.assertTrue(np.allclose(xbar, self.shape[0]/2.0))
        self.assertTrue(np.allclose(ybar, self.shape[0]/2.0))

    def test_normalisation_of_LSF(self):

        line_wave=np.median(np.exp(self.logLamdas))

        self.assertEqual(G.gaussian(self.logLamdas, line_wave, 2.51).sum(), 1.0)

    def test_centre_of_general_gaussian(self):


        xcen=18.0
        ycen=13.0
        light_params={'X':xcen,
                    'Y':ycen,
                    'ROTATION':np.pi/4.,
                    'XWIDTH':5.0,
                    'YWIDTH':1.0,
                    'OFFSET':0.0,
                    'Amp':1e-18
                    }

        y, x=np.indices(self.shape)
        light_profile=G.twoD_Gaussian(light_params, x, y)

        xbar=np.sum(x*light_profile)/np.sum(light_profile)
        ybar=np.sum(y*light_profile)/np.sum(light_profile)

        centres=np.array([xcen, ycen])
        derived_centres=np.array([xbar, ybar])

        #See if these are less than 0.05 pixels off
        self.assertTrue(np.all( (centres-derived_centres) < 0.05))

    def test_amplitude_of_Gaussian(self):

        xcen=18.0
        ycen=13.0
        amplitude=1e-18
        light_params={'X':xcen,
                    'Y':ycen,
                    'ROTATION':np.pi/4.,
                    'XWIDTH':5.0,
                    'YWIDTH':1.0,
                    'OFFSET':0.0,
                    'Amp':amplitude
                    }

        y, x=np.indices(self.shape)
        light_profile=G.twoD_Gaussian(light_params, x, y)


        self.assertEqual(light_profile.max(), amplitude)


    