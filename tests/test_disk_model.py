import unittest
import numpy as np
import scipy.constants as const

from ThreeDGF import disk_model as DM
 

class Test_Disk_Model(unittest.TestCase):

    def setUp(self):

        self.params={'PA':45.0, 'xc':13.0, 'yc':17.0, 'v0':20.0, 'log_r0':1.0, 'log_s0' :10.0, 'theta':45.0}
        self.shape=(30, 30)
        self.velfield=DM.velfield(self.params, self.shape, 1)


    def test_rotate_coordinates_xy(self):

        x0=1
        y0=0
        theta=np.pi/2.0

        self.assertTrue(np.allclose(DM.rotate_coordinates(x0, y0, theta), (0, 1)))

    def test_rotate_coordinates_arrays_shape(self):

        x0=np.ones(10)
        y0=np.zeros(10)
        theta=np.pi/2.0

        X, Y=DM.rotate_coordinates(x0, y0, theta)

        self.assertEqual(X.shape, x0.shape)
        self.assertEqual(Y.shape, y0.shape)

    def test_velfield_fails_for_float_oversample(self):


        self.assertRaises(AssertionError, DM.velfield, *(self.params, (30, 30), 2.0))

    def test_velfield_oversamples_correctly(self):

        oversample=10
        oversampled=DM.velfield(self.params, (30, 30), oversample)
        self.assertEqual(self.velfield.size*oversample*oversample, oversampled.size)

    def test_map_centre_is_V0(self):

        xc=int(self.params['xc'])
        yc=int(self.params['yc'])
        self.assertTrue(np.allclose(self.velfield[yc, xc], self.params['v0']))

    def test_shift_moves_centre_of_map(self):

        #Check that shifting the map moves our xc, yc pixel to where it should be

        shift=[-4, -4]
        shifted=DM.shift_rotate_velfield(self.velfield, shift=shift ,PA=0.0)


        new_xc=int(self.params['xc'])-shift[0]
        new_yc=int(self.params['yc'])-shift[1]



        self.assertTrue(np.allclose(shifted[new_yc, new_xc], self.params['v0']))


class Test_Model_Cube(unittest.TestCase):

    def setUp(self):

        self.params={'PA':45.0, 'xc':13.0, 'yc':17.0, 'v0':20.0, 'log_r0':1.0, 'log_s0' :10.0, 'theta':45.0}
        self.shape=(30, 30)
        self.velfield=DM.velfield(self.params, self.shape, 1)

        self.lam0=0.8 #microns

        self.logLamdas=np.linspace(np.log(0.779999971389771), np.log(1.0898486661608342), 2048) #not quite ppxf logLam but close enough


        self.model_cube=DM._make_velocity_cube(self.velfield, self.lam0, self.logLamdas)


    # def test_vel_shifting_in_model(self):

    #     dv=(self.logLamdas[1]-self.logLamdas[0])*const.c/1000.0

    #     spec=self.model_cube[:, 0, 0]

    #     peak_wavelength=np.exp(logLamdas[np.argmax(model_cube[:, 0, 0])])

    #     #Shift in pixels of the peak
    #     npix_shift=self.velfield[0, 0]/dv

    #     self.assertTrue(lam0+dv*1000.0/const.c)