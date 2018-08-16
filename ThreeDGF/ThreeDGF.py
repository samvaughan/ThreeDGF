import numpy as np 
import matplotlib.pyplot as plt 
from astropy.io import fits

from . import disk_model as DM


class FitCube():

    def init(self, datacube, lamdas, instrumental_resolution, seeing):

        self.datacube=datacube
        self.lamdas=lamdas
        self.inst_res=instrumental_resolution
        self.seeing=seeing

    def load_bins(self, x, y, bins):

        if not len(x)==len(y)==len(bins):
            raise ValueError('x, y and bin lists must be the same length')
        if not self.datacube.shape[1]*self.datacube.shape[2] == len(bins):
            raise ValueError('Must have the same number of bins as pixels')

        self.x_coords_1d=x
        self.y_coords_1d=y
        self.bins_1d=bins


    def get_disk_model():

        self.starting_disk_parameters={'PA':45.0, 'xc':13.0, 'yc':17.0, 'v0':20.0, 'log_r0':1.0, 'log_s0' :10.0, 'theta':45.0}

        self.model_vfield=DM.get_disk_model(self.starting_disk_parameters)

        