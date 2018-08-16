import numpy as np 
import matplotlib.pyplot as plt 

from . import disk_model as DM


class FitCube():

    def init(self, datacube, lamdas, instrumental_resolution, seeing):

        self.datacube=datacube
        self.lamdas=lamdas
        self.inst_res=instrumental_resolution
        self.seeing=seeing


    def get_disk_model():

        self.model_cube=DM.get_disk_model(parameters)

        