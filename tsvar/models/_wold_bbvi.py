import torch

from . import WoldModel, ModelBlackBoxVariational
from ..fitter import FitterVariationalEM


class WoldModelBBVI(ModelBlackBoxVariational, WoldModel, FitterVariationalEM):

    def fit(self, *args, **kwargs):
        super().fit(objective_func=self.bbvi_objective, *args, **kwargs)
