

from . import WoldModel, ModelBlackBoxVariational
from ..fitter import FitterSGD


class WoldModelBBVI(ModelBlackBoxVariational, WoldModel, FitterSGD):

    def fit(self, *args, **kwargs):
        super().fit(objective_func=self.bbvi_objective, *args, **kwargs)
