from . import WoldModel, ModelBlackBoxVariational
from ..fitter import FitterVariationalEM


class WoldModelBBVI(ModelBlackBoxVariational, WoldModel, FitterVariationalEM):

    def fit(self, *args, **kwargs):
        return super().fit(objective_func=self.bbvi_objective, *args, **kwargs)
