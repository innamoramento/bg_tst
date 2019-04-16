from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Module for data transformation
    and
    Feature Engeneering
    """

    def __init__(self):
        pass


    def transform(self, data):
        """Method for creating new features and necessary preprocessing"""
        # for example:
        # data['symmetry_se_log'] = np.log(data['symmetry_se'] + 1)

        return data.as_matrix()


    def fit(self, data, y=None, **fit_params):

        return self

