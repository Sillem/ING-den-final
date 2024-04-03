import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import roc_auc_score

def gini(y_true, y_score):
    return 2*roc_auc_score(y_true, y_score)-1

class GiniSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold # threshold for column 

    def fit(self, X, y=None):
        self.col_to_drop = [ i for i in X.columns if gini(y, X[i]) <= self.threshold]
        return self

    def transform(self, X):
        # Tworzymy kopię, aby nie modyfikować oryginalnego DataFrame
        return X.copy(deep=True).drop(self.col_to_drop ,axis=1)
    
    def get_feature_names_out(self, input_features=None):
        input_features_set = set(input_features)
        col_to_drop_set = set(self.col_to_drop)
        return input_features_set - col_to_drop_set