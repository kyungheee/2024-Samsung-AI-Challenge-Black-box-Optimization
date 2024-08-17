from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class BimodalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.log1p(X) # 로그 변환 수행