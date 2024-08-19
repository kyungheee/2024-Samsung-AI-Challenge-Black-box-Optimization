import pandas as pd
import numpy as np
from imbens.sampler import RandomOverSampler

import pandas as pd
import numpy as np
from imbens.sampler import RandomUnderSampler

def undersample_small_y(df, target_col='y', quantile=0.90, sampling_strategy=0.5):
    """
    Undersamples the smaller y values based on a specified quantile for regression.

    Parameters:
    df (pd.DataFrame): Input dataframe containing features and target.
    target_col (str): Name of the target column. Default is 'y'.
    quantile (float): Quantile below which y values will be undersampled. Default is 0.25.
    sampling_strategy (float): Ratio of the number of samples after resampling. Default is 0.5.

    Returns:
    pd.DataFrame: Dataframe with undersampled small y values and original large y values.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Define the threshold for small y values
    threshold = y.quantile(quantile)
    
    # Bin y values into categories
    y_binned = np.where(y < threshold, 0, 1)
    
    # Combine X and y into a single DataFrame for resampling
    df_combined = pd.concat([X, y], axis=1)
    
    # Apply undersampling to the binned categories
    undersample = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    df_res, y_res_binned = undersample.fit_resample(df_combined, y_binned)
    
    return df_res



def oversample_large_y(df, target_col='y', quantile=0.90, sampling_strategy=0.5):
    """
    Oversamples the larger y values based on a specified quantile for regression.

    Parameters:
    df (pd.DataFrame): Input dataframe containing features and target.
    target_col (str): Name of the target column. Default is 'y'.
    quantile (float): Quantile above which y values will be oversampled. Default is 0.75.
    sampling_strategy (float): Ratio of the number of samples after resampling. Default is 0.5.

    Returns:
    pd.DataFrame: Dataframe with oversampled large y values and original small y values.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Define the threshold for large y values
    threshold = y.quantile(quantile)
    
    # Bin y values into categories
    y_binned = np.where(y >= threshold, 1, 0)
    
    # Combine X and y into a single DataFrame for resampling
    df_combined = pd.concat([X, y], axis=1)
    
    # Apply oversampling to the binned categorie
    oversample = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
    df_res, y_res_binned = oversample.fit_resample(df_combined, y_binned)
    
    return df_res


from sklearn.base import BaseEstimator, TransformerMixin

class UndersampleTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_col='y', quantile=0.90, sampling_strategy=0.5):
        self.target_col = target_col
        self.quantile = quantile
        self.sampling_strategy = sampling_strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = pd.concat([X, y], axis=1)
        df_res = undersample_small_y(df, target_col=self.target_col, quantile=self.quantile, sampling_strategy=self.sampling_strategy)
        X_res = df_res.drop(columns=[self.target_col])
        y_res = df_res[self.target_col]
        return X_res, y_res


class OversampleTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_col='y', quantile=0.90, sampling_strategy=0.5):
        self.target_col = target_col
        self.quantile = quantile
        self.sampling_strategy = sampling_strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = pd.concat([X, y], axis=1)
        df_res = oversample_large_y(df, target_col=self.target_col, quantile=self.quantile, sampling_strategy=self.sampling_strategy)
        X_res = df_res.drop(columns=[self.target_col])
        y_res = df_res[self.target_col]
        return X_res, y_res

