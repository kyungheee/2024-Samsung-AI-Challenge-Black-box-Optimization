
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

from .sampling import OversampleTransformer, UndersampleTransformer
# Define the custom transformers for sampling

def create_pipeline(df, target_col='y', oversample_quantile=0.75, undersample_quantile=0.25, sampling_strategy=0.5):
    scaler = MinMaxScaler()

    # Define transformers for all features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
        ('scaler', scaler)  # Scale features
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, df.columns)  # Apply transformations to all columns
        ]
    )

    # Define sampling transformers
    oversample_transformer = OversampleTransformer(target_col=target_col, quantile=oversample_quantile, sampling_strategy=sampling_strategy)
    undersample_transformer = UndersampleTransformer(target_col=target_col, quantile=undersample_quantile, sampling_strategy=sampling_strategy)

    # Define the overall pipeline
    pipeline = Pipeline(steps=[
        ('oversample', oversample_transformer),
        ('undersample', undersample_transformer),
        ('preprocessor', preprocessor)
    ])

    return pipeline