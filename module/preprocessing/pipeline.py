# ??? ?Ό?΄λΈλ¬λ¦? κ°?? Έ?€κΈ?
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.impute import SimpleImputer

# λ§λ  ?¨??€ κ°?? Έ?€κΈ?
from .detect_skewness import detect_skewness
from .detect_bimodal import detect_bimodal
from .bimodal_transformer import BimodalTransformer


# ??΄??Ό?Έ λ§λ€κΈ?
def create_pipeline(df):
    
    scaler = MinMaxScaler()
    
    # ?Όμ²? λΆλ₯
    df = df.drop(columns=['y'])
    skewed_features = detect_skewness(df)
    bimodal_features = detect_bimodal(df)
    
    # skewed_features? ???? pipeline
    skewed_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Handle missing values
        ('power', PowerTransformer()) # Normalize skewed features
    ])
    
    # bimodal_features? ???? pipeline
    bimodal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('bimodal', BimodalTransformer()), # Custom transformation for bimodal distributions
        ('scaler', scaler) # Scale features using selected scaler
    ])
    
    # ?λ¨Έμ?? features? ???? pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', scaler)
    ])
    
    # apply transformation
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, [col for col in df.columns if col not in (skewed_features + bimodal_features)]),
            ('skewed', skewed_transformer, skewed_features),
            ('bimodal', bimodal_transformer, bimodal_features)
        ]
    )
    
    # single pipeline?Όλ‘? combine
    return Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    