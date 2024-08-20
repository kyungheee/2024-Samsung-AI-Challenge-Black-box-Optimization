# ?•„?š”?•œ ?¼?´ë¸ŒëŸ¬ë¦? ê°?? ¸?˜¤ê¸?
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.impute import SimpleImputer

# ë§Œë“  ?•¨?ˆ˜?“¤ ê°?? ¸?˜¤ê¸?
from .detect_skewness import detect_skewness
from .detect_bimodal import detect_bimodal
from .bimodal_transformer import BimodalTransformer


# ?ŒŒ?´?”„?¼?¸ ë§Œë“¤ê¸?
def create_pipeline(df):
    
    scaler = MinMaxScaler()
    
    # ?”¼ì²? ë¶„ë¥˜
    df = df.drop(columns=['y'])
    skewed_features = detect_skewness(df)
    bimodal_features = detect_bimodal(df)
    
    # skewed_features?— ????•œ pipeline
    skewed_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Handle missing values
        ('power', PowerTransformer()) # Normalize skewed features
    ])
    
    # bimodal_features?— ????•œ pipeline
    bimodal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('bimodal', BimodalTransformer()), # Custom transformation for bimodal distributions
        ('scaler', scaler) # Scale features using selected scaler
    ])
    
    # ?‚˜ë¨¸ì?? features?— ????•œ pipeline
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
    
    # single pipeline?œ¼ë¡? combine
    return Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    