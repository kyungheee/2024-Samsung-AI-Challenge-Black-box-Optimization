# 필요한 라이브러리 가져오기
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.impute import SimpleImputer

# 만든 함수들 가져오기
from .detect_skewness import detect_skewness
from .detect_bimodal import detect_bimodal
from .bimodal_transformer import BimodalTransformer

# 파이프라인 만들기
def create_pipeline(df, minmax=MinMaxScaler):
    
    scaler = minmax()
    
    # 피처 분류
    skewed_features = detect_skewness(df)
    bimodal_features = detect_bimodal(df)
    
    # skewed_features에 대한 pipeline
    skewed_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Handle missing values
        ('power', PowerTransformer()) # Normalize skewed features
    ])
    
    # bimodal_features에 대한 pipeline
    bimodal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('bimodal', BimodalTransformer()), # Custom transformation for bimodal distributions
        ('scaler', scaler) # Scale features using selected scaler
    ])
    
    # 나머지 features에 대한 pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', scaler)
    ])
    
    # apply transformation
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, [col for col in df.columns if col not in skewed_features + bimodal_features])
            ('skewed', skewed_transformer, skewed_features)
            ('bimodal', bimodal_transformer, bimodal_features)
        ]
    )
    
    # single pipeline으로 combine
    return Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    