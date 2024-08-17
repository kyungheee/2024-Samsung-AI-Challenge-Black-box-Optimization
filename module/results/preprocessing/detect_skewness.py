def detect_skewness(df, skew_threshold=0.5):
    skew_features = df.apply(lambda x: x.skew())
    return skew_features[skew_features.abs() > skew_threshold].index.tolist()
