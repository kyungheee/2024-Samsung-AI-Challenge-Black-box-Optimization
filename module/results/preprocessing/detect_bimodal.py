import numpy as np

def detect_bimodal(df):
    bimodal_features = []
    for col in df.columns:
        hist, bin_edges = np.histogram(df[col].dropna(), bins=10)
        peaks = np.where(hist > np.mean(hist))[0]
        if len(peaks) > 1:
            bimodal_features.append(col)
            
    return bimodal_features