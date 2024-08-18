import pandas as pd
from pycaret.regression import setup, compare_models, save_model, load_model, get_config, finalize_model
# from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def run_automl(X, y):
    reg_setup = setup(data=pd.concat([X, y], axis=1), target=y, verbose=False)
    top5_models = compare_models(sort='MAE', n_select=5)
    
    for idx, model in enumerate(top5_models):
        save_model(model, f'top_model_{idx+1}')
    
    return top5_models

def get_top_features(model, X):
    model = finalize_model(model)
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = model.coef_
    else:
        raise AttributeError(f"The {model} does not have 'feature_importances_' or 'coef_' attribute.")

    feature_importance = pd.DataFrame({
        'feature' : X.columns,
        'importance' : importance
    }).sort_values(by='importance', ascending=False)
    
    top_5_features = feature_importance['feature'].head(5)
    
    return top_5_features

def train_with_top(X, y, top_features, model):
    X_top = X[top_features]
    model = finalize_model(model)
    model.fit(X_top, y)
    y_pred = model.predict(X_top)
    mae = mean_absolute_error(y, y_pred)
    print(f'model trained with top5 features >>> model : {model}, mae : {mae}')
    
    return model
    
