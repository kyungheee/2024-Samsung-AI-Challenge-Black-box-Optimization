import pandas as pd
from pycaret.regression import setup, compare_models, save_model, load_model, get_config, finalize_model
# from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def run_automl(X, y):
    reg_setup = setup(data=pd.concat([X, y], axis=1), target=y, verbose=False)
    top5_models = compare_models(sort='MAE', n_select=5)
    
    for idx, model in enumerate(top5_models):
        save_model(model, f'top{idx+1}_{model}')
    
    return top5_models