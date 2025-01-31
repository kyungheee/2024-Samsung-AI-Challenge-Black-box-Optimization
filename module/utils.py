import json
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer

def save_scaler(scaler, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(scaler, f)

def load_scaler(file_path):
    with open(file_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

# Function to handle missing values
def handle_missing_values(features):
    imputer = SimpleImputer(strategy='mean')  # Use 'mean', 'median', etc. to fill missing values
    return imputer.fit_transform(features)

def load_best_hyperparameters(file_path="results/best_hyperparameters.json"):
    with open(file_path, 'r') as f:
        hyperparams = json.load(f)
    return hyperparams

def save_latent_variables(latent_train, file_path):
    pd.DataFrame(latent_train).to_csv(file_path, index=False)

def save_submission(predictions):
    submission = pd.DataFrame({'y': predictions})
    submission.to_csv(f'results/submission.csv', index=False)