import pickle
import torch
import pandas as pd
import json
from .vae import VAE, load_vae_model
from .utils import save_submission, handle_missing_values, load_best_hyperparameters
from sklearn.linear_model import LinearRegression


# perform inference
def run_inference():
    
    # load test data
    test_data = pd.read_csv('data/test.csv').values
    
    # apply scaler to test data
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    test_data_scaled = scaler.transform(test_data)
    
    # to tensor
    test_tensor = torch.tensor(test_data_scaled, dtype=torch.float32)
    
    # load best hyperparameters from JSON file
    best_hyperparams = load_best_hyperparameters()
    
    # load the trained VAE model using best hyperparameters
    vae_model = load_vae_model(input_dim=test_tensor.shape[1], hidden_dim=best_hyperparams['hidden_dim'] ,latent_dim=best_hyperparams['latent_dim'])
    
    # encode the test data using VAE 
    with torch.no_grad():
        latent_vars, _ = vae_model.encode(test_tensor)
            
        # Check for empty latent variables
        if latent_vars.shape[1] == 0:
            raise ValueError("Latent variables have no features. Check the VAE model architecture or input data.")

        latent_vars = handle_missing_values(latent_vars.detach().numpy())
        
        
    
    # # load the trained regression model
    # with open('models/regression_model.pkl', 'rb') as f:
    #     regression_model = pickle.load(f)
    
    regression_model = LinearRegression()
    
    # predict based on latent variables
    pred = regression_model.predict(latent_vars)
    
    # load the sample submission file
    submission = pd.read_csv('data/sample_submission.csv')
    submission['y'] = pred
        
    # save the results
    save_submission(submission)

# start inference
if __name__ == '__main__':
    run_inference()
