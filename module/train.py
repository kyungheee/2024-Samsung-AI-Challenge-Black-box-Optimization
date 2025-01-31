import json
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from .vae import VAE
from .dataset import train_val_split, get_data_loader
from .utils import save_latent_variables, load_best_hyperparameters, handle_missing_values
from .loss import total_loss
from uncertainty_guided_optimization.bayesian_optimization import bayesian_optimization
import os
import random
from sklearn.linear_model import LinearRegression

# common train function for VAE + regression
def train_vae(vae_model, train_loader, val_loader, optimizer, epochs, writer=None):
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        vae_model.train()
        total_train_loss = 0

        for batch_features, batch_target in train_loader:
            optimizer.zero_grad()

            # forward pass through VAE
            reconstructed, mu, logvar = vae_model(batch_features)
            latent_vars, _ = vae_model.encode(batch_features)
            
            # Check for empty latent variables
            if latent_vars.shape[1] == 0:
                raise ValueError("Latent variables have no features. Check the VAE model architecture or input data.")

            latent_vars = handle_missing_values(latent_vars.detach().numpy())
            
            # define regression model and fit it
            regression_model = LinearRegression()
            regression_model.fit(latent_vars, batch_target.detach().numpy())

            # compute VAE + regression loss
            loss = total_loss(reconstructed, batch_features, mu, logvar, latent_vars, batch_target, regression_model)
            total_train_loss += loss.item()

            # backpropagation
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # tensorboard
        if writer:
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)

        # validation phase
        vae_model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_features, batch_target in val_loader:
                reconstructed, mu, logvar = vae_model(batch_features)
                latent_vars, _ = vae_model.encode(batch_features)

                # Check for empty latent variables during validation
                if latent_vars.shape[1] == 0:
                    raise ValueError("Latent variables have no features.")

                latent_vars = handle_missing_values(latent_vars.detach().numpy())
                
                regression_model.fit(latent_vars, batch_target.detach().numpy())
                val_loss = total_loss(reconstructed, batch_features, mu, logvar, latent_vars, batch_target, regression_model)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        if writer:
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return np.mean(train_losses), np.mean(val_losses)

# train model with random hyperparameters
def random_training(train_loader, val_loader, random_state=42, epochs=10):
    random.seed(random_state)
    hyperparams = {
        'hidden_dim': random.choice([16, 32, 64, 128]),
        'latent_dim': random.choice([2, 4, 6, 8, 10]),
        'learning_rate': random.choice([1e-4, 1e-3, 1e-2]),
        'batch_size': random.choice([16, 32, 64])
    }
    
    # initialize VAE model
    vae_model = VAE(input_dim=train_loader.dataset.features.shape[1], hidden_dim=hyperparams['hidden_dim'], latent_dim=hyperparams['latent_dim'])
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=hyperparams['learning_rate'])
    
    # train the model
    train_loss, val_loss = train_vae(vae_model, train_loader, val_loader, optimizer, epochs)
    
    return {**hyperparams, 'train_loss': train_loss, 'val_loss': val_loss}

# optimize hyperparameters using Bayesian Optimization
def optimized_training(train_loader, val_loader, epochs=10):
    bounds = {
        'hidden_dim': [16, 32, 64, 128],
        'latent_dim': [2, 4, 6, 8, 10],
        'learning_rate': [1e-4, 1e-3, 1e-2],
        'batch_size': [16, 32, 64]
    }

    # Bayesian Optimization logic here
    best_params = bayesian_optimization(train_loader, val_loader, bounds, train_fn, num_iterations=25)

    vae_model = VAE(input_dim=train_loader.dataset.features.shape[1], hidden_dim=best_params['hidden_dim'], latent_dim=best_params['latent_dim'])
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=best_params['learning_rate'])

    # train the model with optimized hyperparameters
    train_loss, val_loss = train_vae(vae_model, train_loader, val_loader, optimizer, epochs)

    # save best hyperparameters as a JSON file
    best_hyperparams = {**best_params, 'train_loss': train_loss, 'val_loss': val_loss}

    os.makedirs('results', exist_ok=True)
    with open('results/best_hyperparameters.json', 'w') as f:
        json.dump(best_hyperparams, f)

    return best_hyperparams

def train_fn(hyperparams, train_loader, val_loader, epochs=50):
    hidden_dim, latent_dim, learning_rate, batch_size = hyperparams

    # initialize and train the model with these hyperparameters
    vae_model = VAE(input_dim=train_loader.dataset.features.shape[1], latent_dim=latent_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=learning_rate)

    # train and evaluate the model
    train_loss, val_loss = train_vae(vae_model, train_loader, val_loader, optimizer, epochs)

    return val_loss
