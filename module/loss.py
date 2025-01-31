import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error

def total_loss(reconstructed, original, mu, logvar, latent_vars, target, regression_model):
    '''
    VAE loss : reconstruction loss + KL divergence
    regression loss : MSE for the regression model
    latent distance : regularization term based on the distance in latent space
    '''
    # VAE loss = reconstruction loss + KL divergence loss
    recon_loss = F.mse_loss(reconstructed, original)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    vae_loss = recon_loss + kld
    
    # Regression loss on latent variables
    latent_pred = regression_model.predict(latent_vars)  # No detach() needed
    regression_loss = F.mse_loss(torch.tensor(latent_pred, dtype=torch.float32), target)

    # Latent space regularization (encourages latent variables to be well-formed)
    latent_distance_loss = torch.mean(torch.norm(mu, dim=1))  # L2 norm regularization
    
    # Combine all losses
    total_loss = vae_loss + regression_loss + 0.1 * latent_distance_loss
    
    return total_loss
