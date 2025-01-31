import torch

def importance_sampling(vae_model, x, num_samples=100):
    """
    Perform importance sampling to estimate uncertainty in the VAE decoder.
    """
    vae_model.eval()
    with torch.no_grad():
        z_mean, z_logvar = vae_model.encoder(x)
        z_samples = torch.randn(num_samples, *z_mean.shape) * torch.exp(0.5 * z_logvar) + z_mean
        decoded_samples = vae_model.decoder(z_samples)
        log_probs = vae_model.log_prob(decoded_samples, x)
        log_weights = torch.logsumexp(log_probs, dim=0) - torch.log(torch.tensor(num_samples))
        weights = torch.exp(log_weights)
        normalized_weights = weights / torch.sum(weights)
        return normalized_weights
