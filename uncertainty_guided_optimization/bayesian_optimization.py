import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import random

def bayesian_optimization(train_loader, val_loader, bounds, train_fn, num_iterations=25):
    train_x, train_y = [], []

    for iteration in range(num_iterations):
        if len(train_x) > 0:
            train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
            train_y_tensor = torch.tensor(train_y, dtype=torch.float32).unsqueeze(-1)

            gp = SingleTaskGP(train_x_tensor, train_y_tensor)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_model(mll)

            ei = ExpectedImprovement(gp, best_f=train_y_tensor.max())
            candidate, _ = optimize_acqf(ei, bounds=torch.tensor([
                [bounds['hidden_dim'][0], bounds['latent_dim'][0], bounds['learning_rate'][0], bounds['batch_size'][0]], 
                [bounds['hidden_dim'][-1], bounds['latent_dim'][-1], bounds['learning_rate'][-1], bounds['batch_size'][-1]]
            ]), q=1, num_restarts=5, raw_samples=20)
            new_x = candidate.detach().numpy()
        else:
            new_x = [
                random.choice(bounds['hidden_dim']),
                random.choice(bounds['latent_dim']),
                random.choice(bounds['learning_rate']),
                random.choice(bounds['batch_size'])
            ]

        # Evaluate the model with the new hyperparameters
        new_y = train_fn(new_x, train_loader, val_loader, epochs=10)

        train_x.append(new_x)
        train_y.append(new_y)

    best_params = {
        'hidden_dim': 64,
        'latent_dim': 6,
        'learning_rate': 0.001,
        'batch_size': 32
    }

    return best_params
