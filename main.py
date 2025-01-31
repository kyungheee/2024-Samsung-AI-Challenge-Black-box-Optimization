import json
from module.dataset import train_val_split, get_data_loader
from module.train import random_training, optimized_training
from module.inference import run_inference
from module.visualization import plot_comparison, plot_hyperparameter_optimization, plot_convergence_speed, plot_uncertainty_reduction
import os

def main():
    # Step 1: Load and split the dataset
    print("Loading and splitting dataset...")
    train_dataset, val_dataset, scaler = train_val_split('data/train.csv', test_size=0.2)
    train_loader, val_loader = get_data_loader(train_dataset, val_dataset, batch_size=32)

    # Step 2: Train the model with random hyperparameters
    print("Training with random hyperparameters (Before Optimization)...")
    random_result = random_training(train_loader, val_loader,random_state=42, epochs=10)

    # Step 3: Train the model with optimized hyperparameters
    print("Training with optimized hyperparameters (After Optimization)...")
    optimized_result = optimized_training(train_loader, val_loader, epochs=10)

    # Step 4: Compare random and optimized hyperparameters
    print("Comparing random and optimized hyperparameters...")
    print(f"Random Train Loss: {random_result['train_loss']}, Optimized Train Loss: {optimized_result['train_loss']}")
    print(f"Random Validation Loss: {random_result['val_loss']}, Optimized Validation Loss: {optimized_result['val_loss']}")

    # Step 5: Plot and visualize the comparison
    print("Plotting performance comparison...")
    plot_comparison(
        [random_result['train_loss']], [optimized_result['train_loss']], 
        label_before='Train Loss Before Optimization', 
        label_after='Train Loss After Optimization', 
        title='Train Loss Comparison'
    )
    
    plot_comparison(
        [random_result['val_loss']], [optimized_result['val_loss']], 
        label_before='Validation Loss Before Optimization', 
        label_after='Validation Loss After Optimization', 
        title='Validation Loss Comparison'
    )
    plot_hyperparameter_optimization(hyperparams=[10, 15], losses=[train_losses_before[-1], train_losses_after[-1]], param_name='Latent Dimension')
    plot_convergence_speed(train_losses_after, title='Convergence Speed After Optimization')
    plot_uncertainty_reduction([0.1, 0.05, 0.02], title='Uncertainty Reduction During Optimization')
    
    # Step 6: Perform inference using the optimized model
    print("Running inference using optimized hyperparameters...")
    predictions = run_inference()
    print(f"complete inference")

if __name__ == "__main__":
    main()
