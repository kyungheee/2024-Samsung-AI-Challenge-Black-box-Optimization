import matplotlib.pyplot as plt
import os


def save_plot(filename, title):
    save_path = os.path.join('../visualization', filename)
    plt.savefig(save_path)
    print(f"{title} saved at {save_path}")

def plot_hyperparameter_optimization(hyperparams, losses, param_name, title='Hyperparameter Optimization'):
    plt.scatter(hyperparams, losses, c='blue', marker='o')
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel('Loss')
    plt.show()
    save_plot('hyperparameter_optimization.png', title)

def plot_convergence_speed(losses, title='Convergence Speed'):
    epochs = range(1, len(losses) + 1)
    plt.plot(epochs, losses, 'g', label='Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    save_plot('convergence_speed.png', title)

def plot_uncertainty_reduction(uncertainties, title='Uncertainty Reduction'):
    epochs = range(1, len(uncertainties) + 1)
    plt.plot(epochs, uncertainties, 'r', label='Uncertainty')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Uncertainty')
    plt.legend()
    plt.show()
    save_plot('uncertainty_reduction.png', title)
    
def plot_comparison(train_before, train_after, label_before, label_after, title='Comparison', ylabel='Loss'):
    epochs = range(1, len(train_before) + 1)
    
    plt.plot(epochs, train_before, 'b', label=label_before)  # 최적화 전 - 파란색
    plt.plot(epochs, train_after, 'r', label=label_after)    # 최적화 후 - 빨간색
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    save_plot(f'{title.lower().replace(" ", "_")}.png', title)