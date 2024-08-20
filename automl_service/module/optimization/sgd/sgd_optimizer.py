import numpy as np

class SGDOptimizer:
    def __init__(self, learning_rate = 0.01, max_iters=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.tolerance = tolerance
    
    def optimize(self, initial_params, gradient_func, data):
        params = np.array(initial_params)
        for i in range(self.max_iters):
            np.random.shuffle(data)
            for x in data:
                gradients = np.array(gradient_func(params, x))
                params = params - self.learning_rate*gradients
                
            if np.linalg.norm(gradients) < self.tolerance:
                print(f'convergence reached at iteration {i}')
                break
        
        return params
        