#linear regression model using gradient descent.
import numpy as np

class LinearRegressionScratch:
    
    def __init__(self, learning_rate=0.01, iterations=1000, verbose=True):

        self.learning_rate = learning_rate
        self.iterations = iterations
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.cost_history = []

    def _cost_function(self, X, y, predictions):
    #Calculates the mse cost
        m = len(y)
        cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
        return cost

    def fit(self, X, y):
        
        num_samples, num_features = X.shape
        
        self.weights = np.zeros(num_features)
        self.bias = 0
        self.cost_history = []
        y = y.reshape(-1) 
        if self.verbose:
            print("Starting Gradient Descent...")
            print(f"Num samples: {num_samples}, Num features: {num_features}")
            print(f"Learning Rate: {self.learning_rate}, Iterations: {self.iterations}")

        # Gradient Descent
        for i in range(self.iterations):
            # h(x) = X * w + b
            predictions = self._predict_raw(X)
            dw = (1 / num_samples) * np.dot(X.T, (predictions - y))
            db = (1 / num_samples) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            cost = self._cost_function(X, y, predictions)
            self.cost_history.append(cost)
            if self.verbose and (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{self.iterations}, Cost: {cost:.4f}")

        if self.verbose:
            print("Gradient Descent Finished.")
            print(f"Final Cost: {self.cost_history[-1]:.4f}")


    def _predict_raw(self, X):
       
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        
        if self.weights is None or self.bias is None:
            raise Exception("Model has not been trained yet. Call fit() first.")
        return self._predict_raw(X)

    def get_params(self):
        if self.weights is None or self.bias is None:
             raise Exception("Model has not been trained yet.")
        return {'weights': self.weights, 'bias': self.bias}

    def set_params(self, params):
        self.weights = params.get('weights', None)
        self.bias = params.get('bias', None)
        if self.weights is None or self.bias is None:
            raise ValueError("Invalid parameters provided. Need 'weights' and 'bias'.")