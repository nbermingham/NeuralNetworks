import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Dense layer
class Layer_Dense:
    
        # Layer initialization
        def __init__(self, n_inputs, n_neurons):
            # Initialize weights and biases
            self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
            self.biases = np.zeros((1, n_neurons))
    
        # Forward pass
        def forward(self, inputs):
            # Calculate output values from inputs, weights and biases
            self.output = np.dot(inputs, self.weights) + self.biases

# ReLU activation
class Activation_ReLU:
        
            # Forward pass
            def forward(self, inputs):
                # Calculate output values from inputs
                self.output = np.maximum(0, inputs)

# Softmax activation
class Activation_Softmax:
        
            # Forward pass
            def forward(self, inputs):
        
                # Get unnormalized probabilities
                exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
                # Normalize them for each sample
                probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
                self.output = probabilities

# Common loss class
class Loss:
        
            # Calculates the data and regularization losses
            # given model output and ground truth values
            def calculate(self, output, y):
            
                # Calculate sample losses
                sample_losses = self.forward(output, y)
            
                # Calculate mean loss
                data_loss = np.mean(sample_losses)
            
                # Return loss
                return data_loss

class Loss_CategoricalCrossEntropy(Loss):
       def forward(self, y_pred, y_true):
            samples = len(y_pred)
            y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

            if len(y_true.shape) == 1:
                correct_confidences = y_pred_clipped[range(samples), y_true]
            elif len(y_true.shape) == 2:
                correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

            negative_log_likelihoods = -np.log(correct_confidences)
            return negative_log_likelihoods
            

X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)

print('loss:', loss)