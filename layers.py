import numpy as np
from activationFunctions import Linear as Linear
from layer import Layer


class Dense(Layer):
    def __init__(self, input_size=1, output_size=1, activation_function=Linear(), name= None):
        self.type = "Dense"
        self.name = name
#         self.weights = np.random.uniform(low = -.5 ,high = .5, size=(output_size, input_size))
        self.weights = np.random.randn(output_size, input_size) # Random initialization fo weights
        self.biases = np.zeros((output_size, 1))
        self.activation_function = activation_function
        self.tunable_params = output_size*(input_size + 1)
        
    def forward_propagation(self, inputs):
        self.inputs = inputs 
        self.z = (self.inputs @ self.weights.T)+self.biases.T
        return self.activation_function.activation(self.z)

    def backward_propagation(self, gradient, learning_rate):
        batch_size = self.inputs.shape[0]
        gradient_A = self.activation_function.derivative(self.z)
        gradient_Z = np.multiply(gradient, gradient_A)
        gradient_prev = gradient_Z @ self.weights 
        
        # update weights, biases & backpropogate gradient
        gradient_B = learning_rate * np.sum(gradient_Z, axis = 0, keepdims=True).T/batch_size
        gradient_W = learning_rate * (gradient_Z.T @ self.inputs)/batch_size 
        self.weights -= gradient_W
        self.biases -= gradient_B
        return gradient_prev 
    
    def print_layer(self):
        print("[layer({}): {}]\n\tactivation: {}, input_shape = {}, output_shape = {}, w: {}, b:{} tunable params: {}".format(
            self.type,self.name, self.activation_function.name, self.weights.shape[1], 
            self.weights.shape[0], self.weights.shape, self.biases.shape,
            self.tunable_params))
    
    def update_self(self):
        self.tunable_params = self.weights.shape[0]*self.weights.shape[1] + self.biases.shape[0]*self.biases.shape[1]

class Conv(Layer):
    def __init__(self, input_shape= (1,1,1), output_shape = (1,1,1), num_filters= 1, filter_size = (1,1,1) activation_function=Linear(), name= None):
        self.name = name
        
        #TODO: modify these accordingly
        if num_filters <1:
            raise ValueError("Error: The number of filters must be at least 1.")
        if num_filters == 1:
            if not len(output_shape) == len(input_shape)+1:
                raise ValueError("Error: The input shape: {} and output shape: {} must have the same dimension".format(
        if num_filters>1:
            if not len(output_shape) == len(input_shape)+1:
                raise ValueError("Error: The input shape: {} and output shape: {} are incompatible given the number of filters {}".format(input_shape, output_shape, num_filters))
        #self. filters = np.array(shape = (num_filters, *filter_size))
        
        
        
        
        