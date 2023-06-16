import numpy as np

class Layer:
    def __init__(self, input_size=1, output_size=1, activation_function=None, name= None):
        pass
    def forward_propagation(self, inputs):
        raise NotImplementedError
    def backward_propagation(self, delta, learning_rate):
        raise NotImplementedError
    def print_layer(self):
        raise NotImplementedError
    def update_self(self):
        raise NotImplementedError

