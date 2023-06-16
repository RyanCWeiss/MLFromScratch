import numpy as np

class NeuralNetwork():
    def __init__(self,name = "Unimplemented Super", loss_fn = None):
        # Initialize the network parameters
        self.layers = None
        self.loss = loss_fn
        self.name = name
        self.tunable_params = None
    def forward_propagation(self, x):
        raise NotImplementedError
    def backward_propagation(self, y_pred, y_train, learning_rate):
        raise NotImplementedError
    def print_network(self):
        raise NotImplementedError
    def train(self, y_pred, y_train, learning_rate):
        raise NotImplementedError
    def predict(self, x_pred):
         raise NotImplementedError
