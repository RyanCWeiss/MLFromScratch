import numpy as np

class ActivationFunction:
    def __init__(self):
        self.name = "Virtual Super"
    
    def activation(self,z):
        raise NotImplementedError
    
    def derivative(self, z):
        raise NotImplementedError
