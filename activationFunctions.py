import numpy as np
from activationFunction import ActivationFunction

class ReLu(ActivationFunction):
    def __init__(self):
        self.name = "Relu"
    
    def activation(self,z):
        return np.maximum(0, z)
    
    def derivative(self, z):
        return np.where(z > 0, 1, 0)

class ClampedReLu(ActivationFunction):
    def __init__(self, min_out = 0, max_out = 1):
        self.name = "Clamped Relu"
        self.min_out = min_out
        self.max_out = max_out
        self.m = max_out-min_out
        
    def activation(self,z):
        tmp_ = np.maximum(self.min_out, z) # clamp lower output
        return np.minimum(tmp_, self.max_out) # clamp upper output
    
    def derivative(self, z):
        tmp_ = np.where(z > self.min_out, z, 0) # set derivative to 0 when z< min
        return np.where(tmp_ < self.max_out, self.m, 0) # set derivative to 0 when z> max, return m in bounds

class LeakyReLu(ActivationFunction):
    def __init__(self, m_1 = 0.01, m_2 = 1):
        self.name = "Leaky Relu"
        self.m_1 = m_1
        self.m_2 = m_2
    def activation(self,z):
        return np.maximum(self.m_1*z, self.m_2*z)
    
    def derivative(self, z):
        return np.where(z > 0, self.m_2, self.m_1)
    
class Sigmoid(ActivationFunction):
    def __init__(self, scale = 1):
        self.name = "Sigmoid"
        self.scale = scale
    def activation(self,z):
        return self.scale / (1 + np.exp(-z))
    
    def derivative(self, z):
        sigmoid = self.activation(z)
        return self.scale* (sigmoid * (1 - sigmoid))

class Linear(ActivationFunction):
    def __init__(self, m = 1, b = 0):
        self.name = "Linear"
        self.m = m
        self.b = b
    
    def activation(self,z):
        return self.m * z + self.b
    
    def derivative(self, z):
        return np.ones(shape = z.shape) * self.m

class Tanh(ActivationFunction):
    def __init__(self,scale = 1):
        self.name = "Tanh"
        self.scale = scale
    def activation(self,z):
        return self.scale * np.tanh(z)

    def derivative(self,z):
        return self.scale *(1 - np.tanh(z) ** 2)

class Sin(ActivationFunction):
    def __init__(self, scale = 1):
        self.name = "Sin"
        self.scale = scale
    def activation(self,z):
        return self.scale*np.sin(z)

    def derivative(self,z):
        return self.scale*np.cos(z)

class Cos(ActivationFunction):
    def __init__(self, scale = 1):
        self.name = "Cos"
        self.scale = scale
    def activation(self,z):
        return self.scale*np.cos(z)

    def derivative(self,z):
        return -self.scale*np.sin(z)

class Gaussian(ActivationFunction):
    def __init__(self, mu= 0, sigma = 1,scale = 1):
        self.name = "Gaussian"
        self.mu = mu
        self.sigma = sigma
        self.scale = scale
    def activation(self,z):
        return self.scale*(1/(self.sigma*np.sqrt(np.pi)))* np.exp(-.5*np.power((z-self.mu)/self.sigma,2))

    def derivative(self,z):
        return self.scale*(1/(self.sigma*np.sqrt(np.pi))) * np.exp(-.5*np.power((z-self.mu)/self.sigma,2)) * 2*((z-self.mu)/self.sigma)
