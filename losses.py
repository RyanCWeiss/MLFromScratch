import numpy as np
from loss import Loss

class MSE(Loss):
    def __init__(self):
        self.name = "MSE"
#         super().__init__()
        pass
    def cost(self, y_pred, y_train):
        return np.mean((y_pred - y_train)**2)
    
    def derivative(self,y_pred, y_train):
        return 2.0 * np.mean((y_pred - y_train), axis = 0)
    
class RMSE(Loss):
    def __init__(self):
        self.name = "RMSE"
#         super().__init__()
        pass
    def cost(self, y_pred, y_train):
        return np.sqrt(np.mean((y_pred - y_train)**2))
    
    def derivative(self,y_pred, y_train):
        n = y_pred.shape[0]
        return np.sqrt(n*np.sum((y_pred - y_train), axis=0)**2)

class BCE(Loss):
    def __init__(self):
        self.name = "Binary Cross Entropy"
#         super().__init__()
        pass
    def cost(self, y_pred, y_train):
        return np.mean(-y_train * np.log(y_pred) - (1 - y_train) * np.log(1 - y_pred))
    
    def derivative(self,y_pred, y_train):
        return ((1 - y_train) / (1 - y_pred) - y_train / y_pred) / np.size(y_train)
