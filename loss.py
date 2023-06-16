import numpy as np

class Loss():
    def __init__(self):
        self.name = "Virtual Super"
        pass
    def cost(self,y_pred, y_train):
        raise NotImplementedError
    def derivative(self,y_pred, y_train):
        raise NotImplementedError
        
