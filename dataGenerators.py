import numpy as np
from numpy import random as random
from itertools import cycle, islice

import matplotlib.pyplot as plt

class Featurespace_Generator():
    def __init__(self):
        pass
    
    def generate_categorical_response(self,_x):
        return _x
#         raise NotImplementedError
    
    def generate_continuous_response(self,_x, degree_flexibility = 1, coeff_flexibility = 1):
        degree = np.random.poisson(0+degree_flexibility)+1
        coeffs = np.random.normal(0,10*coeff_flexibility, size = (degree,1))/degree/4
        x = np.polynomial.polynomial.polyval(_x, coeffs)
#         print(coeffs)
        return x
#         raise NotImplementedError

    def generate_discrete_response(self, _x):
        
        return _x
#         raise NotImplementedError
        
    def generate_data(self,nObs = 0, seedDistr = "normal", nCategorical=0, categoricalLevels = [],categoricalProbabilities = [], nContinuous=0, nDiscrete=0, 
                     mean = 0, scale= 1, low = -1, high = 1, noise = 1):
            
        if len(categoricalLevels)< nCategorical:
            if len(categoricalLevels) == 0:
                msg = "Failed to provide levels for categoricals"
                raise ValueException(msg)
            categoricalLevels = list(islice(cycle(categoricalLevels), nCategorical))
            msg = "Insufficient sets of levels provided, recycling levels, {}".format(categoricalLevels)
            warn(msg)
        if not categoricalProbabilities:
            categoricalProbabilities = [[1/len(categoricalLevels[i])]*len(categoricalLevels[i]) for i in range(nCategorical)]
            
        size = (nObs,1)
        generator = np.random.default_rng()
        
        ###
        ### Avoid duplication by having the initial vector duplicated by nFeatures then 
        ### apply each function to a individual column??
        ###
        dist_map = {
            'normal': [generator.normal, (mean,scale,size)],
            'lognormal':[generator.lognormal,(mean,scale,size)],
            'uniform': [generator.uniform,(low,high,size)],
            'exponential': [generator.exponential,(1,size)],
            'logistic': [generator.logistic,(mean,scale,size)]
            
            # Add more distributions and functions as needed
        }

        if seedDistr not in dist_map:
            raise ValueError(f"Invalid distribution: {seedDistr}")
        gArgs = dist_map[seedDistr]
        _x = gArgs[0](*gArgs[1])
        
        featureCols = [None for i in range(nContinuous+nCategorical+nDiscrete)]
        for i in range(0,nCategorical):
            categories = categoricalLevels[i]
            featureCols[i] = self.generate_categorical_response(_x, categories)
        for i in range(nCategorical, nDiscrete):
            featureCols[i] = self.generate_discrete_response(_x)
        for i in range(nDiscrete, nContinuous):
            featureCols[i] = self.generate_continuous_response(_x)
#         res = np.stack(*featureCols).T
        return featureCols
            
    def display_partial_relationships(self):
        raise NotImplementedError

# perhaps I should create a true predictor space such that each predictor has partial shared relationship


# class Regression_Generator():
#     def __init__(self, nContinuousPredictors=0, continuousDistributions=["Unif"]*nContinuousPredictors, 
#                  nDiscretePredictors=0, discreteDistributions=["Unif"]*nDiscretePredictors, 
#                  nCategoricalPredictors=0, categoricalPredictorsLevels = [2]*nCategoricalPredictors, categoricalProbabilities = [[.5]*2]):
#         pass
#     def get_responses(predictor_shape,response_shape):
#         raise NotImplementedError
#     def display_partial_relationships():
#         raise NotImplementedError

# class Classification_Generator():
#     def __init__(self, nContinuousPredictors=0, continuousDistributions=["Unif"]*nContinuousPredictors, 
#                  nDiscretePredictors=0, discreteDistributions=["Unif"]*nDiscretePredictors, 
#                  nCategoricalPredictors=0, categoricalPredictorsLevels = [2]*nCategoricalPredictors, categoricalProbabilities = [[.5]*2]):
#         pass
#     def get_responses(predictor_shape,response_shape):
#         raise NotImplementedError
#     def display_partial_relationships():
#         raise NotImplementedError
