import numpy as np
from neuralNetwork import NeuralNetwork
from losses import MSE
import matplotlib.pyplot as plt

class SequentialModel(NeuralNetwork):
    """
    Sequential Models operate using feed forward (forward propogation) and feed backward (backward propogation). Forward propogation arrives at values given the linear functions of W*X+B passed through activation functions belonging to each layer. These values are that of the predictions, and can be used to determine the error in the output, then update the weights and biases resulting in gradient descent(moving your parameters in the direction of a negative gradient on the loss function). 
    
    Layers: 
        - Dense(Fully Connected)
        * Soon to be implemented:
            - Reshape
            - Conv3D Convolutional(3D)
            - Dropout
            - Normalization
    
    Loss Functions:
        - MSE
        - RMSE
        * Soon to be implemented:
            - Binary Cross Entropy
            - Categorical Cross Entropy
        
    Activation Functions:
    (Because Activation Functions can be used as last layer output they have been given flexible definitions, see activationFunctions.py)
        - Linear
        - ReLu
        - LeakyReLu
        - ClampedReLu 
        - Sigmoid
        - Tanh
        - Sin
        - Cos
        - Gaussian
        * Soon to be implemented:
            - SoftMax
    """
    def __init__(self,name = "New Net", loss_fn = MSE()):
        # Initialize the network parameters
        self.layers = []
        self.loss = loss_fn
        self.name = name
        self.tunable_params = sum([self.layers.tunable_params for layer in self.layers])
        
    def add_layer(self, layer):
        if self.layers:
            input_size = self.layers[-1].weights.shape[0]
            layer.weights = np.random.randn(layer.weights.shape[0], input_size)
        if layer.name == None:
            layer.name = len(self.layers)+1
        self.layers.append(layer)
        layer.update_self()
        self.tunable_params+= layer.tunable_params
    
    def forward_propagation(self, x):
        # Perform forward propagation
        for layer in self.layers:
            x = layer.forward_propagation(x)
        return x
    
    def backward_propagation(self, y_pred, y_train, learning_rate):
        # Perform backward propagation
        gradient = self.loss.derivative(y_pred, y_train)
        
        for layer in reversed(self.layers):
            gradient = layer.backward_propagation(gradient, learning_rate)   
    
    def print_network(self):
        if not self.layers:
            "NN has no layers"
        print("Name: {}, tunable_params: {}".format(self.name,self.tunable_params))
        print("input size: {}".format(self.layers[0].weights.shape[1]))
        for layer in self.layers:
            layer.print_layer()
        print("output size: {}".format(self.layers[-1].weights.shape[0]))
        
    def train(self, x_train, y_train, num_epochs=10, batch_size=32, learning_rate = 0.01, verbose = 0):
        if not len(x_train.shape) == 2:
            raise ValueError("Input shape should be 2 dimensional: dim 1: observations, dim 2: number of features")
        num_samples = x_train.shape[0]
        num_batches = num_samples // batch_size # is it worth it to squeeze in the partial batch?
        steps = []
        loss_values = []
        epochs = []
        loss_epochs= []
        for epoch in range(num_epochs):
            # Shuffle the training data
            indices = np.random.permutation(num_samples)
            x_train = x_train[indices]
            y_train = y_train[indices]
            
            error = 0
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = (batch + 1) * batch_size
                end_idx = min(num_samples,end_idx)
                
                # Forward propagation
                x_batch = x_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                y_pred = self.forward_propagation(x_batch)
                
                # get batch cost
                batch_cost = self.loss.cost(y_pred, y_batch)
                error += batch_cost
                
                # Backward propagation
                self.backward_propagation(y_pred, y_batch, learning_rate)
                
                # Store loss and epoch values for plotting
                loss_values.append(batch_cost)
                step = epoch*(num_batches+1) + batch
                steps.append(step)
                if verbose == 0:
                    pass
                elif verbose == 1:
                    pass
                elif verbose == 2:
                    print("(epoch, batch): {}, loss: {}".format(steps[-1], loss_values[-1]))
                else:
                    raise NotImplementedError
                    
            loss_epochs.append(error/(num_batches+1))
            epochs.append(epoch)
            if verbose == 1:
                print("epoch: {}, loss: {}".format(epochs[-1], loss_epochs[-1]))
            
        # Update the plot
        plt.plot(epochs, loss_epochs)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()
    
    def predict(self, x_pred):
        # Todo: Check that this impl is sufficient
        return self.forward_propagation(x_pred)