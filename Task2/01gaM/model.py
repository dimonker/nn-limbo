import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.layer1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.layer2 = ReLULayer()
        self.layer3 = FullyConnectedLayer(hidden_layer_size, n_output)
        
    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        
        
        for param_index in self.params():
            param = self.params()[param_index]
            param.grad = np.zeros(param.grad.shape)
            
        y1 = self.layer1.forward(X)
        y2 = self.layer2.forward(y1)
        y3 = self.layer3.forward(y2)
        
        loss, dL = softmax_with_cross_entropy(y3, y)
        
        dy3 = self.layer3.backward(dL)
        dy2 = self.layer2.backward(dy3)
        dy1 = self.layer1.backward(dy2)
        
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!

        for param_index in self.params():
            param = self.params()[param_index]
            loss_p, grad_p = l2_regularization(param.value, self.reg)
            loss += loss_p
            param.grad += grad_p
            

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)

        y1 = self.layer1.forward(X)
        y2 = self.layer2.forward(y1)
        y3 = self.layer3.forward(y2)      
        pred = np.argmax(y3, axis = 1)
        
        return pred

    def params(self):
        # TODO Implement aggregating all of the params
        result = {'W1': self.layer1.W, 'B1': self.layer1.B, 'W2': self.layer3.W, 'B2': self.layer3.B}


        return result
