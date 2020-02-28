import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


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
        self.dense1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.act1 = ReLULayer()
        self.dense2 = FullyConnectedLayer(hidden_layer_size, n_output)

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
        # Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        params = self.params()
        for key in params.keys():
          params[key].grad = np.zeros_like(params[key].value)
        
      
        # Compute loss and fill param gradients
        # by running forward and backward passes through the model
        out = self.dense2.forward(self.act1.forward(self.dense1.forward(X)))

        loss, grad = softmax_with_cross_entropy(out, y)

        self.dense1.backward(self.act1.backward(self.dense2.backward(grad)))

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!

        for key in params.keys():
          l2_loss, l2_grad = l2_regularization(params[key].value, self.reg)
          loss += l2_loss
          params[key].grad += l2_grad

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
 
        out = self.dense2.forward(self.act1.forward(self.dense1.forward(X)))

        return out.argmax(axis=1)

    def params(self):
        result = {}

        result['W1'] = self.dense1.W
        result['W2'] = self.dense2.W

        result['B1'] = self.dense1.B
        result['B2'] = self.dense2.B


        return result
