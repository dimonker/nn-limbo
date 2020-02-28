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
        self.layers = [
            FullyConnectedLayer(n_input, hidden_layer_size),
            ReLULayer(),
            FullyConnectedLayer(hidden_layer_size, n_output)
        ]

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
        for key, param in self.params().items():
            param.grad = np.zeros_like(param.value)
                
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
#         layers[0].forward()
        for layer in self.layers:
            X = layer.forward(X)
        
        loss, grad = softmax_with_cross_entropy(X, y)
        
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        for key, param in self.params().items():
            reg_loss, reg_grad = l2_regularization(param.value, self.reg)
            loss += reg_loss
            param.grad += reg_grad

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
        for layer in self.layers:
            X = layer.forward(X)

        return np.argmax(X, axis=1)

    def params(self):
        result = {
            'W0': self.layers[0].params()['W'],
            'B0': self.layers[0].params()['B'],
            'W2': self.layers[2].params()['W'],
            'B2': self.layers[2].params()['B']
        }

        return result
