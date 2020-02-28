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
        # TODO Create necessary layers

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
        params = self.params()
        for key in params.keys():
            params[key].grad = np.zeros_like(params[key].value)

        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model

        layer_result = X
        for i in range(len(self.layers)):
            layer_result = self.layers[i].forward(layer_result)
        
        loss, grad = softmax_with_cross_entropy(layer_result, y)

        dX = grad
        for i in reversed(range(len(self.layers))):
            dX = self.layers[i].backward(dX)
        
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
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused

        layer_result = X
        for i in range(len(self.layers)):
            layer_result = self.layers[i].forward(layer_result)
        
        return np.argmax(layer_result, axis=1)

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params

        result['W1'] = self.layers[0].W
        result['B1'] = self.layers[0].B
        result['W2'] = self.layers[2].W
        result['B2'] = self.layers[2].B

        return result
