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
        self.layer_1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu = ReLULayer()
        self.layer_2 = FullyConnectedLayer(hidden_layer_size, n_output)

        #raise Exception("Not implemented!")

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

        params['W1'].grad = np.zeros_like(params['W1'].value)
        params['W2'].grad = np.zeros_like(params['W2'].value)
        params['B1'].grad = np.zeros_like(params['B1'].value)
        params['B2'].grad = np.zeros_like(params['B2'].value)
        #raise Exception("Not implemented!")
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        L1 = self.layer_1.forward(X)
        R1 = self.relu.forward(L1)
        L2 = self.layer_2.forward(R1)
        loss, loss_grad = softmax_with_cross_entropy(L2, y)
        
        d_w2 = self.layer_2.backward(loss_grad)
        d_relu = self.relu.backward(d_w2)
        d_w1 = self.layer_1.backward(d_relu)

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        #raise Exception("Not implemented!")
        #l2_reg = np.array([2, 1])
        for i in params:
          temp = l2_regularization(params[i].value, self.reg)
          loss += temp[0]
          params[i].grad += temp[1]

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
        #pred = np.zeros(X.shape[0], np.int)
        L1 = self.layer_1.forward(X)
        R1 = self.relu.forward(L1)
        L2 = self.layer_2.forward(R1)
        #L2_normalized = L2 - np.max(L2, axis=1)[:, np.newaxis]
        #soft_max = np.exp(L2_normalized) / np.sum(np.exp(L2_normalized), axis=1)[:, np.newaxis]
        pred = np.argmax(L2, axis=1)
        
        #raise Exception("Not implemented!")
        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params
        result['W1'] = self.layer_1.params()['W']
        result['W2'] = self.layer_2.params()['W']
        result['B1'] = self.layer_1.params()['B']
        result['B2'] = self.layer_2.params()['B']
        #raise Exception("Not implemented!")

        return result
