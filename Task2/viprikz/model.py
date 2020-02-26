import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        hidden_layer_size, int - number of neurons in the hidden layer
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        reg, float - L2 regularization strength
        """
        self.reg = reg
        
        # TODO Create necessary layers
        self.layer1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.layer_relu = ReLULayer()
        self.layer2 = FullyConnectedLayer(hidden_layer_size, n_output)

        
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
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model

        y1 = self.layer1.forward(X)  # (batch_size, N)
        y2 = self.layer_relu.forward(y1)  # (batch_size, N)
        y3 = self.layer2.forward(y2)  # (batch_size, num_classes)
        
        batch_size = X.shape[0]
        loss = 0
        dy3 = np.zeros_like(y3, dtype=np.float)
        
        probs = np.zeros_like(y3, dtype=np.float)
        
        for i in range(batch_size):
            # Computing loss
            denominator = np.sum(np.exp(y3[i]))
            probs[i] = np.exp(y3[i]) / denominator
            
            loss -= np.log(probs[i][y[i]])
            
            # Computing gradient of loss with respect to y3(second layer output)
            
            for j in range(y3.shape[1]):
                if j == y[i]:
                    dy3[i][j] = -1 + probs[i][j]
                else:
                    dy3[i][j] = probs[i][j]
               
        loss /= batch_size
        dy3 /= batch_size
        
        dy2 = self.layer2.backward(dy3)
        dy1 = self.layer_relu.backward(dy2)
        dX = self.layer1.backward(dy1)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        params = self.params().values()
        
        l2_loss = self.reg * np.sum(np.linalg.norm(x.value)**2 for x in params)
        
        loss += l2_loss
        
        for param in params:
            param.grad += 2 * self.reg * param.value
        
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
        
        y1 = self.layer1.forward(X)  # (batch_size, N)
        y2 = self.layer_relu.forward(y1)  # (batch_size, N)
        y3 = self.layer2.forward(y2)  # (batch_size, num_classes)
                
        for i in range(X.shape[0]):
            ind = np.argsort(np.exp(y3[i]))[::-1][0]
            pred[i] = ind
            
        return pred

    def params(self):
        # TODO Implement aggregating all of the params
        params1 = self.layer1.params()
        params2 = self.layer2.params()
        result = {'W1': params1['W'], 'B1': params1['B'], 'W2': params2['W'], 'B2': params2['B']}
        return result
