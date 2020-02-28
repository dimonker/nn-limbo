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
        self.fc1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu = ReLULayer()
        self.fc2 = FullyConnectedLayer(hidden_layer_size, n_output)

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
        
        for i_param in self.params():
            param = self.params()[i_param]
            param.grad = np.zeros_like(param.grad)
        
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        a = self.fc1.forward(X)
        b = self.relu.forward(a)
        f = self.fc2.forward(b)
                
        loss, dL = softmax_with_cross_entropy(f, y)
        
        df = self.fc2.backward(dL)
        db = self.relu.backward(df)
        da = self.fc1.backward(db)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        for i_param in self.params():
            param = self.params()[i_param]
            param_loss, param_grad = l2_regularization(param.value, self.reg)
            param.grad += param_grad
            loss += param_loss     

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
        
        a = self.fc1.forward(X)
        b = self.relu.forward(a)
        f = self.fc2.forward(b)
        pred = f.argmax(axis = 1)

        return pred

    def params(self):
        # TODO Implement aggregating all of the params
        result = {'W1': self.fc1.W, 'B1': self.fc1.B, 'W2': self.fc2.W, 'B2': self.fc2.B}

        return result
