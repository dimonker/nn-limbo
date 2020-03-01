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
        self.fulllayer1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.reglayer1 = ReLULayer()
        self.fulllayer2 = FullyConnectedLayer(hidden_layer_size, n_output)

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
        self.fulllayer1.W.grad = np.zeros_like(self.fulllayer1.W.grad)
        self.fulllayer1.B.grad = np.zeros_like(self.fulllayer1.B.grad)
        self.fulllayer2.W.grad = np.zeros_like(self.fulllayer2.W.grad)
        self.fulllayer2.B.grad = np.zeros_like(self.fulllayer2.B.grad)


        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        res = self.fulllayer1.forward(X)
        res2 = self.reglayer1.forward(res)
        res3 = self.fulllayer2.forward(res2)

        loss, grad = softmax_with_cross_entropy(res3, y)

        back3 = self.fulllayer2.backward(grad)
        back2 = self.reglayer1.backward(back3)
        back = self.fulllayer1.backward(back2)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!

        for params in self.params().keys():
          # print(params)
          # print(self.params()[params].value)
          loc_loss, loc_grad = l2_regularization(self.params()[params].value, self.reg)
          loss += loc_loss
          self.params()[params].grad += loc_grad

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
        res = self.fulllayer1.forward(X)
        res2 = self.reglayer1.forward(res)
        # pred = np.argmax(self.fulllayer2.forward(res2), axis=0)
        pred = np.argmax(self.fulllayer2.forward(res2), axis=1)
        return pred

    def params(self):
        # TODO Implement aggregating all of the params
        result = {'W1':self.fulllayer1.W, 'B1':self.fulllayer1.B, 'W2':self.fulllayer2.W, 'B2':self.fulllayer2.B}
        
        return result
