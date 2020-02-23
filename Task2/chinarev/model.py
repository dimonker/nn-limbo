import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, cross_entropy_loss, softmax


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
        self.layer1 = FullyConnectedLayer(n_input, n_output)
        self.layer2 = ReLULayer()
        self.layer3 = FullyConnectedLayer(n_output, n_output)
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
        #raise Exception("Not implemented!")
        self.layer1.W.grad = np.zeros_like(self.layer1.W.grad)
        self.layer3.W.grad = np.zeros_like(self.layer3.W.grad)
        self.layer1.B.grad = np.zeros_like(self.layer1.B.grad)
        self.layer3.B.grad = np.zeros_like(self.layer3.B.grad)
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        out1 = self.layer1.forward(X)
        out2 = self.layer2.forward(out1)
        out3 = self.layer3.forward(out2)
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        #raise Exception("Not implemented!")
        loss, grad = softmax_with_cross_entropy(out3, y) #+ l2_regularization(W, reg)
        back3 = self.layer3.backward(grad) # бэкпропагатион 
        reg_loss3 , dw_dr3 = l2_regularization(self.layer3.W.value, self.reg) # считаем регуляризационные члены
        self.layer3.W.grad += dw_dr3
        
        back2 = self.layer2.backward(back3)
        
        back1 = self.layer1.backward(back2)
        reg_loss1 , dw_dr1 = l2_regularization(self.layer1.W.value, self.reg)
        self.layer1.W.grad += dw_dr1
        
        return loss+reg_loss1+reg_loss3

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
        out1 = self.layer1.forward(X)
        out2 = self.layer2.forward(out1)
        out3 = self.layer3.forward(out2)
        pred = np.argmax(out3, axis=1)
        #raise Exception("Not implemented!")
        return pred

    def params(self):
        result = {'layer1.W': self.layer1.W, 'layer1.B': self.layer1.B, 'layer3.W': self.layer3.W, 'layer1.B': self.layer3.B}

        # TODO Implement aggregating all of the params

        #raise Exception("Not implemented!")

        return result