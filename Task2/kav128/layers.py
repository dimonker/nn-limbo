import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    
    loss = reg_strength * np.linalg.norm(W) ** 2
    grad = reg_strength * 2 * W
    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    if len(predictions.shape) == 1:
        f = predictions.copy()
        f -= np.max(f)
        return np.exp(f) / np.sum(np.exp(f))
    else:
        f = predictions.copy()
        f -= np.max(f, axis=1).reshape((f.shape[0], -1))
        return np.exp(f) / np.sum(np.exp(f), axis=1).reshape((f.shape[0], -1))

def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss
    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss: single value
    '''
    if (type(target_index) == int):
        return -np.log(probs)[target_index]
    else:
        return -np.mean(np.log(probs[range(len(target_index)), target_index]))


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    
    sfmx = softmax(predictions)
    loss = cross_entropy_loss(sfmx, target_index)
    if type(target_index) == int:
        grad = sfmx
        grad[target_index] -= 1
        return loss, grad
    else:
        m = target_index.shape[0]
        grad = sfmx
        grad[range(m), target_index] -= 1
        return loss, grad / m


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        
        self.multiplier = (X > 0)
        return X * self.multiplier
    
    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """

        return d_out * self.multiplier

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return np.matmul(X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        
        self.W.grad += np.matmul(self.X.T, d_out)
        self.B.grad += np.sum(d_out, axis=0)
        d_input = np.matmul(d_out, self.W.value.T)
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}