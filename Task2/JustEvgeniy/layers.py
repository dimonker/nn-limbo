import numpy as np


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
    if predictions.ndim == 1:
        probs = predictions - np.max(predictions)
        probs = np.exp(probs)
        probs /= probs.sum()
        assert np.isclose(probs.sum(), 1)
    else:
        batch_size = predictions.shape[0]
        probs = predictions - np.max(predictions, axis=1).reshape(batch_size, 1)
        probs = np.exp(probs)
        probs /= probs.sum(axis=1).reshape(batch_size, 1)
        assert np.all(np.isclose(probs.sum(axis=1), 1))
    
    return probs


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
    if probs.ndim == 1:
        loss = -np.log(probs[target_index])
    else:
        batch_size = probs.shape[0]
        target_index = (np.arange(batch_size), target_index.reshape(batch_size))
        loss = np.mean(-np.log(probs[target_index]))
    
    return loss

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
    loss = np.sum(W**2) * reg_strength
    grad = 2 * reg_strength * W

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    """
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
    """
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs.copy()
    
    if predictions.ndim == 1:
        dprediction[target_index] -= 1
    else:
        batch_size = predictions.shape[0]
        ti = (np.arange(batch_size), target_index.reshape(batch_size))
        dprediction[ti] -= 1
        dprediction /= batch_size
    
    return loss, dprediction


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
        self.yes_grad = X > 0
        return X * self.yes_grad

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
        return d_out * self.yes_grad

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
        return np.dot(X, self.W.value) + self.B.value

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
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        self.B.grad += np.sum(d_out, axis=0).reshape(1, -1)
        self.W.grad += self.X.T.dot(d_out)

        return d_out.dot(self.W.value.T)

    def params(self):
        return {'W': self.W, 'B': self.B}
