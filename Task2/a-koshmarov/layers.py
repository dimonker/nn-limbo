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
    loss = reg_strength * np.linalg.norm(W)**2
    grad = reg_strength * 2 * W

    return loss, grad

def softmax(predictions):
    pred = predictions.copy()
    if predictions.ndim == 1:
      pred -= np.max(pred)
      return np.exp(pred) / np.sum(np.exp(pred))
    else:
      pred -= np.max(predictions, axis=1).reshape(-1, 1)
      return np.exp(pred)/np.sum(np.exp(pred), axis=1).reshape(-1, 1)
    

def cross_entropy_loss(probs, target_index):
    if probs.ndim == 1:
      return -np.log(probs[target_index])
    else:
      loss = 0

      for i in range(probs.shape[0]):
        loss -= np.log(probs[i, target_index[i]])
      # return -np.mean(np.log(probs[np.arange(probs.shape[0]), target_index]))
      return loss/probs.shape[0]



def softmax_with_cross_entropy(preds, target_index):
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
    s = softmax(preds)
    loss = cross_entropy_loss(s, target_index)

    if preds.ndim == 1: 
      s[target_index] -= 1
    else:
      for i in range(s.shape[0]):
        s[i, target_index[i]] -= 1
      s /= s.shape[0]
    return loss, s


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
        self.X = X
        pred = np.maximum(0, X)
        return pred

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
        d_result = d_out * (self.X >= 0)
        return d_result

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
        pred = np.dot(X, self.W.value) + self.B.value
        return pred

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
        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = np.sum(d_out, axis=0)[:, np.newaxis].T
        d_input = np.dot(d_out, self.W.value.T)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
