import numpy as np

def softmax(predictions):

    prediction = predictions.copy()
    len_pred = len(predictions.shape)

    if (len_pred == 1):
        e_prob = np.exp(prediction - np.max(prediction))
        probs = np.array(list(map(lambda x: x / np.sum(e_prob), e_prob)))
    else:
        pred = list(map(lambda x: x - np.max(x), prediction))
        e_prob = np.exp(pred)
        probs = np.array(list(map(lambda x: x / np.sum(x), e_prob)))

    return probs

def cross_entropy_loss(probs, target_index):

    len_probs = len(probs.shape)

    if (len_probs == 1):
        loss = -np.log(probs[target_index])
    else:
        batch_size = np.arange(target_index.shape[0])
        loss = np.sum(-np.log(probs[batch_size,target_index.flatten()])) / target_index.shape[0]

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
    # TODO: Copy from the previous assignment
    loss = reg_strength * np.sum(W**2)
    grad = reg_strength * 2 * W

    return loss, grad


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
    # TODO: Copy from the previous assignment
    prediction = preds.copy()
    len_pred = len(preds.shape)
    len_target = 1
    probs = softmax(preds)
    dprediction = probs
    loss = cross_entropy_loss(probs, target_index)

    if (len_pred == 1):
        dprediction[target_index] -= 1
    else:        
        batch_size = np.arange(target_index.shape[0])
        dprediction[batch_size, target_index.flatten()] -= 1
        len_target = target_index.shape[0]

    return loss, dprediction/len_target


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
        self.X = X
        
        return np.maximum(X, 0)

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
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result = np.where(self.X > 0, d_out, 0)
        
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
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        result = np.dot(X, self.W.value) + self.B.value
        self.X = X
        return result

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
        self.B.grad += np.sum(d_out, axis = 0)
        self.W.grad += np.dot(self.X.T, d_out)
        d_input = np.dot(d_out, (self.W.value).T)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
