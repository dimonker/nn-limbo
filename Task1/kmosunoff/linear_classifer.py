import numpy as np


def softmax(predictions):
    """
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    """

    axis = predictions.ndim - 1

    m = np.max(predictions, axis=axis)
    if predictions.ndim == 2:
        m = m.reshape(-1, 1)
    probabilities = np.exp(predictions - m)

    s = np.sum(probabilities, axis=axis)
    if predictions.ndim == 2:
        s = s.reshape(-1, 1)
    probabilities = np.divide(probabilities, s)

    return probabilities


def cross_entropy_loss(probabilities, target_index):
    """
    Computes cross-entropy loss

    Arguments:
      probabilities: np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    """

    if probabilities.ndim == 1:
        loss = - np.log(probabilities[target_index])
    else:
        loss = np.sum(- np.log(probabilities[np.arange(probabilities.shape[0]), target_index]))

    return loss


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions: np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value - cross-entropy loss
      dpredictions: np array same shape as predictions - gradient of predictions by loss value
    """

    probabilities = softmax(predictions)
    target_index = target_index.reshape(-1)
    loss = cross_entropy_loss(probabilities, target_index)
    p = np.zeros(predictions.shape)
    if predictions.ndim == 1:
        p[target_index] = 1
    else:
        p[np.arange(p.shape[0]), target_index] = 1
    dpredictions = probabilities - p

    return loss, dpredictions


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

    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W

    return loss, grad
    

def linear_softmax(X, W, target_index):
    """
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    """

    predictions = np.dot(X, W)
    loss, dpredictions = softmax_with_cross_entropy(predictions, target_index)
    dW = np.dot(X.T, dpredictions)

    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self, W=None):
        self.W = W

    def copy(self):
        return LinearSoftmaxClassifier(self.W.copy())

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        """
        Trains linear classifier

        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        """

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            for batch in batches_indices:
                loss, dW = linear_softmax(X[batch], self.W, y[batch])
                l2_loss, l2_grad = l2_regularization(self.W, reg)
                self.W -= learning_rate * (dW + l2_grad)
                loss_history.append(loss + l2_loss)

            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        predictions = np.dot(X, self.W)
        y_pred = np.argmax(predictions, axis=1)
        return y_pred



                
                                                          

            

                
