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
        predictions_new = predictions - np.max(predictions)
    else:
        maximum = np.max(predictions, axis=1)
        predictions_new = predictions - maximum[:, np.newaxis]
    predictions_new = np.exp(predictions_new)
    predictions_sum = np.sum(predictions_new, axis=(predictions.ndim - 1))
    if predictions.ndim == 1:
        probabilities = predictions_new / predictions_sum
    else:
        probabilities = predictions_new / predictions_sum[:, np.newaxis]
    return probabilities


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
    mask_target = np.zeros(probs.shape)
    if probs.ndim == 1:
        mask_target[target_index] = 1
    else:
        mask_target[tuple(np.arange(0, probs.shape[0])), tuple(target_index.T[0])] = 1

    loss = -np.sum(mask_target * np.log(probs))
    return loss


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
    if predictions.ndim == 1:
        predictions_new = predictions - np.max(predictions)
    else:
        maximum = np.max(predictions, axis=1)
        predictions_new = predictions - maximum[:, np.newaxis]
    predictions_new = np.exp(predictions_new)
    predictions_sum = np.sum(predictions_new, axis=(predictions.ndim-1))
    if predictions.ndim == 1:
        probabilities = predictions_new / predictions_sum
    else:
        probabilities = predictions_new / predictions_sum[:, np.newaxis]

    mask_target = np.zeros(probabilities.shape)
    if probabilities.ndim == 1:
        mask_target[target_index] = 1
    elif target_index.ndim == 1:
        mask_target[tuple(np.arange(0, probabilities.shape[0])), tuple(target_index)] = 1
    else:
        mask_target[tuple(np.arange(0, probabilities.shape[0])), tuple(target_index.T[0])] = 1

    loss = -np.sum(mask_target * np.log(probabilities))

    dprediction = probabilities
    dprediction[mask_target.astype(bool)] = dprediction[mask_target.astype(bool)]-1

    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = reg_strength*np.sum(W**2)
    gradient = reg_strength*2*W

    return loss, gradient
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)
    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)
    dW = np.dot(X.T, dprediction)
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

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
            batch = X[batches_indices[0]]
            target_index = y[batches_indices[0]]
            loss_cross, dW = linear_softmax(batch, self.W, target_index)
            loss_reg, grad_reg = l2_regularization(self.W, reg)
            self.W = self.W - learning_rate*(dW+grad_reg)
            loss = loss_reg + loss_cross
            loss_history.append(loss)
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        predictions = np.dot(X, self.W)
        probabilities = softmax(predictions)
        y_pred = np.argmax(probabilities, axis=1)

        return y_pred



                
                                                          

            

                
