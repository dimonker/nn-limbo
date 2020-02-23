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
    predictions_minus = np.array(predictions.copy())
    # predictions_minus -= np.max(predictions)
    # print(predictions.ndim)
    if predictions_minus.ndim == 1 : predictions_minus = predictions_minus[np.newaxis, :]
    predictions_minus -= np.max(predictions_minus, axis=1)[:, np.newaxis]
    sm = np.exp(predictions_minus) / np.sum(np.exp(predictions_minus), axis = 1)[:, np.newaxis]
    if predictions.ndim == 1 : return sm[0]
    return sm


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
    if type(target_index) == int: return -np.log(probs[target_index])
    return - np.mean(np.log(probs[range(target_index.shape[0]), target_index]))


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
    softmax_res = softmax(predictions)    
    loss = cross_entropy_loss(softmax_res, target_index)
    dprediction = softmax_res
    if type(target_index) == int: 
      dprediction[target_index] -= 1
      return loss, dprediction
    dprediction[range(target_index.shape[0]), target_index] -= 1
    return loss, dprediction/target_index.shape[0]


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
    loss = reg_strength*(W**2).sum()
    grad = 2 * reg_strength * W
    return loss, grad
    

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
    loss, dW = softmax_with_cross_entropy(predictions, target_index)
    dW = np.dot(X.T, dW)

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

            for indices in batches_indices:
              batch_x = X[indices]
              batch_y = y[indices]
            
              loss, grad = linear_softmax(batch_x, self.W, batch_y)
              loss_l2, grad_l2 = l2_regularization(self.W, reg)

              loss += loss_l2
              grad += grad_l2

              loss_history.append(loss)
              self.W -= learning_rate*grad


            if (epoch+1)%10 == 0: print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        return np.argmax(softmax(np.dot(X, self.W)), axis = 1)



                
                                                          

            

                
