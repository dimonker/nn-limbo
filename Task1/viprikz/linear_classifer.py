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
    # TODO implement softmax
    
    shape = predictions.shape
    
    if len(shape) == 1:
        p = predictions.copy()[None]
    else:
        p = predictions.copy()
    
    p -= np.max(p)
 
    probs = np.zeros_like(p, dtype=np.float)
    
    num_batches, num_classes = p.shape
    
    for i in range(num_batches):
        denominator = np.sum(np.exp(p[i]))
        
        for j in range(num_classes):
            probs[i][j] = np.exp(p[i][j]) / denominator
    
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
    # TODO implement cross-entropy
    
    if isinstance(target_index, int):
        target_index = np.array([target_index], np.int)
    
    num_batches, num_classes = probs.shape
    
    mean_loss = 0
    for i in range(num_batches):
        mean_loss -= np.log(probs[i][target_index[i]])
    
    mean_loss /= num_batches
    
    return mean_loss


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
    # TODO implement softmax with cross-entropy
    
    if isinstance(target_index, int):
        target_index = np.array([target_index], np.int)
        
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    
    num_batches, num_classes = probs.shape
    
    dprediction = np.zeros_like(probs, dtype=np.float)
    
    for i in range(num_batches):
        for j in range(num_classes):
            if j == target_index[i]:
                dprediction[i][j] = - 1 + probs[i][j]
            else:
                dprediction[i][j] = probs[i][j]
                
    dprediction /= num_batches   
    
    return loss, dprediction


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
    
    num_batches = X.shape[0]
    num_features, num_classes = W.shape
    
    # TODO implement prediction and gradient over W
    loss, dy = softmax_with_cross_entropy(predictions, target_index)
    
    dW = np.dot(X.T, dy)
    
    return loss, dW


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

    # TODO: implement l2 regularization and gradient
    
    loss = reg_strength * np.linalg.norm(W)**2
    grad = 2 * reg_strength * W
    
    return loss, grad
    

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

            # TODO implement generating batches from indices
            
            for ind in batches_indices:
                batches = np.array([X[i] for i in ind], np.float)
                target = np.array([y[i] for i in ind])
            # Compute loss and gradients
                loss, grad = linear_softmax(batches, self.W, target)
                l2_loss, l2_grad = l2_regularization(self.W, reg)
            # Apply gradient to weights using learning rate
                self.W -= learning_rate * (grad + l2_grad)
            # Don't forget to add both cross-entropy loss
            # and regularization!
                loss += l2_loss
                
            loss_history.append(loss)
            # end
            print(f"Epoch {epoch}, loss: {loss}")

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        
        predictions = np.dot(X, self.W)
        
        for i in range(X.shape[0]):
            n = np.argsort(predictions[i])[::-1][0]
            y_pred[i] = n
        
        return y_pred



                
                                                          

            

                
