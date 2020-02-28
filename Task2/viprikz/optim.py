import numpy as np


class SGD:
    """
    Implements vanilla SGD update
    """
    def update(self, w, d_w, learning_rate):
        """
        Performs SGD update

        Arguments:
        w, np array - weights
        d_w, np array, same shape as w - gradient
        learning_rate, float - learning rate

        Returns:
        updated_weights, np array same shape as w
        """
        
        if w.shape[0] == 1:
            d_w = np.mean(d_w, axis=0)
        return w - d_w * learning_rate


class MomentumSGD:
    """
    Implements Momentum SGD update
    """
    def __init__(self, momentum=0.9):
        self.momentum = 0.9
        self.v = 0
    
    def update(self, w, d_w, learning_rate):
        """
        Performs Momentum SGD update

        Arguments:
        w, np array - weights
        d_w, np array, same shape as w - gradient
        learning_rate, float - learning rate

        Returns:
        updated_weights, np array same shape as w
        """
        # TODO Implement momentum update
        # Hint: you'll need to introduce some variables to remember
        # velocity from the previous updates
        
        self.v = self.momentum * self.v - learning_rate * d_w # integrate velocity
        if w.shape[0] == 1:
            self.v = np.mean(self.v, axis=0)
        
        w = w + self.v # integrate position
            
        return w
