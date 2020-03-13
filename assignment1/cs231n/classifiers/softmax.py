import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    N = X.shape[0]
    C = W.shape[1]
    D = W.shape[0]
    softmax = np.zeros((N,C))
    ############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    
   
     ## softmax loss
    for i in range (N):
        for j in range (C):
            for k in range (D):
                softmax[i,j] += X[i,k]*W[k,j] 
        softmax[i] -= np.max(softmax[i]) # numerical stability
        softmax[i,:] = np.exp(softmax[i,:])
        softmax[i,:] /= np.sum(softmax[i,:])
        
    for i in range (N):
        loss -= np.log(softmax[i,y[i]])
    loss /= N
    loss+= 0.5*reg*np.sum(W**2)
    
    ## gradient of W
    
    softmax[np.arange(N),y] -=1 # softmax[i,y[i]]  all Li and diff of sfm is pi-1. (Li = -log(pi))
    for i in range (N):
        for j in range (D):
            for k in range (C):
                dW [j,k] += X[i,j]*softmax[i,k]
                
    dW /= N
    dW += reg*W
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    N = X.shape[0]
    scores = np.dot(X, W)
    scores = scores - np.max(scores,axis=1,keepdims =True)
    softmax = np.exp(scores)
    
    softmax /= np.sum(softmax, axis = 1, keepdims = True)
    loss -= np.sum(np.log(softmax[np.arange(N),y]))
    loss /= N
    loss += 0.5 * reg *np.sum(W**2)
    
    softmax[np.arange(N),y] -= 1
    dW = np.dot(X.T,softmax)
    dW /= N
    dW += reg*W
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

