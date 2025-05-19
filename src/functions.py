import numpy as np

#weights and biases------------------------------------------------------------
def Z(W, X, track_gradient=True):
    # W is an m by n matrix where m is the output size and n is the input size
    # the last entries in the axis=2 of W is the weights
    # X is an d by b matrix where d is the num features and b is the batch size
    # to add bias we appent ones along axis = 0 of x'
    out = np.matmul(W, np.append(X,np.ones((1,X.shape[1])), axis=0))
    if track_gradient:
      dZ_dX = W
      return(out, dZ_dX)
    else:
      return(out)
    
#activation functions----------------------------------------------------------
def relu(Z, track_gradient=True):
    if track_gradient:
      #we define the derivative of ReLU to be 0 in all flat regions, else 1
      dH_dZ = np.where(Z<=0, 0, 1)
      out = np.multiply(dH_dZ, Z)
      return(out, dH_dZ)
    else:
      return(np.minimum(Z,0))

#leaky relu is a relu variant where the flat portion is not entirely flat
#this is to avoid dead relu problem
def leaky_relu(Z, m=0.01,track_gradient=True):
    if track_gradient:
      dH_dZ = np.where(Z<=0, m*Z, 1)
      out = np.multiply(dH_dZ, Z)
      return(out, dH_dZ)
    else:
      return(np.minimum(Z,m*Z))

def sigmoid(Z, track_gradient=True):
    out = 1/(1+np.exp(-1*Z))
    if track_gradient:
      dH_dZ = 1 - out
      return(out, dH_dZ)
    else:
      return(out)

#loss functions----------------------------------------------------------------
def MSE(y, y_hat):
    error = y-y_hat
    squared_error = error*error
    mean_squared_error = np.sum(squared_error)/y.shape[1]
    return(mean_squared_error, 2*(error)/y.shape[1])