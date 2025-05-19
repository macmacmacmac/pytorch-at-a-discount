import numpy as np
from dollarstore_torch.functions import Z

#just a function describing various weight initialization paradigms
#in general, setting weights to be all zeros or ones is a bad idea
def init_weights(in_dim, out_dim, option=''):
    if option == 'kaiming_normal':
      W = np.random.normal(0,np.sqrt(2/in_dim),size=(out_dim, in_dim))
      b = np.zeros((out_dim,))
      return(np.hstack([W, b.reshape(-1, 1)]))
    elif option == 'kaiming_uniform':
      bound = np.sqrt(6/in_dim)
      W = np.random.uniform(-bound, +bound, size=(out_dim, in_dim))
      b = np.zeros((out_dim,))
      return(np.hstack([W, b.reshape(-1, 1)]))
    else:
      W = np.random.normal(0,0.5,size=(out_dim, in_dim))
      b = np.zeros((out_dim,))
      return(np.vstack([W, b.reshape(-1, 1)]))
    
def forward_layer(layer, X, in_eval_mode=False):
    if not in_eval_mode:
      if 'weight' in layer:
          W = layer['weight']
          return(Z(W, X))
      elif 'activation' in layer:
          return(layer['activation'](X))
      elif 'loss' in layer:
          return(layer['loss'](layer['Y'], X))
    else:
      if 'weight' in layer:
          W = layer['weight']
          return(Z(W, X, track_gradient=False))
      elif 'activation' in layer:
          return(layer['activation'](X, track_gradient=False))
    return(X,True)

def forward_network(network, X, in_eval_mode=False):
  for layer in network:
      output, gradient = forward_layer(layer, X, in_eval_mode)
      layer['output'] = output
      layer['gradient'] = gradient
      X = output

def back_propagate(network, verbose=False, lr=0.0005):
  delta = 1
  layer_num = len(network)-1
  for layer in reversed(network):
    #delta here is always of shape d by b
    if 'weight' in layer:
      layer_gradient = layer['gradient']
      #delta is d[layer+1] by b
      #prev layer is d[layer]+1 by b
      #for the bias term
      prev_layer_activation = network[layer_num-1]['output']
      prev_layer_activation = np.append(
          prev_layer_activation,
          np.ones((1,prev_layer_activation.shape[1]
          )), axis=0
      )
      #need to turn this shit into a d[layer+1] by d[layer]+1 by b matrix
      prev_dim, batch_dim = prev_layer_activation.shape
      delta_dim = delta.shape[0]
      reshape_prev_layer_activation = prev_layer_activation.reshape(1,prev_dim,batch_dim).repeat(delta_dim,axis=0)
      reshaped_delta = delta.reshape(delta_dim,1,batch_dim).repeat(prev_dim,axis=1)
      # (d.reshape(2,1,3).repeat(13,axis=1)
      #*
      # inp.reshape(1,13,3).repeat(2,axis=0)).mean(axis=2)
      update = (reshape_prev_layer_activation*reshaped_delta).mean(axis=2)

      weight = layer['weight']
      if verbose:
        print(f'\n\nThis is the delta: \n{delta}')
        print(f'\n\nThis is input: \n{prev_layer_activation}')
        print(f'\n\nThis is weights pre update: \n{weight}')
        print(f'\n\nThis is the update: \n{update}')

      layer['weight'] = layer['weight'] + update*lr

      delta = np.matmul(layer_gradient[:,:-1].T,delta)
    else:
      layer_gradient = layer['gradient']
      delta = delta * layer_gradient
    layer_num-=1
  return(delta)