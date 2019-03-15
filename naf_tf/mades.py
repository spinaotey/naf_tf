import tensorflow as tf
import numpy as np
import numpy.random as rng
import naf_tf.utils.nn as nn_
from naf_tf.utils.nn import log

dtype = tf.float32

from functools import reduce

def get_rank(max_rank, num_out):
    """
    Creates rank_out vector with values from 0 to max_rank-1 of size num_out.
    :param max_rank: maximum rank of output vector
    :param num_out: size of output vector
    :return: vector rank_out of size num_out
    """
    rank_out = np.array([])
    while len(rank_out) < num_out:
        rank_out = np.concatenate([rank_out, np.arange(max_rank)])
    excess = len(rank_out) - num_out
    remove_ind = rng.choice(max_rank,excess,False)
    rank_out = np.delete(rank_out,remove_ind)
    rng.shuffle(rank_out)
    return rank_out.astype('float32')
    

def get_mask_from_ranks(r1, r2):
    """
    Creates binary mask of zeros and ones to connect ranks properly.
    :param r1: input rank
    :param r2: output rank
    :return: binary matrix to connect r1 to r2
    """
    return (r2[:, None] >= r1[None, :]).astype('float32')

def get_masks_all(ds, fixed_order=False, derank=1):
    """
    Creates list of masks for all hidden layers given their dimensions in ds.
    Also outputs the random rank assigned to each variable at the beggining.
    :params ds: list of dimensions dx, d1, d2, ... dh, dx (h hidden layers)
    :params fixed_order: fix order of inputs. If False, they will be randomized
    :params derank: only used for self connection, dim > 1
    :return: (ms,rx), ms list of masks, rx rank order of iputs.
    """
    dx = ds[0]
    ms = list()
    rx = get_rank(dx, dx)
    if fixed_order:
        rx = np.sort(rx)
    r1 = rx
    if dx != 1:
        for d in ds[1:-1]:
            r2 = get_rank(dx-derank, d)
            ms.append(get_mask_from_ranks(r1, r2))
            r1 = r2
        r2 = rx - derank
        ms.append(get_mask_from_ranks(r1, r2))
    else:
        ms = [np.zeros([ds[i+1],ds[i]]).astype('float32') for \
              i in range(len(ds)-1)]
    if derank==1:
        assert np.all(np.diag(reduce(np.dot,ms[::-1])) == 0), 'wrong masks'
    
    return ms, rx


def get_masks(dim, dh, num_layers, num_outlayers, fixed_order=False, derank=1):
    """
    Creates all masks for a MADE for different number of outlayers for each 
    input dimension.
    :params dim: input dimension
    :params dh: hidden layers dimension
    :params num_layers: number of hidden layers
    :params num_outlayers: number of output layers for each input dimension
    :params fixed_order: fix order of inputs. If False, they will be randomized
    :params derank: only used for self connection, dim > 1
    :return: (ms,rx), ms list of masks, rx rank order of iputs.
    """
    ms, rx = get_masks_all([dim,]+[dh for i in range(num_layers-1)]+[dim,],
                           fixed_order, derank)
    ml = ms[-1]
    ml_ = (ml.transpose(1,0)[:,:,None]*([np.cast['float32'](1),] *\
                           num_outlayers)).reshape(
                           dh, dim*num_outlayers).transpose(1,0)
    ms[-1]  = ml_
    ms = [m.T for m in ms]
    return ms, rx

class cMADE:
    """
    Conditional MADE class using conditional weight normalization as conection with masks.
    """
    def __init__(self, dim, hid_dim, context_dim, num_layers,
                 num_outlayers=1, activation=tf.nn.elu, fixed_order=False,
                 derank=1, input=None, context=None):
        """
        Class initializer.
        :params dim: input dimension
        :params hid_dim: hidden layers dimension
        :params context_dim: context dimension of the conditional space
        :params num_layers: number of hidden layers
        :params num_outlayers: number of output layers for each input dimension
        :params activation: tensorflow activation function between layers
        :params fixed_order: fix order of inputs. If False, they will be randomized
        :params derank: only used for self connection, dim > 1
        """
        
        self.dim = dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.context_dim = context_dim
        self.num_outlayers = num_outlayers
        self.activation = activation
        self.params = list()
        self.input = tf.placeholder(dtype=dtype,shape=[None,in_features],name='x') if input is None else input
        self.context = tf.placeholder(dtype=dtype,shape=[None,context_features],name='context') if context is None\
                                                                                                else context
        
        # Get masks as tensorflow tensors and input order
        ms, rx = get_masks(dim, hid_dim, num_layers, num_outlayers,
                           fixed_order, derank)
        ms = [tf.convert_to_tensor(m,dtype=dtype) for m in ms]
        self.ms = ms
        self.rx = rx
        
        for i in range(num_layers-1):
            if i==0:
                layer = nn_.CWNlinear(dim,hid_dim,context_dim,ms[i],False,self.input,self.context)
                h = activation(layer.output)
                self.params += layer.params
            else:
                layer = nn_.CWNlinear(hid_dim,hid_dim,context_dim,ms[i],False,h,self.context)
                h = activation(layer.output)
                self.params += layer.params
        layer = nn_.CWNlinear(hid_dim, dim*num_outlayers, context_dim, ms[-1],input=h,context=self.context)
        self.output = tf.reshape(layer.output,[-1,self.dim, self.num_outlayers])
        self.params += layer.params