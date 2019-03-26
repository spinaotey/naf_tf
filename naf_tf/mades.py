import tensorflow as tf
import numpy as np
import numpy.random as rng

dtype = tf.float32

def create_degrees(dim, n_hiddens, n_outputs, input_order, mode):
    """
    Generates a degree for each hidden and input unit. A unit with degree d can only receive input from units with
    degree less than d.
    :param n_inputs: the number of inputs
    :param n_hiddens: a list with the number of hidden units
    :param input_order: the order of the inputs; can be 'random', 'sequential', or an array of an explicit order
    :param mode: the strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
    :return: list of degrees
    """

    degrees = []

    # create degrees for inputs
    if isinstance(input_order, str):

        if input_order == 'random':
            degrees_0 = np.arange(1, dim + 1)
            rng.shuffle(degrees_0)

        elif input_order == 'sequential':
            degrees_0 = np.arange(1, dim + 1)

        else:
            raise ValueError('invalid input order')

    else:
        input_order = np.array(input_order)
        assert np.all(np.sort(input_order) == np.arange(1, dim + 1)), 'invalid input order'
        degrees_0 = input_order
    degrees.append(degrees_0)

    # create degrees for hiddens
    if mode == 'random':
        for N in n_hiddens:
            min_prev_degree = min(np.min(degrees[-1]), dim - 1)
            degrees_l = rng.randint(min_prev_degree, dim, N)
            degrees.append(degrees_l)

    elif mode == 'sequential':
        for N in n_hiddens:
            degrees_l = np.arange(N) % max(1, dim - 1) + min(1, dim - 1)
            degrees.append(degrees_l)

    else:
        raise ValueError('invalid mode')
        
    degrees.append(np.repeat(degrees_0,n_outputs))

    return degrees

def create_masks(degrees):
    """
    Creates the binary masks that make the connectivity autoregressive.
    :param degrees: a list of degrees for every layer
    :return: list of all masks, as theano shared variables
    """

    Ms = []

    for l, (d0, d1) in enumerate(zip(degrees[:-1], degrees[1:])):
        M = d0[:, np.newaxis] <= d1
        M = tf.constant(M, dtype=dtype, name='M' + str(l+1))
        Ms.append(M)

    return Ms

def create_weights(dim, n_hiddens, n_outputs):
    """
    Creates all learnable weight matrices and bias vectors.
    :param n_inputs: the number of inputs
    :param n_hiddens: a list with the number of hidden units
    :param n_comps: number of gaussian components
    :return: weights and biases, as tensorflow variables
    """

    Ws = []
    bs = []

    n_units = np.concatenate(([dim], n_hiddens, [dim*n_outputs]))

    for l, (N0, N1) in enumerate(zip(n_units[:-1], n_units[1:])):
        W = tf.Variable((rng.randn(N0, N1) / np.sqrt(N0 + 1)), dtype=dtype, name='W' + str(l+1))
        b = tf.Variable(np.zeros([1,N1]), dtype=dtype, name='b' + str(l+1))
        Ws.append(W)
        bs.append(b)
    
    return Ws, bs

def create_weights_conditional(dim, context_dim, n_hiddens, n_outputs):
    """
    Creates all learnable weight matrices and bias vectors for a conditional made.
    :param n_inputs: the number of (conditional) inputs
    :param n_outputs: the number of outputs
    :param n_hiddens: a list with the number of hidden units
    :param n_comps: number of gaussian components
    :return: weights and biases, as tensorflow variables
    """

    Wx = tf.Variable(rng.randn(context_dim, n_hiddens[0]) / np.sqrt(context_dim + 1), dtype=dtype, name='Wx')

    return (Wx,) + create_weights(dim, n_hiddens, n_outputs)

class cMADE:
    """
    Implements a Made, where each conditional probability is modelled by a single gaussian component. The made has
    inputs which is always conditioned on, and whose probability it doesn't model.
    """
    def __init__(self, dim, hid_dim, context_dim, num_layers,
                 num_outlayers=1, activation=tf.nn.elu, output_order='sequential', 
                 mode='sequential', input=None, context=None):
        
        self.dim = dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.context_dim = context_dim
        self.num_outlayers = num_outlayers
        self.activation = activation
        self.mode = mode
        self.hiddens = [hid_dim for _ in range(num_layers)]
        self.params = list()
        self.input = tf.placeholder(dtype=dtype,shape=[None,in_features],name='x') if input is None else input
        self.context = tf.placeholder(dtype=dtype,shape=[None,context_features],name='context') if context is None\
                                                                                                else context
        # create network's parameters
        degrees = create_degrees(dim, self.hiddens, num_outlayers, output_order, mode)
        Ms = create_masks(degrees)
        Wx, Ws, bs = create_weights_conditional(dim, context_dim, self.hiddens, num_outlayers)
        self.parms = [Wx] + Ws + bs
        self.output_order = degrees[0]
        
        # feedforward propagation
        f = activation
        h = f(tf.matmul(self.context, Wx) + tf.matmul(self.input, Ms[0] * Ws[0]) + bs[0],name='h1')
        for l, (M, W, b) in enumerate(zip(Ms[1:-1], Ws[1:-1], bs[1:-1])):
            h = f(tf.matmul(h, M * W) + b,name='h'+str(l + 2))
        h = tf.matmul(h, Ms[-1] * Ws[-1]) + bs[-1]
        self.output = tf.reshape(h,[-1,self.dim, self.num_outlayers],name='output')