import numpy as np
import numpy.random as rng
import tensorflow as tf

dtype = tf.float32

delta = 1e-6
softplus = lambda x: tf.math.softplus(x) + delta 
sigmoid = lambda x: tf.math.sigmoid(x) * (1-delta) + 0.5 * delta 
sigmoid2 = lambda x: sigmoid(x) * 2.0
logsigmoid = lambda x: -softplus(-x)
log = lambda x: tf.log(x*1e2)-np.log(1e2)
logit = lambda x: log(x) - log(1-x)
def softmax(x, axis=-1):
    e_x = tf.exp(x - tf.reduce_max(x,axis=axis, keepdims=True)[0])
    out = e_x / tf.reduce_sum(e_x, axis=axis, keepdims=True)
    return out

class CWNlinear:
    """
    Conditional Weight Normalization linear layer class. Constructs y = x*W + b, where the weights
    W are build using a direction matrix d, (optionally) normalized for each row and then scaled
    according to the conditional context features. Biases are also build uppon the context features.
    The weights can be masked in some way (not masked by default).
    """
    def __init__(self, in_features, out_features, context_features,
                 mask=None, norm=True, input=None, context=None):
        """
        Initializer.
        :param in_features: # of input features (x dim)
        :param out_features: # of output features (y dim)
        :param context_features: # of context features (context dim)
        :param mask: mask to multiply the weights W (None by default)
        :param norm: boolean to normalize the weights before scaling for each ouput dimension
        """
        self.in_features = in_features
        self.out_features = out_features
        self.context_features = context_features
        self.mask = mask
        self.norm = norm
        self.input = tf.placeholder(dtype=dtype,shape=[None,in_features],name='x') if input is None else input
        self.context = tf.placeholder(dtype=dtype,shape=[None,context_features],name='context') if context is None\
                                                                                                else context
        self.params = list()
        
        # Direction of weights to be normalized, independent of context features
        self.direction = tf.Variable(rng.normal(scale=0.001,
                                                size=(in_features,out_features)),
                                                dtype=dtype,name='direction')
        # Context scale and biases computed from context features
        sqrtk = context_features**-0.5
        self.cscale = tf.layers.Dense(out_features,
                                      kernel_initializer=tf.initializers.random_normal(stddev=0.001),
                                      bias_initializer=tf.initializers.random_uniform(-sqrtk,sqrtk),
                                      dtype=dtype)
        self.cbias = tf.layers.Dense(out_features,
                                     kernel_initializer=tf.initializers.random_normal(stddev=0.001),
                                     bias_initializer=tf.initializers.random_uniform(-sqrtk,sqrtk),
                                     dtype=dtype)
        
        self.scale = self.cscale(self.context)
        self.bias = self.cbias(self.context)
        
        self.params += self.cscale.weights+self.cbias.weights+[self.direction]
        
        if norm: 
            self.weight = self.direction/tf.norm(self.direction,axis=0)
        else:
            self.weight = self.direction
        #Multiply by mask if available
        if self.mask is not None:
            self.weight = self.weight * self.mask
        self.output =  self.scale * tf.matmul(self.input,self.weight) + self.bias