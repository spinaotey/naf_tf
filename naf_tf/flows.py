import numpy as np
import numpy.random as rng
import tensorflow as tf
import naf_tf.mades as mades
import naf_tf.utils.nn as nn_
from naf_tf.mades import cMADE
from naf_tf.utils.nn import log
from functools import reduce

dtype = tf.float32

class SigmoidFlow:
    
    def __init__(self, num_ds_dim=4):
        self.num_ds_dim = num_ds_dim
        
        self.act_a = lambda x: nn_.softplus(x)
        self.act_b = lambda x: x
        self.act_w = lambda x: nn_.softmax(x,axis=2)
        
    def apply(self, x, logdet, dsparams, mollify=0.0, delta=nn_.delta):
        
        ndim = self.num_ds_dim
        a_ = self.act_a(dsparams[:,:,0*ndim:1*ndim])
        b_ = self.act_b(dsparams[:,:,1*ndim:2*ndim])
        w = self.act_w(dsparams[:,:,2*ndim:3*ndim])
        
        a = a_ * (1-mollify) + 1.0 * mollify
        b = b_ * (1-mollify) + 0.0 * mollify
        
        pre_sigm = a * x[:,:,None] + b
        sigm = tf.sigmoid(pre_sigm)
        x_pre = tf.reduce_sum(w*sigm, axis=2)
        x_pre_clipped = x_pre * (1-delta) + delta * 0.5
        x_ = log(x_pre_clipped) - log(1-x_pre_clipped)
        xnew = x_
        
        logj = tf.nn.log_softmax(dsparams[:,:,2*ndim:3*ndim], axis=2) + \
               nn_.logsigmoid(pre_sigm) + \
               nn_.logsigmoid(-pre_sigm) + log(a)
        
        logj = tf.reduce_sum(tf.math.reduce_logsumexp(logj,axis=2,keepdims=True),axis=2)
        logdet_ = logj + np.log(1-delta) - \
        (log(x_pre_clipped) + log(-x_pre_clipped+1))
        logdet = tf.reduce_sum(logdet_,axis=1) + logdet
        
        
        return xnew, logdet
    
class IAF_DSF:
    
    def __init__(self, dim, hid_dim, context_dim, num_layers,
                 activation=tf.nn.elu, fixed_order=False,
                 num_ds_dim=4, num_ds_layers=1, num_ds_multiplier=3,
                 input=None,context=None):
        mollify=0.0
        self.dim = dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.num_ds_dim = num_ds_dim
        self.num_ds_layers = num_ds_layers
        self.input = tf.placeholder(dtype=dtype,shape=[None,dim],name='x') if input is None else input
        if context_dim == 0:
            self.context = tf.reshape(tf.fill(tf.shape(self.input)[:1], np.float32(1.)),[-1,1])
        else:
            self.context = tf.placeholder(dtype=dtype,shape=[None,context_dim],name='context') if context is None\
                                                                                               else context
        logdet = tf.fill(tf.shape(self.input)[:1], np.float32(0.))
        
        self.model = cMADE(dim, hid_dim, int(self.context.shape[1]), num_layers, 
                           num_ds_multiplier*(hid_dim//dim)*num_ds_layers, 
                           activation, fixed_order,input=self.input,
                           context=self.context)
        self.parms = self.model.params
        self.convlayer = tf.layers.Conv1D(3*num_ds_layers*num_ds_dim,1)
        self.MADEout = self.model.output
        self.dsparams = self.convlayer(self.MADEout)
        self.sf = SigmoidFlow(num_ds_dim)
        nparams = self.num_ds_dim*3
        h = self.input
        for i in range(self.num_ds_layers):
            params = self.dsparams[:,:,i*nparams:(i+1)*nparams]
            h, logdet = self.sf.apply(h, logdet, params, mollify)
        self.output = h
        self.logdet = tf.reshape(logdet,[-1,1])
        self.L = -0.5 * dim * np.log(2 * np.pi) - 0.5 * tf.reduce_sum(self.output ** 2, axis=1,keepdims=True) + \
                 self.logdet
        self.trn_loss = -tf.reduce_mean(self.L)
        
    def eval(self, X, sess, log=True,batch_size=100000):
        """
        Evaluate log probabilities for given input-context pairs.
        :param X: a pair (x, y) where x rows are inputs and y rows are context variables
        :param sess: tensorflow session where the graph is run
        :param log: whether to return probabilities in the log domain
        :param batch_size: number of samples to be evaluated in one batch
        :return: log probabilities: log p(y|x)
        """
        
        if context_dim == 0:
            n = len(X)//batch_size
            lprob = []
            for i in range(n):
                lprob.append(sess.run(self.L,feed_dict={self.input:x[i*batch_size:(i+1)*batch_size]}))

            if n != len(y)/batch_size:
                lprob.append(sess.run(self.L,feed_dict={self.input:X[n*batch_size:]}))
            
        else:
            x, y = xy
            n = len(y)//batch_size
            lprob = []
            for i in range(n):
                lprob.append(sess.run(self.L,feed_dict={self.input:y[i*batch_size:(i+1)*batch_size],
                                                        self.context:x[i*batch_size:(i+1)*batch_size]}))

            if n != len(y)/batch_size:
                lprob.append(sess.run(self.L,feed_dict={self.input:y[n*batch_size:],
                                                        self.context:x[n*batch_size:]}))
        
        lprob = np.row_stack(lprob)

        return lprob if log else np.exp(lprob)
    
class DenseSigmoidFlow:
    
    def __init__(self, in_dim, hidden_dim, out_dim):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        self.act_a = lambda x: nn_.softplus(x)
        self.act_b = lambda x: x
        self.act_w = lambda x: nn_.softmax(x,axis=3)
        self.act_u = lambda x: nn_.softmax(x,axis=3)
        self.u_ = tf.Variable(rng.uniform(-0.001, 0.001,size=[hidden_dim, in_dim]),dtype=dtype)
        self.w_ = tf.Variable(rng.uniform(-0.001, 0.001,size=[out_dim, hidden_dim]),dtype=dtype)
        self.parms = [self.u_,self.w_]

    def apply(self, x, logdet, dsparams):
        inv = np.log(np.exp(1-nn_.delta)-1) 
        ndim = self.hidden_dim
        pre_u = self.u_[None,None,:,:]+dsparams[:,:,-self.in_dim:][:,:,None,:]
        pre_w = self.w_[None,None,:,:]+dsparams[:,:,2*ndim:3*ndim][:,:,None,:]
        a = self.act_a(dsparams[:,:,0*ndim:1*ndim]+inv)
        b = self.act_b(dsparams[:,:,1*ndim:2*ndim])
        w = self.act_w(pre_w)
        u = self.act_u(pre_u)
        
        pre_sigm = tf.reduce_sum(u * a[:,:,:,None] * x[:,:,None,:], axis=3) + b
        sigm = tf.sigmoid(pre_sigm)
        x_pre = tf.reduce_sum(w*sigm[:,:,None,:], axis=3)
        x_pre_clipped = x_pre * (1-nn_.delta) + nn_.delta * 0.5
        x_ = log(x_pre_clipped) - log(1-x_pre_clipped)
        xnew = x_
        
        logj =  tf.nn.log_softmax(pre_w, axis=3) + \
            nn_.logsigmoid(pre_sigm[:,:,None,:]) + \
            nn_.logsigmoid(-pre_sigm[:,:,None,:]) + log(a[:,:,None,:])
        # n, d, d2, dh
        
        logj = logj[:,:,:,:,None] + tf.nn.log_softmax(pre_u, axis=3)[:,:,None,:,:]
        # n, d, d2, dh, d1
        
        logj = tf.reduce_sum(tf.math.reduce_logsumexp(logj,axis=3,keepdims=True),axis=3)
        # n, d, d2, d1
        
        logdet_ = logj + np.log(1-nn_.delta) - \
            (log(x_pre_clipped) + log(-x_pre_clipped+1))[:,:,:,None]
        
        
        logdet = tf.reduce_sum(tf.math.reduce_logsumexp(
            logdet_[:,:,:,:,None] + logdet[:,:,None,:,:], axis=3,keepdims=True),axis=3)
        # n, d, d2, d1, d0 -> n, d, d2, d0
        
        return xnew, logdet
    
class IAF_DDSF:
    
    def __init__(self, dim, hid_dim, context_dim, num_layers,
                 activation=tf.nn.elu, fixed_order=False,
                 num_ds_dim=4, num_ds_layers=1, num_ds_multiplier=3,
                 input=None,context=None):
        
        self.dim = dim
        self.context_dim = context_dim
        self.num_ds_dim = num_ds_dim
        self.num_ds_layers = num_ds_layers
        
        self.input = tf.placeholder(dtype=dtype,shape=[None,dim],name='x') if input is None else input
        if context_dim == 0:
            self.context = tf.reshape(tf.fill(tf.shape(self.input)[:1], np.float32(1.)),[-1,1])
        else:
            self.context = tf.placeholder(dtype=dtype,shape=[None,context_dim],name='context') if context is None\
                                                                                               else context
        logdet = tf.fill(tf.concat([tf.shape(self.input),
                                    tf.constant([1,1],dtype=tf.int32)],axis=0),
                         np.float32(0.))

        self.model = cMADE(dim, hid_dim, int(self.context.shape[1]), num_layers, 
                   num_ds_multiplier*(hid_dim//dim)*num_ds_layers, 
                   activation, fixed_order,input=self.input,
                   context=self.context)
        
        num_dsparams = 0
        self.sf = list()
        self.MADEout = self.model.output
        
        for i in range(num_ds_layers):
            if i == 0:
                in_dim = 1
            else:
                in_dim = num_ds_dim
            if i == num_ds_layers-1:
                out_dim = 1
            else:
                out_dim = num_ds_dim
          
            u_dim = in_dim
            w_dim = num_ds_dim
            a_dim = b_dim = num_ds_dim
            num_dsparams += u_dim + w_dim + a_dim + b_dim
            
            dsf = DenseSigmoidFlow(in_dim,num_ds_dim,out_dim)
            self.sf.append(dsf)
        
        self.convlayer = tf.layers.Conv1D(num_dsparams,1)
        dsparams = self.convlayer(self.MADEout)
        
        h = self.input[:,:,None]
        start = 0
        
        for i in range(self.num_ds_layers):
            if i == 0:
                in_dim = 1
            else:
                in_dim = self.num_ds_dim
            if i == self.num_ds_layers-1:
                out_dim = 1
            else:
                out_dim = self.num_ds_dim
            
            u_dim = in_dim
            w_dim = self.num_ds_dim
            a_dim = b_dim = self.num_ds_dim
            end = start + u_dim + w_dim + a_dim + b_dim
            
            params = dsparams[:,:,start:end]
            h, logdet = self.sf[i].apply(h,logdet, params)
            start = end
        
        assert out_dim == 1, 'last dsf out dim should be 1'
        
        self.output = h[:,:,0]
        self.logdet = tf.reduce_sum(logdet[:,:,0,0],axis=1,keepdims=True)
        
        self.L = -0.5 * dim * np.log(2 * np.pi) - 0.5 * tf.reduce_sum(self.output ** 2, axis=1,keepdims=True) + \
                 self.logdet
        self.trn_loss = -tf.reduce_mean(self.L)
        # TODO Define self.parms