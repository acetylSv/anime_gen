import numpy as np
import tensorflow as tf

def linear(input_, output_size, stddev=0.02, name='linear'):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        W = tf.get_variable("W", [shape[-1], output_size], tf.float32,
                                     tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable("b", [output_size])
    return tf.matmul(input_, W) + b

def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
    name='conv2d', padding='SAME'):
    with tf.variable_scope(name):
        W = tf.get_variable('W',
                [k_h, k_w, input_.get_shape()[-1], output_dim],
                initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, W, strides=[1, d_h, d_w, 1], padding=padding)

        b = tf.get_variable('b', [output_dim],
            initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, b)
    return conv

def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name='deconv2d'):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        W = tf.get_variable('W',
                [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, W, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        b = tf.get_variable('b', [output_shape[-1]],
            initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, b)
    return deconv

def batch_norm(x, name='batch_norm'):
    epsilon = 1e-5
    momentum = 0.9
    y = tf.contrib.layers.batch_norm(
            x, decay=momentum, updates_collections=None,
            epsilon=epsilon, scale=True, is_training=True,
            scope=name)
    return y

def instance_norm(x, name='instance_norm'):
    axis = [1,2] # for format: NHWC
    epsilon = 1e-5
    mean, var = tf.nn.moments(x, axis, keep_dims=True)
    return (x - mean) / tf.sqrt(var+epsilon)

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


attr_dict = {
    'orange hair':1, 'white hair':2, 'aqua hair':3, 'gray hair':4,
    'green hair':5, 'red hair':6, 'purple hair':7, 'pink hair':8,
    'blue hair':9, 'black hair':10, 'brown hair':11, 'blonde hair':12,
    'no_hair_color':13,

    'short hair':14, 'long hair':15,
    'no_hair_length':16,

    'gray eyes':17, 'bicolored eyes':18, 'black eyes':19, 'orange eyes':20,
    'pink eyes':21, 'yellow eyes':22, 'aqua eyes':23, 'purple eyes':24,
    'green eyes':25, 'brown eyes':26, 'red eyes':27, 'blue eyes':28,
    'no_eyes_color':29
    }
inv_map = {v: k for k, v in attr_dict.items()}

def attr_lookup(vec):
    attrs = []
    for attr_idx in list(np.where(vec==1))[0]:
        attrs.append(inv_map[attr_idx+1])
    return attrs

def attr_txt2vec(txt):
    # one attr to one vec
    attrs = txt.strip().split(',')
    vec = np.zeros(29)
    for attr_txt in attrs:
        vec[attr_dict[attr_txt]-1] = 1
    return vec

'''
class ResidualBlock() :
    def __init__(self,name,filters,filter_size=3,non_linearity=Lrelu,normal_method=InstanceNorm) :
        self.conv_1 = Conv2d(name+'_1',filters,filters,filter_size,filter_size,1,1)
        self.normal = normal_method(name+'_norm')
        self.nl = non_linearity()
        self.conv_2 = Conv2d(name+'_2',filters,filters,filter_size,filter_size,1,1)
    def __call__(self,input_var) :
        _t = self.conv_1(input_var)
        _t = self.normal(_t)
        _t = self.nl(_t)
        _t = self.conv_2(_t)
        return input_var + _t
'''
