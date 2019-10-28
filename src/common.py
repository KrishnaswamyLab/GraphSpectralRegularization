"""
Common model functions
"""
import re
from functools import partial

import tensorflow as tf
from tensorflow.python.pywrap_tensorflow import NewCheckpointReader

class custom_initializer():
    def __init__(self, warm_start_encoder_dir):
        self.regex = '(^(e|d1|c).*)'
        self.warm_start = warm_start_encoder_dir
        if self.warm_start:
            self.reader = NewCheckpointReader(tf.train.latest_checkpoint(self.warm_start))
    def init_kernel(self, name, shape, dtype=None, partition_info=None):
        return self.get_tensor(name + '/kernel', shape, 'he_normal')
    def init_bias(self, name, shape, dtype=None, partition_info=None):
        return self.get_tensor(name + '/bias', shape, 'zeros')
    def get_tensor(self, name, shape, default):
        if not self.warm_start:
            return tf.keras.initializers.get(default)(shape)
        if not re.match(self.regex, name):
            return tf.keras.initializers.get(default)(shape)
        tensor = self.reader.get_tensor(name)
        print("Load", name, shape)
        if list(tensor.shape) != shape:
            raise RuntimeError('Shape Mismatch on layer %s loaded %s expected %s' 
                               %(name, list(tensor.shape), shape))
        return tensor

def dense(name, width=50, activation=tf.nn.leaky_relu, trainable=True, initializer=None, **kwargs): 
    return tf.keras.layers.Dense(
        width, 
        activation = activation,
        name=name,
        kernel_initializer=partial(initializer.init_kernel, name),
        bias_initializer=partial(initializer.init_bias, name),
        trainable=trainable,
        **kwargs,
    )

def conv(name, depth, width=5, activation = tf.nn.leaky_relu, trainable=True, initializer=None):
    return tf.keras.layers.Conv2D(
            depth, 
            width, 
            padding='same', 
            activation=activation, 
            kernel_initializer=partial(initializer.init_kernel, name),
            bias_initializer=partial(initializer.init_bias, name),
            name=name,
            trainable=trainable,
    )

class IsingKernelRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, L, w=0., l1=0., normalize=True):
        self.w = w
        self.L = L
        self.l1 = l1
        self.normalize = normalize
    def __call__(self, x):
        r = 0.
        print('WARNING: Ising of abs(x)')
        x = tf.abs(x)
        numerator = tf.matmul(tf.transpose(x), tf.matmul(self.L, x)) # [Out x Out]
        if self.normalize:
            denominator = tf.matmul(tf.transpose(x), x)  # [Out x Out]
            r += self.w * tf.trace(tf.divide(numerator, denominator))
        else:
            r += self.w * tf.trace(numerator)
        #r = tf.Print(r, [r], 'reg')
        r += self.l1 * tf.reduce_sum(tf.abs(x))
        return r

def test_ising_kernel_regularizer():
    import dataset

