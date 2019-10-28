import re
from functools import partial, reduce

import numpy as np
import tensorflow as tf
from tensorflow.python.pywrap_tensorflow import NewCheckpointReader

import ising
from common import custom_initializer, dense, conv
import common
import util

def create_small_autoencoder(flags, input_width):
    print('WARNING: SMALL NETWORK')
    h, w, lnum, mw = (flags.height, flags.width, flags.layer_num, flags.middle_width)
    trainable_encoder = not flags.warm_start_encoder
    print('Trainable Encoder:', trainable_encoder)
    ci = custom_initializer(flags.warm_start_encoder)
    d = partial(dense, trainable=trainable_encoder, initializer=ci)
    l = tf.keras.layers
    ws = [h*w, input_width]
    if isinstance(input_width, tuple):
        inputs  = l.Reshape((1, reduce(lambda x,y: x*y, input_width),), input_shape=input_width)(inputs)
    else:
        inputs  = tf.keras.Input(shape=[None, input_width])
    embed   = d('embed',        ws[0], activation=tf.nn.leaky_relu)(inputs)
    outputs = dense('output',   ws[1], initializer=ci)(embed)
    return tf.keras.Model(inputs = inputs, outputs = [embed, embed, embed, embed, embed, outputs])

class FixedInitializer(tf.keras.initializers.Initializer):
    def __init__(self, mat):
        self.mat = mat
    def __call__(self, shape, dtype=None, partition_info=None):
        return self.mat

def create_autoencoder(flags, input_width):
    h, w, lnum, mw = (flags.height, flags.width, flags.layer_num, flags.middle_width)
    trainable_encoder = not flags.warm_start_encoder
    print('Trainable Encoder:', trainable_encoder)
    ci = custom_initializer(flags.warm_start_encoder)
    d = partial(dense, trainable=trainable_encoder, initializer=ci)
    l = tf.keras.layers
    ws = [50,50,mw,50,50,input_width]
    ws[lnum] = h*w
    matmul_layer = partial(l.Dense, use_bias=False, trainable=False)
    embed_act = None
    if flags.relu_embedding:
        embed_act = tf.nn.relu
    if isinstance(input_width, tuple):
        inputs  = tf.keras.Input(shape=input_width)
        inputs  = l.Reshape((1, reduce(lambda x,y: x*y, input_width),), input_shape=input_width)(inputs)
    else:
        inputs  = tf.keras.Input(shape=[None, input_width])
    if flags.add_input_noise > 0:
        n0      = l.GaussianNoise(flags.add_input_noise)(inputs)
        e1      = d('e1',           ws[0])(n0)
    else:
        e1      = d('e1',           ws[0])(inputs)
    e2      = d('e2',           ws[1])(e1)
    #e2      = d('e3', 1)(e15)
    if flags.add_noise:
        n1 = l.GaussianNoise(1)(e2)
        embed   = d('embed',    ws[2], activation=embed_act)(n1)
    elif flags.softmax_embedding:
        s1 = d('s1', ws[2], activation=None)(e2)
        ainv = np.transpose(ising.load_kernel_mat(flags.heat_kernel_path, 'backwardshat'))
        a    = np.transpose(ising.load_kernel_mat(flags.heat_kernel_path, 'forwardshats'))
        #ainv = np.transpose(ising.load_kernel_mat(flags.heat_kernel_path, 'backward'))
        #a    = np.transpose(ising.load_kernel_mat(flags.heat_kernel_path, 'forward'))
        #a,ainv = ising.get_heat_kernels(1,ws[2],"rings", power=5, subsample=2)
        s2 = matmul_layer(kernel_initializer=FixedInitializer(ainv), units=a.shape[0])(s1)
        #s3 = l.Maximum(name='filter_loadings')(s2)
        s3 = l.Softmax(name='filter_loadings')(s2)
        embed = matmul_layer(kernel_initializer=FixedInitializer(a), units=ws[2])(s3)
        #embed = l.Softmax(name='embed')(s1)
    else:
        embed   = d('embed',    ws[2], activation=embed_act)(e2)
    d1      = d('d1',           ws[3])(embed)
    d2      = dense('d2',       ws[4], initializer=ci)(d1)
    outputs = dense('output',   ws[5], initializer=ci)(d2)
    if flags.softmax_embedding: return tf.keras.Model(inputs = inputs, outputs = [e1, e2, embed, d1, s2, outputs])
    return tf.keras.Model(inputs = inputs, outputs = [e1, e2, embed, d1, d2, outputs])

def create_autoencoder_convolutional_decoder(flags, input_width):
    h, w, lnum, mw = (flags.height, flags.width, flags.layer_num, flags.middle_width)
    assert h == 8 and w == 8
    l = tf.keras.layers
    ci = custom_initializer(flags.warm_start_encoder)
    d = partial(dense, trainable=not flags.warm_start_encoder, initializer=ci)
    c = partial(conv, initializer=ci)
    ws = [50,50,mw,h*w]

    inputs  = tf.keras.Input(shape=[None, input_width])
    e1      = d('e1',           ws[0])(inputs)
    e2      = d('e2',           ws[1])(e1)
    embed   = d('embed',        ws[2], activation=None)(e2)
    d1      = d('d1',           ws[3])(embed)
    tmp     = l.Reshape((h,w,1), input_shape = [h*w])(d1)
    d2      = c('d2', 16)(tmp)
    du2     = l.UpSampling2D((2,2))(d2)
    d3      = c('d3', 16)(du2)
    du3     = l.UpSampling2D((2,2))(d3)
    outputs = l.Conv2D(1, 5, name='output')(du3)
    outputs2 = l.Reshape((input_width,))(outputs)
    return tf.keras.Model(inputs = inputs, outputs = [e1, e2, embed, d1, outputs, outputs2])

def create_autoencoder_convolutional_decoder_1d(flags, input_width):
    h, w, lnum, mw = (flags.height, flags.width, flags.layer_num, flags.middle_width)
    assert lnum==2
    l = tf.keras.layers
    ci = custom_initializer(flags.warm_start_encoder)
    d = partial(dense, trainable=not flags.warm_start_encoder, initializer=ci)
    c = partial(conv, initializer=ci)
    ws = [50,50,mw]
    ws[lnum] = h*w

    inputs  = tf.keras.Input(shape=[None, input_width])
    e1      = d('e1',           ws[0])(inputs)
    e2      = d('e2',           ws[1])(e1)
    embed   = d('embed',        ws[2], activation=None)(e2)
    #d1      = d('d1',           ws[3])(embed)
    tmp     = l.Reshape((h*w,1), input_shape = [h*w])(embed)
    d1      = tf.layers.conv1d(tmp, 16, 2, padding='SAME')
    m1      = tf.layers.max_pooling1d(d1, 2, 2)
    d2      = tf.layers.conv1d(m1, 16, 2, padding='SAME')
    m2      = tf.layers.max_pooling1d(d2, 2, 2)
    d3      = tf.layers.conv1d(m2, 32, 3, padding='SAME')
    d4      = tf.layers.flatten(d3)
    outputs = d('out', input_width)(d4)
    #outputs2 = l.Reshape((input_width,))(outputs)
    return tf.keras.Model(inputs = inputs, outputs = [e1, e2, embed, d1, d3, outputs])

def create_convolutional_autoencoder(flags, input_width):
    data_format, h, w = (flags.data_format, flags.height, flags.width)
    input_shape = util.get_input_shape(data_format)
    if isinstance(input_width, tuple) or isinstance(input_width, list):
        input_shape = input_width
    ci = common.custom_initializer(flags.warm_start_encoder)
    trainable_encoder = not flags.warm_start_encoder
    l = tf.keras.layers
    max_pool = l.MaxPooling2D((2,2), (2,2), padding='same', data_format = data_format)
    max_pool4 = l.MaxPooling2D(4, 4, padding='same', data_format = data_format)
    print("Trainable is", trainable_encoder)
    conv = partial(common.conv, trainable=trainable_encoder, initializer=ci)
    dense = partial(common.dense, trainable=trainable_encoder, initializer=ci)
    layers = [
        l.Reshape(target_shape = input_shape, input_shape  = (28 * 28,)),
        #l.Reshape(target_shape = [40,40,1], input_shape  = (40*40,)),
        conv('c1', 64), max_pool, # 14 x 14
        conv('c2', 64), max_pool, # 7 x 7
        conv('c3', 64),
        l.Flatten(),
        dense('embed', h*w, activation=None),
        l.Reshape(target_shape = [h,w,1], input_shape=(h*w,)),
        conv('dc01', 32, width=3), max_pool4, # 4x4
        conv('dc2', 32, width=3), max_pool, # 2x2
        conv('dc3', 32, width=2), max_pool, # 1x1x16
        l.UpSampling2D(7),
        conv('dc4', 32, width=5), l.UpSampling2D(2), # 14x14x32
        conv('dc5', 32, width=5), l.UpSampling2D(2), # 14x14x32
        conv('dc6', 1, width=5),
        l.Flatten(),
    ]
    inputs = tf.keras.Input(shape=[None,784])
    outs   = [inputs]
    for i,layer in enumerate(layers):
        outs.append(layers[i](outs[i]))
    ising = outs[8]
    outputs = [ising, ising, ising, ising, ising, outs[-1]]
    for i,out in enumerate(outs):
        print(i+1,out.shape)
    return tf.keras.Model(inputs = inputs, outputs = outputs)

def model_fn(features, labels, mode, params):
    input_width = params['input_width']
#    if isinstance(input_width, tuple):
#        print('input width is tuple')
#        print(input_width)
#        input_width = reduce(lambda x,y: x*y, input_width)
#        print(input_width)
    flags = params['flags']
    mfn = create_autoencoder
    if flags.conv_decoder:
        if flags.height == 1:
            mfn = create_autoencoder_convolutional_decoder_1d
        else:
            mfn = create_autoencoder_convolutional_decoder
    if flags.conv_autoencoder:
        mfn = create_convolutional_autoencoder
    #mfn = create_autoencoder_convolutional_decoder if flags.conv_decoder else create_autoencoder
    #mfn = create_autoencoder_convolutional_decoder if flags.conv_decoder else create_small_autoencoder
    model = mfn(flags, input_width)
    training = (mode != tf.estimator.ModeKeys.PREDICT)
    outs = model(features, training=training)
    e1, e2, embed, d1, d2, logits = outs
    true_labels = labels
    labels = features
    #print('label shape', labels.shape)
    ising_layer = tf.identity(outs[flags.layer_num], 'ising_layer')

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'prediction': logits,
            'e2': e2,
            'd1': d1,
            'embedding': embed,
            'ising_layer': ising_layer,
            'input' : features,
        }
        return tf.estimator.EstimatorSpec(
            mode = mode,
            predictions = predictions,
            export_outputs = {
                'classify': tf.estimator.export.PredictOutput(predictions)
            })

    h, w = flags.height, flags.width
    # Create Ising loss
    ising_loss_val = 0
    if flags.decay < 1:
        ir = tf.train.exponential_decay(
                learning_rate=flags.ising_regularization,
                global_step = tf.train.get_global_step(),
                decay_steps=100,
                decay_rate=flags.decay,
                staircase=True)
    else:
        ir = flags.ising_regularization
    if flags.ising_regularization != 0:
        if flags.add_labels_to_ising:
            ising_laplacian = ising.create_laplacian(h+1, w, flags.ising_graph_type, flags.laplacian_type)
            ising_loss_val = ising.ising_loss_laplacian(ising_laplacian, ising_layer, 
                    fixed_values=tf.one_hot(true_labels, 10), weight = ir)
        else:
            ising_laplacian = ising.create_laplacian(h, w, flags.ising_graph_type, flags.laplacian_type)
            ising_loss_val = ising.ising_loss_laplacian(ising_laplacian, ising_layer, 
                    fixed_values=None, weight = ir)
    def hinge_loss(vec):
        abs_vec = tf.abs(vec)
        vec_reg = tf.where(abs_vec > 0.5,
                           tf.sqrt(abs_vec),
                           abs_vec)
        vec_reg = tf.where(tf.is_nan(vec_reg),
                           tf.zeros_like(vec_reg),
                           vec_reg)
        return tf.reduce_sum(vec_reg)

    l2_loss = tf.identity(0., 'l2_loss')
    
    #l1_loss = tf.identity(0., 'l1_loss')
    #l1_hinge_loss = tf.identity(0., 'l1_hinge_loss')
    #if flags.l2_reg > 0:
    l2_loss += flags.l2_reg * tf.nn.l2_loss(ising_layer)
    l1_loss = flags.l1_reg * tf.reduce_sum(tf.abs(ising_layer))
    l1_hinge_loss = flags.l1_hinge_reg * hinge_loss(ising_layer)
    #l2_loss += tf.reduce_sum(tf.abs(ising_layer)) * 0.0001
    
    heat_loss = 0
    if flags.heat_reg > 0:
        if flags.heat_kernel_path:
            heat_kernel = ising.load_kernel_mat(flags.heat_kernel_path, flags.heat_kernel_name)
        else:
            heat_kernel = ising.create_weighted_eigenvector_kernel(h, w, flags.ising_graph_type)
        heat_loss = flags.heat_reg * ising.heat_loss(heat_kernel, ising_layer)
    #heat_loss = flags.heat_reg * ising.heat_loss2(heat_kernel, ising_layer)

    if flags.batch_regularization > 0:
        #batch_loss = tf.identity(flags.batch_regularization * tf.reduce_sum(tf.abs(tf.reduce_mean(ising_layer, axis=0))), 'batch_loss')
        batch_loss = tf.identity(flags.batch_regularization * tf.reduce_sum(tf.square(tf.reduce_mean(ising_layer, axis=0))), 'batch_loss')
    else:
        batch_loss = tf.identity(0., 'batch_loss') # for Tensor Logging purposes

    # Create total loss
    mseloss = tf.losses.mean_squared_error(labels, logits)
    loss = mseloss + ising_loss_val + batch_loss + l2_loss + l1_loss + l1_hinge_loss + heat_loss
    lr = flags.learning_rate

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-5)
        train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())

        tf.identity(mseloss, 'mse_loss')
        tf.identity(lr, 'learning_rate')
        tf.identity(ir, 'ising_weight')
        tf.identity(ising_loss_val, 'ising_loss')
        tf.identity(l1_loss, 'l1_loss')
        tf.identity(l1_hinge_loss, 'l1_hinge_loss')
        tf.identity(heat_loss, 'heat_loss')
        tf.summary.scalar('mse', mseloss)

        # Add histograms
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops={'rmse': tf.metrics.root_mean_squared_error(labels, logits)})

