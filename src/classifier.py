import tensorflow as tf
from functools import partial

import ising
import util
import common

def create_baseline_classifier(flags, input_width, num_classes, network, normalize = True):
    """ Create baseline classifier """
    l = tf.keras.layers
    inputs = tf.keras.Input(shape=[None,input_width])
    kr = common.IsingKernelRegularizer(network, flags.ising_regularization, flags.l1_reg, normalize)
    layers = [
        l.Reshape(target_shape = (input_width,), input_shape=(-1, input_width,)),
        l.Dense(num_classes, 
                kernel_regularizer = kr),
    ]
    outs   = [inputs]
    for i,layer in enumerate(layers):
        outs.append(layers[i](outs[i]))
    return tf.keras.Model(inputs = inputs, outputs = outs)

def baseline_model_fn(features, labels, mode, params):
    flags = params['flags']
    ising_graph = ising.normalize(params['network'])
    model = create_baseline_classifier(flags, params['input_width'], 2, ising_graph, flags.normalize_ising_reg)
    training = (mode != tf.estimator.ModeKeys.PREDICT)
    outs = model(features, training=training)
    logits = outs[-1]
    dense1 = outs[-1]
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits),
            'input': features,
        }
        return tf.estimator.EstimatorSpec(
            mode = mode,
            predictions = predictions,
            export_outputs = {
                'classify': tf.estimator.export.PredictOutput(predictions)
            })

    # Create loss
    ising_loss, l1_loss, l2_loss, batch_loss = (0.,0.,0.,0.)
    for l in tf.losses.get_regularization_losses():
        ising_loss += l
    batch_loss = tf.identity(batch_loss, 'batch_loss')
    """
    # These losses are added on the weights for this one
    if flags.batch_regularization:
        batch_loss += flags.batch_regularization * tf.nn.l2_loss(tf.reduce_mean(ising_layer)), 'batch_loss'
    if flags.l1_reg:
        l1_loss += flags.l1_reg * tf.reduce_sum(tf.abs(ising_layer))
    if flags.l2_reg:
        l2_loss += flags.l2_reg * tf.nn.l2_loss(ising_layer)
    """
    celoss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    loss = celoss + ising_loss + batch_loss + l1_loss + l2_loss

    # Construct metrics
    accuracy = tf.metrics.accuracy(labels = labels, predictions=tf.argmax(logits, axis=1))

    auc = tf.metrics.auc(tf.reshape(tf.equal(labels, 1), [-1,1]), 
            predictions=tf.reshape(tf.nn.softmax(logits)[:,1],[-1,1]),
                         #curve='PR',
                         summation_method='careful_interpolation',
                        )
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        #optimizer = tf.train.AdamOptimizer(learning_rate=flags.learning_rate)
        optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=flags.learning_rate)
        train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())


        # Assign convienient names for tensors in the graph
        tf.identity(flags.learning_rate, 'learning_rate')
        tf.identity(ising_loss, 'ising_loss')
        tf.identity(celoss, 'mse_loss') #TODO fix hax
        tf.identity(accuracy[1], name='train_accuracy') # could this be better maybe a mean?
        #tf.identity(auc[0], name='auc')
        tf.summary.scalar('train_accuracy', accuracy[1])
        tf.summary.scalar('ising_loss', ising_loss)
        #tf.summary.scalar('auc', auc[0])

        # Add histograms
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops={'accuracy': accuracy, 'auc': auc})

def create_classifier(flags, input_width, num_classes = 10):
    data_format, h, w = (flags.data_format, flags.height, flags.width)
    input_shape = util.get_input_shape(data_format)
    if isinstance(input_width, tuple) or isinstance(input_width, list):
        input_shape = input_width
    ci = common.custom_initializer(flags.warm_start_encoder)
    trainable_encoder = not flags.warm_start_encoder
    l = tf.keras.layers
    max_pool = l.MaxPooling2D((2,2), (2,2), padding='same', data_format = data_format)
    print("Trainable is", trainable_encoder)
    conv = partial(common.conv, trainable=trainable_encoder, initializer=ci)
    dense = partial(common.dense, trainable=trainable_encoder, initializer=ci)
    print('input shape', input_shape)
    layers = [
        l.Reshape(target_shape = (28,28,1), input_shape  = (28 * 28,)),
        #l.Reshape(target_shape = [40,40,1], input_shape  = (40*40,)),
        conv('c1', 32),
        max_pool,
        #l.Dropout(0.4),
        conv('c2', 64),
        max_pool,
        #l.Dropout(0.4),
        l.Flatten(),
        dense('d1', h*w),
        l.Dropout(0.4),
    ]
    if flags.conv_decoder:
        layers.append(l.Reshape((h,w,1), input_shape=(h*w,)))
        layers.append(common.conv('d2', 16, width=3, trainable=True, initializer=ci)) 
        layers.append(max_pool) # 4x4
        layers.append(common.conv('d3', 16, width=3, trainable=True, initializer=ci))
        layers.append(max_pool) # 2x2
        layers.append(common.conv('d4', 16, width=3, trainable=True, initializer=ci))
        #layers.append(max_pool) # 2x2
        #layers.append(common.conv('d5', 16, width=3, trainable=True, initializer=ci))
        layers.append(l.Reshape((2*2*16,)))
        #layers.append(l.Reshape((4*4*16,)))
    layers.append(l.Dense(num_classes)) 
    inputs = tf.keras.Input(shape=[None,784])
    outs   = [inputs]
    for i,layer in enumerate(layers):
        outs.append(layers[i](outs[i]))
    return tf.keras.Model(inputs = inputs, outputs = outs)

def regressor(features, labels, mode, params):
    
    flags = params['flags']
    assert flags.use_regressor
    model = create_classifier(flags, params['input_width'], num_classes=1)
    training = (mode != tf.estimator.ModeKeys.PREDICT)
    outs = model(features, training=training)
    logits = outs[-1]
    dense1 = tf.identity(outs[7], name='dense1')
    h, w = flags.height, flags.width

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'predictions': logits,
            'dense1': dense1,
            'ising_layer': dense1,
            'input': features,
        }
        return tf.estimator.EstimatorSpec(
            mode = mode,
            predictions = predictions,
            export_outputs = {
                'classify': tf.estimator.export.PredictOutput(predictions)
            })

    # Create Ising loss
    ising_loss_val = 0
    if flags.ising_regularization !=0:
        ising_graph = ising.create_laplacian(h, w, flags.ising_graph_type, flags.laplacian_type)
        ising_loss_val = ising.ising_loss_laplacian(ising_graph, dense1, weight = flags.ising_regularization)
        ising_layer = tf.identity(dense1, name='ising_layer')

    # Add l2 activation loss
    #l2_loss = tf.identity(flags.l2_regularization * tf.nn.l2_loss(dense1), 'l2_loss')
    l2_loss = tf.identity(flags.batch_regularization * tf.nn.l2_loss(tf.reduce_mean(dense1)), 'batch_loss')

    # Create total loss
    labels = tf.reshape(labels,(-1, 1))
    print(labels.shape, logits.shape)
    mse_loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)
    loss = mse_loss + ising_loss_val + l2_loss

    mse = tf.metrics.mean_squared_error(labels=labels, predictions=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=flags.learning_rate)
        train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())

        # Assign convienient names for tensors in the graph
        tf.identity(flags.learning_rate, 'learning_rate')
        tf.identity(ising_loss_val, 'ising_loss')
        tf.identity(mse_loss, 'mse_loss') #TODO fix hax
        #tf.identity(accuracy[1], name='train_accuracy') # could this be better maybe a mean?
        #tf.identity(auc[0], name='auc')
        #tf.summary.scalar('train_accuracy', accuracy[1])
        tf.summary.scalar('mse', mse_loss)
        tf.summary.scalar('ising_loss', ising_loss_val)
        #tf.summary.scalar('auc', auc[0])

        # Add histograms
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops={'mse': mse})


def model_fn(features, labels, mode, params):
    flags = params['flags']
    model = create_classifier(flags, params['input_width'])
    training = (mode != tf.estimator.ModeKeys.PREDICT)
    outs = model(features, training=training)
    logits = outs[-1]
    print(logits.shape)
    dense1 = tf.identity(outs[7], name='dense1')

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits),
            'dense1': dense1,
            'ising_layer': dense1,
            'input': features,
        }
        return tf.estimator.EstimatorSpec(
            mode = mode,
            predictions = predictions,
            export_outputs = {
                'classify': tf.estimator.export.PredictOutput(predictions)
            })

    # Log dense1 layer
    h, w = flags.height, flags.width
    tf.summary.image('dense1', 
        tf.image.resize_images(tf.reshape(dense1, 
            [flags.batch_size, h, w, 1]), 
            [h * 10, w * 10], 
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    ) # Use tf image resize to prevent auto interpolation in tensorboard

    # Create Ising loss
    ising_loss_val = 0
    if flags.ising_regularization !=0:
        ising_graph = ising.create_laplacian(h, w, flags.ising_graph_type, flags.laplacian_type)
        ising_loss_val = ising.ising_loss_laplacian(ising_graph, dense1, weight = flags.ising_regularization)
        ising_layer = tf.identity(dense1, name='ising_layer')

    # Add l2 activation loss
    #l2_loss = tf.identity(flags.l2_regularization * tf.nn.l2_loss(dense1), 'l2_loss')
    l2_loss = tf.identity(flags.batch_regularization * tf.nn.l2_loss(tf.reduce_mean(dense1)), 'batch_loss')

    # Create total loss
    celoss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    loss = celoss + ising_loss_val + l2_loss

    accuracy = tf.metrics.accuracy(labels = labels, predictions=tf.argmax(logits, axis=1))
    #it = tf.Print(i, [i], 'label_tensor:') # CAUTION: tf print output must be used to be computed

    auc = tf.metrics.auc(tf.reshape(tf.equal(labels, 7), [-1,1]), 
            predictions=tf.reshape(tf.nn.softmax(logits)[:,7],[-1,1]),
                         #curve='PR',
                         summation_method='careful_interpolation',
                        )
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=flags.learning_rate)
        train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())

        # Assign convienient names for tensors in the graph
        tf.identity(flags.learning_rate, 'learning_rate')
        tf.identity(ising_loss_val, 'ising_loss')
        tf.identity(celoss, 'mse_loss') #TODO fix hax
        tf.identity(accuracy[1], name='train_accuracy') # could this be better maybe a mean?
        #tf.identity(auc[0], name='auc')
        tf.summary.scalar('train_accuracy', accuracy[1])
        tf.summary.scalar('ising_loss', ising_loss_val)
        #tf.summary.scalar('auc', auc[0])

        # Add histograms
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops={'accuracy': accuracy, 'auc': auc})
