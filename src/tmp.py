import numpy as np
#from tensorflow.estimator import tf.estimator.ModeKeys
import tensorflow as tf
import argparse
import dataset
from magenta.models.image_stylization.image_utils import form_image_grid

class GraphRegularizer(tf.keras.regularizers.Regularizer):
    """Graph based Regularizer.

    Graph Regularization is of the form l * (G x) where x is the weights of a layer.

    Arguments:
        G: [Layer width] x [Layer width] similarity matrix of floats.
        l: Float; Graph regularization factor
    """
    def __init__(self, G = None, l = 0.):
        self.G = tf.constant(G, dtype=tf.float32)
        self.l = l

    def __call__(self, x):
        G = tf.tile(tf.expand_dims(self.G, 0), [x.shape[0], 1,1])
        x = tf.expand_dims(x, -1)
        regularization = tf.reduce_mean(self.l * tf.matmul(G, x))
        return regularization


class Namespace():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def update(self, **kwargs):
        self.__dict__.update(kwargs)


_TENSORS_TO_LOG = dict((x, x) for x in ['learning_rate',])
#                                        'cross_entropy',
#                                        'train_accuracy'])

default_args = Namespace(**{
    'batch_size'            : 100,
    'train_epochs'          : 100,
    'data_format'           : 'channels_last', # Faster on GPU
    'data_dir'              : '/tmp/mnist_data',
    'model_dir'             : '/tmp/mnist_auto_model',
    'epochs_between_evals'  : 1,
    'export_dir'            : './exported',
    'learning_rate'         : 1e-2,
})

def create_model(data_format):
    if data_format == 'channels_first':
        input_shape = [1,28,28]
    elif data_format == 'channels_last':
        input_shape = [28,28,1]
    else:
        raise Exception("Invalid data format")

    l = tf.keras.layers
    dense = lambda width: l.Dense(
        width, 
        activation = tf.nn.tanh, 
        kernel_regularizer = tf.keras.regularizers.l2(l=1e-5)
        #kernel_regularizer = GraphRegularizer(G = np.identity(width), l=1e-2,)
    )
    inputs = tf.keras.Input(shape=[None, 784])
    e1      = dense(50)(inputs)
    e2      = dense(50)(e1)
    middle  = l.Dense(2)(e2)
    d1      = dense(50)(middle)
    d2      = dense(50)(d1)
    outputs = dense(28*28)(d2)
    return tf.keras.Model(inputs = inputs, outputs = [outputs, e1, e2, middle])

#    return tf.keras.Sequential([
#        dense(50),
#        dense(50),
#        l.Dense(2),
#        dense(50),
#        dense(50),
#        dense(28*28),
#    ])

def custom_loss(labels = None, logits = None):
    zeros = tf.zeros_like(logits, dtype=logits.dtype)
    assert labels.get_shape().merge_with(logits.get_shape())
    cond = (logits >= zeros)
    relu_logits = tf.where(cond, logits, zeros)
    neg_abs_logits = tf.where(cond, -logits, logits)
    return tf.reduce_mean(tf.add(relu_logits - logits * labels, tf.log1p(tf.exp(neg_abs_logits))))

def graph_layer_loss(G, activations, batch_size, weight = 1):
    """ Imposes a loss on the activations of a layer of activations

    Let a = activations, then computes scalar value l a^T G a

    Args:
        G: [layer_width] x [layer_width]
        activations: [Batch] x [layer_width]
    """
    G = tf.expand_dims(tf.constant(G, dtype=tf.float32), 0)
    G = tf.tile(G, [batch_size, 1, 1])
    a = tf.expand_dims(activations, -1)
    at = tf.expand_dims(activations, 1)
    return weight * tf.reduce_mean(tf.matmul(tf.matmul(at, G), a))

def model_fn(features, labels, mode, params):
    model = create_model(params['data_format'])
    lr    = params['learning_rate']
    labels = features
    train = (mode != tf.estimator.ModeKeys.PREDICT)
    logits, e1, e2, middle = model(features, training = train)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'prediction' : tf.nn.sigmoid(logits),
            'middle'   : middle,
        }
        return tf.estimator.EstimatorSpec(
            mode = tf.estimator.ModeKeys.PREDICT,
            predictions = predictions,
            export_outputs = {
                'classify' : tf.estimator.export.PredictOutput(predictions),
            })

    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = labels, logits = logits)
    #loss = custom_loss(labels = labels, logits = logits
    #loss += graph_layer_loss(np.identity(2), middle, params['flags'].batch_size, weight = 1e10)

    def create_summaries(x, latent, output):
        def layer_grid_summary(name, var, image_dims, grid_dims = [10,10]):
            print(var)
            prod = np.prod(image_dims)
            grid = form_image_grid(tf.reshape(var, [params['flags'].batch_size, prod]), grid_dims, image_dims, 1)
            return tf.summary.image(name, grid, max_outputs = 1)
        layer_grid_summary("Input", x, [28,28])
        layer_grid_summary("Encoder", latent, [2,1])
        #layer_grid_summary("e2", e2, [50,1], grid_dims = [1,100])
        layer_grid_summary("Output", output, [28,28])
        tf.summary.image("e2", tf.reshape(e2, [1, params['flags'].batch_size, 50, 1]), max_outputs=1)
    create_summaries(features, middle, logits)

    total_loss = loss
    #total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = tf.train.exponential_decay(
            learning_rate=lr,
            global_step = tf.train.get_global_step(),
            decay_steps=1000, 
            decay_rate=0.96, 
            staircase=True
        )
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        tf.identity(learning_rate, 'learning_rate')

        # Add histograms
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        return tf.estimator.EstimatorSpec(
            mode = tf.estimator.ModeKeys.TRAIN,
            loss = total_loss,
            train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step()))

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode = tf.estimator.ModeKeys.EVAL,
            loss = total_loss,
            eval_metric_ops={
                'rmse' : tf.metrics.root_mean_squared_error(labels, logits),
            })

def run_mnist(flags):
    mnist_classifier = tf.estimator.Estimator(
        model_fn = model_fn,
        model_dir = flags.model_dir,
        params = {
            'data_format' : flags.data_format,
            'learning_rate' : flags.learning_rate,
            'flags' : flags,
        },
        config = tf.estimator.RunConfig(session_config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3)))
    )

    def get_train_input():
        ds = dataset.train(flags.data_dir).cache().shuffle(buffer_size=100000).batch(flags.batch_size)
        ds = ds.repeat(flags.epochs_between_evals)
        return ds

    def get_eval_input():
        return dataset.test(flags.data_dir).batch(
            flags.batch_size).make_one_shot_iterator().get_next()

    train_hooks = [
        tf.train.LoggingTensorHook(tensors =_TENSORS_TO_LOG, every_n_iter=100)
    ]

    for _ in range(flags.train_epochs // flags.epochs_between_evals):
        mnist_classifier.train(input_fn=get_train_input, hooks=train_hooks)
        eval_results = mnist_classifier.evaluate(input_fn=get_eval_input)
        print('Evaluation results:\n\t%s\n' % eval_results)

    if flags.export_dir is not None:
        image = tf.placeholder(tf.float32, [None, 28, 28])
        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
            'image' :  image,
        })
        #mnist_classifier.export_savedmodel(flags.export_dir, input_fn) # Why doesn't this work?

def main(argv):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    default_args.update(**vars(args))
    run_mnist(default_args)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
