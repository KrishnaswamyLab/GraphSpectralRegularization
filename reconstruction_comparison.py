"""
This is meant to understand the reconstruction accuracy on basic autoencoders
with:
    No Regularization
    L1 Regularization
    L2 Regularization
    Graph Spectral Regularization
"""

from tensorflow.keras.layers import Dense, LeakyReLU, Input, Flatten
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.utils import CustomObjectScope
import tensorflow.keras.backend as K
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from joblib import Parallel, delayed
import pandas as pd

from common import models
from common import dataset
from common import util
from common import graph
from common import ising
import common as mycommon


def load_model(path, custom_scope={}):
    with open(path + '.json', 'r') as f:
        json = f.read()
    with CustomObjectScope({'GraphLayer': graph.GraphLayer}):
        m = model_from_json(json)
        m.load_weights(path + '.h5')
        return m


class GraphAE(models.DenseAE):
    """ Convolutional AE with Graph Regularization Penalty included on
    embedding layer.
    """
    def __init__(self, model_dir, data_shape, graph_reg_weight=0.0, **kwargs):
        self.graph_reg_weight = graph_reg_weight
    #    self.graph_reg = GraphRegularizer(weight=self.graph_reg_weight)
        super().__init__(model_dir, data_shape, **kwargs)

    def build_encoder(self):
        x = self.input
        x = Flatten()(x)
        for w in self.layer_widths:
            x = Dense(w)(x)
            x = LeakyReLU(0.2)(x)
        x = Dense(self.latent_dim)(x)
        gl = graph.GraphLayer()
        self.graph_output = gl(x)
        self.embedding = x
        return Model(self.input, outputs=[self.embedding, self.graph_output])

    def get_laplacian(self, y_true, y_pred):
        return K.mean(K.abs(self.embedding))

    def build_autoencoder(self):
        embedding, graph_output = self.encoder(self.input)
        return Model(inputs=[self.input],
                     outputs=[self.decoder(embedding), graph_output])

    def compile(self):
        self.model.compile(optimizer='adam', 
                           loss=['mse', 'mae'],
                           loss_weights=[1, self.graph_reg_weight])


class RegularizedAE(models.DenseAE):
    def __init__(self, model_dir, data_shape, coeffs=None, **kwargs):
        if coeffs is None:
            coeffs = (0., 0.)
        self.coeffs = coeffs
        self.regularizer = keras.regularizers.l1_l2(*coeffs)
        super().__init__(model_dir, data_shape, **kwargs)

    def build_encoder(self):
        x = self.input
        x = Flatten()(x)
        for w in self.layer_widths:
            x = Dense(w)(x)
            x = LeakyReLU(0.2)(x)
        self.embedding = Dense(self.latent_dim, 
                               activity_regularizer=self.regularizer)(x)
        return Model(self.input, self.embedding)


def train(coeffs):
    """ Fits a single model and saves the model in the specific path """
    mycommon.util.set_config()
    l1, l2, gsr = coeffs
    path = 'comparison/%1.0E_%1.0E_%1.0E' % tuple(coeffs)
    d = dataset.Mnist_Dataset()
    epochs = 5
    verbosity = 1
    if gsr == 0:
        m = RegularizedAE(path, d.get_shape(), coeffs=(l1, l2))
        m.fit(d.get_train(), d.get_train(), epochs=epochs, verbose=verbosity)
    else:
        m = GraphAE(path, d.get_shape(), graph_reg_weight=gsr)
        graph_layer = m.encoder.get_layer(name='graph_layer')
        g = ising.create_unnormalized_laplacian(4, 8, 'torus')
        graph_layer.set_weights([np.asarray(g)])
        m.fit(d.get_train(), [d.get_train(), np.zeros((60000, 1))], 
              epochs=epochs, verbose=verbosity)
    m.save()


def eval(coeffs):
    """ Evaluate models using average MSE over train and test sets """
    mycommon.util.set_config()
    l1, l2, gsr = coeffs
    path = 'comparison/%1.0E_%1.0E_%1.0E' % tuple(coeffs)
    d = dataset.Mnist_Dataset()
    losses = []
    for data in [d.get_train(), d.get_test()]:
        if gsr == 0:
            m = load_model(path + '/model')
            m.compile(optimizer='adam', loss='mse')
            loss = m.evaluate(data, data)
        else:
            m = load_model(path + '/model')
            m.compile(optimizer='adam', loss='mse')
            _, loss, _ = m.evaluate(data, [data, np.zeros((len(data), 1))])
        losses.append(loss)
    return losses


def main():
    """ Main, runs all tests in parallel using joblib """
    runs = []
    reg_types = ['l1', 'l2', 'gsr']
    runs.append([0., 0., 0.])
    stand_rep = []
    stand_rep.append(['all', 0])
    for i, reg_type in enumerate(reg_types):
        # if i < 2: continue
        for j in range(-5, 6):
            arr = [0., 0., 0.]
            arr[i] = 10 ** j
            stand_rep.append([reg_type, arr[i]])
            runs.append(arr)
    print('Running %d test' % len(runs))
    print(runs)
    losses = Parallel(n_jobs=15, verbose=10)(delayed(eval)(r) for r in runs)
    np.save('losses.npy', losses)
    losses = np.load('losses.npy')
    dfr = pd.DataFrame(stand_rep, columns=['Type', 'Regularization'])
    dfl = pd.DataFrame(losses, columns=['train_mse', 'test_mse'])
    df = pd.concat([dfr, dfl], axis=1)
    print(df)


if __name__ == '__main__':
    main()

