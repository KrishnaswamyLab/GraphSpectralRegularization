import argparse
import os
import networkx as nx
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pygsp
from scipy.spatial.distance import pdist, squareform
import sklearn
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.layers import Layer, Conv2D, LeakyReLU, Flatten, Dense, GaussianNoise
from tensorflow.keras import Model, initializers
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, BaseLogger
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
import pandas as pd
import seaborn as sns

import common.util as util
import common.dataset as dataset
import common.models as models
from common.graph import GraphLayer
import common.old_dataset as od

class GraphAE(models.DenseAE):
    """ Convolutional AE with Graph Regularization Penalty included on embedding layer.
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
        #self.embedding_layer = Layer(activity_regularizer=self.graph_reg, name='embedding')
        self.graph_output = GraphLayer()(x)
        self.embedding = GaussianNoise(0.5)(x)
        #self.embedding = x
        return Model(self.input, outputs = [self.embedding, self.graph_output])

    def get_laplacian(self, y_true, y_pred):
        return K.mean(K.abs(self.embedding))

    def build_autoencoder(self):
        embedding, graph_output = self.encoder(self.input)
        return Model(inputs=[self.input], outputs=[self.decoder(embedding), graph_output])

    def compile(self):
        self.model.compile(optimizer='adam', 
                           #loss=['mse', custom_mae],
                           loss=['mse', 'mae'],
                           loss_weights=[1,self.graph_reg_weight],
        )#, metrics=[self.get_laplacian])

def create_graph_from_embedding(embedding, name):
    latent_dim, batch_size = embedding.shape
    if name =='gaussian':
        # Compute a gaussian kernel over the node activations
        node_distances = squareform(pdist(embedding, 'sqeuclidean'))
        s = 1
        K = np.exp(-node_distances / s**2)
        K[K < 0.1] = 0
        A = K * (np.ones((latent_dim, latent_dim)) - np.identity(latent_dim))
        return A
    elif name == 'knn':
        #nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=5)
        A = sklearn.neighbors.kneighbors_graph(embedding, n_neighbors=5).toarray()
        A = (A + np.transpose(A)) / 2 # Symmetrize knn graph
        return A
    elif name == 'adaptive':
        # Find distance of k-th nearest neighbor and set as bandwidth
        neigh = sklearn.neighbors.NearestNeighbors(n_neighbors=3)
        neigh.fit(embedding)
        dist, _ = neigh.kneighbors(embedding, return_distance=True)
        kdist = dist[:,-1]
        # Apply gaussian kernel with adaptive bandwidth
        node_distances = squareform(pdist(embedding, 'sqeuclidean'))
        K = np.exp(-node_distances / kdist**2)
        A = K * (np.ones((latent_dim, latent_dim)) - np.identity(latent_dim))
        A = (A + np.transpose(A)) / 2 # Symmetrize knn graph
        return A
    else:
        raise RuntimeError('Unknown graph name %s' % name)

def plot(graphs, path):
    fig = plt.figure(figsize=(40,20))
    gs = GridSpec(5,10)
    print(len(graphs))
    for i,a in enumerate(graphs):
        latent_dim = len(a)
        ax = plt.subplot(gs[i % 5, i // 5])
        #g = nx.to_networkx_graph(a)
        #nx.draw(g, ax=ax, node_size=50)
        #plt.draw()
        G = pygsp.graphs.Graph(a)
        G.set_coordinates()
        G.plot(ax = ax)
        G.plot_signal(np.arange(latent_dim), colorbar = False, ax=ax)
    plt.savefig(path)
    plt.close()

def plot_last(graphs, path):
    g = pygsp.graphs.Graph(graphs[-1])
    coords_list = []
    coords = np.load('coords_list.npy')[-1]
    np.save(path + '/coords.npy', coords)
    g.set_coordinates(coords)
    g.plot()
    plt.show()
    return

    fig = plt.figure(figsize=(20,10))
    gs = GridSpec(5,10)
    for i in range(50):
        ax = plt.subplot(gs[i // 10, i % 10])
        g.set_coordinates()
        coords_list.append(g.coords)
        g.plot(ax=ax)
        ax.set_title('Time %d' % i)
        ax.set_xticks([])
        ax.set_yticks([])
    coords_list = np.array(coords_list)
    print(coords_list.shape)
    #np.save('coords_list.npy', coords_list)
    plt.show()


def plot_graphs(graphs, model_dir, path):
    fig = plt.figure(figsize=(20,5))
    gs = GridSpec(1,4)
    print('Num Steps: %d', len(graphs))
    for i,a in enumerate(graphs):
        latent_dim = len(a)
        ax = plt.subplot(gs[0, i])
        #g = nx.to_networkx_graph(a)
        #nx.draw(g, ax=ax, node_size=50)
        #plt.draw()
        G = pygsp.graphs.Graph(a)
        G.set_coordinates()
    #    if i == len(graphs) - 1:
    #        G.set_coordinates(np.load(model_dir + '/coords.npy'))
        G.plot(ax = ax)
        G.plot_signal(np.arange(latent_dim), colorbar = False, ax=ax)
        ax.set_title('Time %d' % i)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(path)
    plt.close()

def load_model(path, custom_scope={}):
    with open(path + '.json', 'r') as f:
        json = f.read()
    with CustomObjectScope({'GraphLayer': GraphLayer}):
        m = model_from_json(json)
        m.load_weights(path + '.h5')
        return m

def load_encoder(path):
    return load_model(path + '/encoder')

def load_ae(path):
    model = load_model(path + '/model')
    encoder = load_model(path + '/encoder')
    decoder = load_model(path + '/decoder')
    return model, encoder, decoder

def plot2(model_path, data, graphs, path):
    G = pygsp.graphs.Graph(graphs[-1])
    G.set_coordinates()
    x_train = data.get_train()
    x_labels = data.get_train_labels()
    d = x_train[list(range(1,21,2))]
    encoder = load_encoder(model_path)
    enc, graph_loss = encoder.predict(x_train)
    enc2 = []
    avg_enc = np.mean(enc, axis=0)
    for i in range(10):
        if i > np.max(x_labels): break
        enc2.append(np.mean(enc[x_labels==i], axis=0) - avg_enc)
    enc = enc2

    fig = plt.figure(figsize=(20,10))
    gs = GridSpec(2,5)
    for i, e in enumerate(enc):
        ax = plt.subplot(gs[i % 2, i // 2])
        ax.set_xticks([])
        ax.set_yticks([])
        G.plot_signal(enc[i], plot_name ='%d' % i, ax=ax, colorbar=False)
    plt.savefig(path)
    plt.close()
    #plt.show()

def plot_branch(model_path, data, graphs, path=None):
    G = pygsp.graphs.Graph(graphs[-1])
    print(G.is_connected())
    G_sub = G.extract_components()
    print(G_sub)
    #print(G_sub.info)
    G.set_coordinates()
    x_train = data.get_train()
    branch_labels = data.get_train_labels()
    trajectory_labels = data.trajectory_labels
    labels = np.vstack([branch_labels, trajectory_labels])
    masks = []

    # Partition Points
    for i in range(1,4):
        t = trajectory_labels[branch_labels == i]
        cut = np.percentile(t, 0)
        branch_end_mask = (branch_labels == i) & (trajectory_labels > cut)
        masks.append(branch_end_mask)
        print(np.sum(branch_end_mask))
    encoder = load_encoder(model_path)
    enc, graph_loss = encoder.predict(x_train)
    enc2 = []

    # Normalize Step
    import functools
    all_points = functools.reduce(lambda x,y: x | y, masks)
    avg_enc = np.mean(all_points, axis=0)
    for i in range(3):
        #enc2.append(np.mean(enc[masks[i]], axis=0) - avg_enc)
        enc2.append(np.mean(enc[masks[i]], axis=0))
    enc = enc2
    #enc.append(np.zeros_like(enc[-1]))
    print(enc)

    # Plot Step
    fig = plt.figure(figsize=(20,5))
    gs = GridSpec(1,3)
    for i, e in enumerate(enc):
        ax = plt.subplot(gs[i // 5, i % 5])
        ax.set_xticks([])
        ax.set_yticks([])
        G.plot_signal(enc[i], plot_name ='%d' % (i+1), ax=ax, colorbar=False)
    if path is None:
        plt.show()
        return
    else:
        plt.savefig(path)
        plt.close()


def plot_ends(model_path, data, graphs, path=None):
    G = pygsp.graphs.Graph(graphs[-1])
    G.set_coordinates(np.load(model_path + '/coords.npy'))
    x_train = data.get_train()
    branch_labels = data.get_train_labels()
    
    trajectory_labels = data.trajectory_labels
    bl, tl = (branch_labels, trajectory_labels)
    labels = np.vstack([branch_labels, trajectory_labels])
    masks = []
    # Beginning
    masks.append((bl == 1) & (tl < np.percentile(tl[bl == 1], 5)))
    masks.append((bl == 1) & (tl < np.percentile(tl[bl == 1], 52.5)) & (tl > np.percentile(tl[bl == 1], 47.5)))
    for i in [1,2,3]:
        masks.append((bl == i) & (tl > np.percentile(tl[bl == i], 95)))
    encoder = load_encoder(model_path)
    enc, graph_loss = encoder.predict(x_train)
    enc2 = []
    import functools
    all_points = functools.reduce(lambda x,y: x | y, masks)
    avg_enc = np.mean(all_points, axis=0)
    for i in range(5):
        #enc2.append(np.mean(enc[masks[i]], axis=0) - avg_enc)
        enc2.append(np.mean(enc[masks[i]], axis=0))
    enc = enc2
    #enc.append(np.zeros_like(enc[-1]))
    print(enc)
    fig = plt.figure(figsize=(20,5))
    gs = GridSpec(1,5)
    for i, e in enumerate(enc):
        ax = plt.subplot(gs[i // 5, i % 5])
        ax.set_xticks([])
        ax.set_yticks([])
        G.plot_signal(enc[i], plot_name ='%d' % i, ax=ax, colorbar=False)
    if path is None:
        plt.show()
        return
    else:
        plt.savefig(path)
        plt.close()

def plot3(model_path, data, graphs, path=None):
    G = pygsp.graphs.Graph(graphs[-1])
    G.set_coordinates()
    x_train = data.get_train()
    branch_labels = data.get_train_labels()
    trajectory_labels = data.trajectory_labels
    labels = np.vstack([branch_labels, trajectory_labels])
    masks = []
    for i in range(5):
        t = trajectory_labels
        low_cut = np.percentile(t, 20 * i + 0.0001)
        high_cut = np.percentile(t, 20 * (i+1) - 0.0001)
        branch_end_mask = ((t < high_cut) & (t > low_cut))
        masks.append(branch_end_mask)
        print(np.sum(branch_end_mask))
        #plt.hist(t, bins=100)
        #plt.show()
    encoder = load_encoder(model_path)
    enc, graph_loss = encoder.predict(x_train)
    enc2 = []
    import functools
    all_points = functools.reduce(lambda x,y: x | y, masks)
    avg_enc = np.mean(all_points, axis=0)
    for i in range(5):
        #enc2.append(np.mean(enc[masks[i]], axis=0) - avg_enc)
        enc2.append(np.mean(enc[masks[i]], axis=0))
    enc = enc2
    #enc.append(np.zeros_like(enc[-1]))
    print(enc)
    fig = plt.figure(figsize=(20,5))
    gs = GridSpec(1,5)
    for i, e in enumerate(enc):
        ax = plt.subplot(gs[i // 5, i % 5])
        ax.set_xticks([])
        ax.set_yticks([])
        G.plot_signal(enc[i], plot_name ='%d' % i, ax=ax, colorbar=False)
    if path is None:
        plt.show()
        return
    else:
        plt.savefig(path)
        plt.close()

def plot_wishbone_heatmap(model_path, data, graphs, path=None):
    df = data.full_data
    x_train = data.get_train()
    encoder = load_encoder(model_path)
    embedding, _ = encoder.predict(x_train)
    #print(embedding)
    outs = pd.concat([pd.DataFrame(embedding, index = df.index), df.Trajectory, df.Branch], axis=1)
    d = []
    for b,g in outs.groupby('Branch'):
        d.append(g.sample(333))
    outs = pd.concat(d)
    outs = outs.sort_values(by=['Branch', 'Trajectory'])
    #print(outs)
    lut = dict(zip(outs.Branch.unique(), 'rbg'))
    row_colors = outs.Branch.map(lut)
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    cmap = cm.Reds
    norm = Normalize(vmin=outs.Trajectory.min(), vmax=outs.Trajectory.max())
    row_colors2 = outs.Trajectory.apply(lambda x: cmap(norm(x)))
    g = sns.clustermap(outs.drop(columns=['Trajectory', 'Branch']), 
            row_colors = pd.concat([row_colors, row_colors2], axis=1),
            row_cluster=False,
            col_cluster=True,
            yticklabels=False,
            #cmap = "RdBu_r",
            cmap = cm.Reds,
    )
    g.ax_heatmap.set_title('Wishbone Heatmap')
    plt.show()

def plot_mutual_information(model_path, data, graphs, path=None):
    encoder = load_encoder(model_path)
    layer_outs, _ = encoder.predict(data.get_train())
    df = data.data
    fdf = data.full_data
    all_data = pd.concat([pd.DataFrame(layer_outs), df, fdf.Branch, fdf.Trajectory], axis=1)
    print(layer_outs.shape)
    d = []
    def plot_heatmap(sampled, title):
        embedding_df = sampled.iloc[:,:100]
        gene_df      = sampled.iloc[:,100:-2]
        corrs = []
        for column in gene_df:
            corrs.append(embedding_df.corrwith(gene_df[column]))
        corrs = pd.DataFrame(corrs)
        print(corrs.shape)
        corrs.index = gene_df.columns
        corrs.columns = embedding_df.columns
        print(gene_df.shape, embedding_df.shape)
        #corrs = embedding_df.corrwith(gene_df)
        print(corrs.shape)
        cg = sns.clustermap(corrs,
                #col_cluster = False,
                #row_cluster=False,
                cmap = "RdBu_r",
                center=0,
        )
        cg.ax_row_dendrogram.set_visible(False)
        plt.savefig(title)
        plt.close()
    for b,g in all_data.groupby('Branch'):
        tmp = g.sample(8000)
        print(tmp.shape)
        plot_heatmap(pd.DataFrame(tmp), model_path + '/Branch_%d_heatmap.png' % b)
        d.append(tmp)
    plot_heatmap(pd.concat(d), model_path + '/All_heatmap.png')

def plot_mutual_information_same_order(model_path, data, graphs, path=None):
    encoder = load_encoder(model_path)
    layer_outs, _ = encoder.predict(data.get_train())
    df = data.data
    fdf = data.full_data
    all_data = pd.concat([pd.DataFrame(layer_outs), df, fdf.Branch, fdf.Trajectory], axis=1)
    def plot_heatmap(corrs, row_order, col_order, ax, title):
        corrs = corrs[col_order]
        corrs = corrs.reindex(row_order)
        cg = sns.heatmap(corrs,
                cmap = "RdBu_r",
                center=0,
                ax=ax,
                cbar=False
        )
    def get_corrs(df):
        embedding_df = df.iloc[:,:100]
        gene_df      = df.iloc[:,100:-2]
        corrs = []
        for column in gene_df:
            corrs.append(embedding_df.corrwith(gene_df[column]))
        corrs = pd.DataFrame(corrs)
        corrs.index = gene_df.columns
        corrs.columns = embedding_df.columns
        return corrs
    def get_order(df):
        cg = sns.clustermap(df,
                #col_cluster = False,
                #row_cluster=False,
                cmap = "RdBu_r",
                center=0,
        )
        plt.close()
        return cg.dendrogram_row.reordered_ind, cg.dendrogram_col.reordered_ind
    maps = []
    for b,g in all_data.groupby('Branch'):
        maps.append(get_corrs(g.sample(8000)))
    _, node_order = get_order(pd.concat(maps))
    gene_order, _ = get_order(pd.concat(maps, axis=1))
    #all_maps = pd.concat(maps).assign(Branch=np.repeat(np.arange(1,4), 20))
    #g = sns.FacetGrid(all_maps, col='Branch')
    #g.map(sns.heatmap)
    gs = GridSpec(1,3)
    fig = plt.figure(figsize=(20, 5))
    for i,m in enumerate(maps): 
        ax = plt.subplot(gs[i // 5, i % 5])
        ax.set_title('Branch %d' % (i+1))
        ax.set_xticks([])
        ax.set_yticks([])
        plot_heatmap(m, m.index[gene_order], node_order, ax, model_path + '/Branch_heatmap.png')
    #plt.show()
    plt.savefig(model_path + '/Branch_heatmap.png')
    plt.close()
    #    d.append(tmp)
    #plot_heatmap(pd.concat(d), model_path + '/All_heatmap.png')


def plot_output(model_path):
    m, encoder, decoder = load_ae(model_path)
    x_train = data.get_train()
    d = x_train[list(range(1,21,2))]
    out, _ = m.predict(d)

def train(m, data):
    sess = tf.Session()
    x_train = data.get_train()
    batch_size = 32
    sess.run(tf.global_variables_initializer())
    with tf.device("/device:GPU:0"):
        zeros = tf.zeros((batch_size,1))
    laplacian = np.identity(m.latent_dim)
    graph_layer = m.encoder.get_layer(name='graph_layer')
    graphs = []
    for i in range(5000):
        points = np.random.randint(len(x_train), size=batch_size)
        data = x_train[points]
        loss = m.model.train_on_batch(data, [data, zeros])
    for i in range(5000):
        points = np.random.randint(len(x_train), size=batch_size)
        data = x_train[points]
        loss = m.model.train_on_batch(data, [data, zeros])
        if (i) % 100 == 0:
            batch_embedding, batch_energy = m.encoder.predict(data, steps=1)
            A = create_graph_from_embedding(np.transpose(batch_embedding), name='knn')
            G = pygsp.graphs.Graph(A)
            if sum(G.e < 0.001) > 1: # Break if graph is disconnected
                for i in range(200):
                    points = np.random.randint(len(x_train), size=batch_size)
                    data = x_train[points]
                    loss = m.model.train_on_batch(data, [data, zeros])
                break
            G.compute_laplacian(lap_type='normalized')
            laplacian = G.L.todense()
            graphs.append(A)
            graph_layer.set_weights([laplacian])
    m.save()
    return np.array(graphs)

def run_all(model_dir_suffix, graph_reg_weight=1):
    path = 'model_dir/%s/%0.2e/' % (model_dir_suffix, graph_reg_weight)
    os.makedirs(path, exist_ok=True)
    #data = dataset.Dataset.factory('mnist')
    data = dataset.Wishbone_Dataset(even_branches=True)
    #data = dataset.Hub_Dataset(modules)
    model = GraphAE(path, data.get_shape(), latent_dim=100, graph_reg_weight=graph_reg_weight)
    graphs = util.npdo(lambda: train(model, data), path + '/graphs.npy')
    #plot_last(graphs, model.model_dir)
    plot_graphs(graphs, model.model_dir, model.model_dir + '/graph.png')
    #plot_output(path)
    #plot3(path, data, graphs, model.model_dir + '/graph3.png')
    #plot_branch(path, data, graphs, model.model_dir + '/branch.png')
    #plot_ends(path, data, graphs, model.model_dir + '/branch_points.png')
    #plot_mutual_information(path, data, graphs)#, model.model_dir + '/branch.png')
    #plot_mutual_information_same_order(path, data, graphs)#, model.model_dir + '/branch.png')
    plot_wishbone_heatmap(path, data, graphs)#, model.model_dir + '/branch.png')
    #plot3(path, data, graphs)
    #plot2(path, data, graphs, model.model_dir + '/graph2.png')

if __name__ == '__main__':
    util.set_config()
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir_suffix', default='wishbone_even_10/')
    parser.add_argument('--ising_weight', '-i', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    run_all(args.model_dir_suffix, args.ising_weight)


