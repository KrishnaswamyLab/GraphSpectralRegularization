import mnist_dataset as md
import fcsparser
import tensorflow as tf
import numpy as np
import util
import os
import zipfile
import tempfile
import posttrain as pt
from six.moves import urllib
import seaborn as sns
from sklearn import datasets, decomposition, manifold
from scipy.io import loadmat
import scipy
import matplotlib.pyplot as plt
import phate
from mpl_toolkits.mplot3d import Axes3D
from functools import reduce, partial
import pandas as pd
from pprint import pprint

class Dataset(object):
    def __init__(self, flags):
        self.flags = flags
    def factory(type):
        if type == 'mnist': return Mnist_Dataset
        if type == 'wishbone': return Wishbone_Dataset
        if type == 'teapot': return Teapot_Dataset
        if type == 'swiss_roll': return partial(Dataset_3d, name = type)
        if type == 's_curve': return partial(Dataset_3d, name = type)
        if type == 's_curve_sanity': return partial(Dataset_3d,name = 's_curve', n=500)
        if type == 'linear': return Linear_Dataset
        if type == 'hub': return Hub_Spoke
        if type == 'hub_perturb': return partial(Hub_Spoke, perturb=True)
        if type == 'hub_no_noise': return partial(Hub_Spoke, noise = 0.0)
        if type == 'cifar10': return partial(Keras_dataset, name='cifar10')
        if type == 'cifar100': return partial(Keras_dataset, name='cifar100')
        if type == 'keras_mnist': return partial(Keras_dataset, name='mnist')
        if type == 'small_mnist': return partial(Keras_dataset, name='mnist', n=1000)
        if type == 'gene_synth': return Gene_Interaction
        if type == 'blend': return Blend_Dataset
        if type == 'noonan': return Noonan_Dataset
        if type == 'mnist_var': return Mnist_Variation
        if type == 'affnist': return Affnist
        if type == 'hidden_linear': return Hidden_Linear
        if type == 'humps': return partial(Hump_Dataset, dims=20)
        raise AssertionError('Bad Dataset Creation: %s' % type)

    factory = staticmethod(factory)

    def get_train_input(self):
        raise NotImplementedError
    def get_predict_input(self):
        raise NotImplementedError
    def get_eval_input(self):
        raise NotImplementedError
    def get_predict_input(self):
        raise NotImplementedError
    def predict(self, preds):
        raise NotImplementedError
    def input_width(self):
        raise NotImplementedError
    def get_dataset_default_flags(self):
        """ Dictionary containing default flag overrides """
        return {}
    def get_network(self):
        return None

def np_to_dataset(npx, npy, flatten=False):
    def decode_image(x):
        x = tf.cast(x, tf.float32) / 255.0
        if flatten:
            return tf.reshape(x, [-1])
        return x
    def decode_label(x):
        return tf.cast(x, tf.int32)
    dsx = tf.data.Dataset.from_tensor_slices(npx).map(decode_image, num_parallel_calls=8)
    dsy = tf.data.Dataset.from_tensor_slices(npy).map(decode_label, num_parallel_calls=8)
    return tf.data.Dataset.zip((dsx, dsy))

class Keras_dataset(Dataset):
    def __init__(self, flags, name='cifar10', n = None):
        """
        Args:
            name: dataset name
            n: number of training examples
        """
        super().__init__(flags)
#        x_train, y_train, x_test, y_test = [np.load('/tmp/%s%d.npy' % (name, i)) for i in range(4)]
#        x_train, y_train, x_test, y_test = util.npdo_list(lambda: keras_load_data(name), ['/tmp/%s%d.npy' % (name,i) for i in range(4)])
        self.flatten = True #not flags.use_classifier

        if name == 'cifar10':
            data = tf.keras.datasets.cifar10
        if name == 'cifar100':
            data = tf.keras.datasets.cifar100
        if name == 'mnist':
            data = tf.keras.datasets.mnist

        (self.x_train, self.y_train), (self.x_test, self.y_test) = data.load_data()
        if n is not None:
            self.x_train = self.x_train[:n]
            self.y_train = self.y_train[:n]

    def get_train_input(self):
        train = np_to_dataset(self.x_train, self.y_train, flatten = self.flatten)
        return train.cache().shuffle(buffer_size=50000).batch(
                self.flags.batch_size).repeat(self.flags.epochs_between_evals)
    def get_predict_input(self):
        test = np_to_dataset(self.x_test, self.y_test, flatten=self.flatten)
        return test.batch(self.flags.batch_size)
    def get_eval_input(self):
        test = np_to_dataset(self.x_test, self.y_test, flatten=self.flatten)
        return test.batch(self.flags.batch_size)
    def input_width(self):
        if not self.flatten:
            return self.x_train.shape[1:]
        return reduce(lambda x,y: x*y, self.x_train.shape[1:])
    def input_tuple(self):
        return self.x_train.shape[1:]
    def input_depth(self):
        if len(self.input_tuple()) > 2:
            return self.input_tuple()[2]
        return 1
    def predict(self, preds):
        flags = self.flags

        if flags.predict == 'classifier':
            training_labels = self.y_test
            plots = [
                lambda savedir: util.plot_mnist_heatmap(preds, training_labels, savedir=savedir),
                lambda savedir: util.plot_average_layer(preds, flags.height, flags.width, 
                                                        layer_name='ising_layer', savedir=savedir),
#                lambda savedir: util.plot_embedding(preds, training_labels, savedir=savedir),
                lambda savedir: util.plot_histogram(preds, savedir=savedir),
                lambda savedir: util.plot_class_average_layer(preds, flags.height, flags.width,
                                                              layer_name='ising_layer',
                                                              labels=training_labels,
                                                              savedir=savedir),
            #    lambda savedir: util.plot_output(preds, ishape=self.input_tuple()[:2], num_channels = self.input_depth(), savedir=savedir),
            ]
            for p in plots: p(flags.model_dir)

        if flags.predict == 'all':
            training_labels = self.y_test
            plots = [
                lambda savedir: util.plot_mnist_heatmap(preds, training_labels, savedir=savedir),
                lambda savedir: util.plot_average_layer(preds, flags.height, flags.width, 
                                                        layer_name='ising_layer', savedir=savedir),
#                lambda savedir: util.plot_embedding(preds, training_labels, savedir=savedir),
                lambda savedir: util.plot_histogram(preds, savedir=savedir),
                lambda savedir: util.plot_class_average_layer(preds, flags.height, flags.width,
                                                              layer_name='ising_layer',
                                                              labels=training_labels,
                                                              savedir=savedir),
                lambda savedir: util.plot_output(preds, ishape=self.input_tuple()[:2], num_channels = self.input_depth(), savedir=savedir),
            ]
            for p in plots: p(flags.model_dir)
            #self.plot_nodes_in_range(preds)
            #self.plot_nodes_in_range(preds, title='activation_embedding_same_scale', same_scale=True)
            util.plot_class_fourier(preds, training_labels, flags.height, flags.width, savedir=flags.model_dir)
        

class Mnist_Dataset(Dataset):
    def __init__(self, flags):
        super().__init__(flags)
        self.data, self.labels = util.npdo_list(self.input_to_numpy, [self.flags.data_dir + '/' +  name for name in ['input.npy', 'labels.npy']])
    def get_train_input(self):
        return md.train(self.flags.data_dir).cache().shuffle(buffer_size=60000).batch(
            self.flags.batch_size).repeat(self.flags.epochs_between_evals)
    def get_eval_input(self):
        return md.test(self.flags.data_dir).batch(self.flags.batch_size).make_one_shot_iterator().get_next()
    def _get_all_train_input(self):
        return md.train(self.flags.data_dir).batch(60000).make_one_shot_iterator().get_next()
    def get_predict_input(self):
        return md.train(self.flags.data_dir).batch(
            self.flags.batch_size).make_one_shot_iterator().get_next()
    def input_to_numpy(self):
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.3)))
        #self.test = sess.run(self.get_eval_input())
        self.train = sess.run(self._get_all_train_input())
        #print(self.train[0].shape, self.test[0].shape)
        #return np.concatenate((self.train[0], self.test[0])), np.concatenate((self.train[1], self.test[1]))
        return self.train

    def predict(self, preds):
        flags = self.flags
        training_labels = self.labels[:60000]
        if flags.predict == 'embed_activations':
            self.plot_nodes_in_range(preds)
            self.plot_nodes_in_range(preds, title='activation_embedding_same_scale', same_scale=True)
        if flags.predict == 'fourier':
            util.plot_class_fourier(preds, training_labels, flags.height, flags.width, savedir=flags.model_dir)
        if flags.predict == 'norm_class':
            util.plot_class_average_layer(preds, flags.height, flags.width,
                                          layer_name='ising_layer',
                                          labels=training_labels,
                                          normalize=True,
                                          savedir=flags.model_dir)
        if flags.predict == 'segment':
            training_labels = self.labels
            util.segment_class_average_layer(preds, flags.height, flags.width, layer_name='ising_layer', labels=training_labels, savedir=flags.model_dir)
            #util.plot_layer_examples(preds, flags.height, flags.width, layer_name='ising_layer', labels=training_labels, savedir=flags.model_dir)

        if flags.predict == 'classifier':
            training_labels = self.labels
            plots = [
                lambda savedir: util.plot_mnist_heatmap(preds, training_labels, savedir=savedir),
                lambda savedir: util.plot_average_layer(preds, flags.height, flags.width, 
                                                        layer_name='ising_layer', savedir=savedir),
#                lambda savedir: util.plot_embedding(preds, training_labels, savedir=savedir),
                lambda savedir: util.plot_histogram(preds, savedir=savedir),
                lambda savedir: util.plot_class_average_layer(preds, flags.height, flags.width,
                                                              layer_name='ising_layer',
                                                              labels=training_labels,
                                                              savedir=savedir),
                lambda savedir: util.plot_layer_examples(preds, flags.height, flags.width,
                                                         layer_name='ising_layer',
                                                         labels=training_labels,
                                                         savedir=savedir),
            #    lambda savedir: util.plot_output(preds, ishape=self.input_tuple()[:2], num_channels = self.input_depth(), savedir=savedir),
            ]
            for p in plots: p(flags.model_dir)
            #self.plot_nodes_in_range(preds)
            #self.plot_nodes_in_range(preds, title='activation_embedding_same_scale', same_scale=True)
            util.plot_class_fourier(preds, training_labels, flags.height, flags.width, savedir=flags.model_dir)
        if flags.predict == 'misses':
            util.plot_misses(preds, flags.height, flags.width,
                             layer_name='ising_layer',
                             labels=self.labels,
                             is_classifier=flags.use_classifier,
                             threshold=True,
                             savedir=flags.model_dir),
        if flags.predict == 'all':
            plots = [
                lambda savedir: util.plot_mnist_heatmap(preds, training_labels, savedir=savedir),
                lambda savedir: util.plot_average_layer(preds, flags.height, flags.width, 
                                                        layer_name='ising_layer', savedir=savedir),
#                lambda savedir: util.plot_embedding(preds, training_labels, savedir=savedir),
                lambda savedir: util.plot_histogram(preds, savedir=savedir),
                lambda savedir: util.plot_class_average_layer(preds, flags.height, flags.width,
                                                              layer_name='ising_layer',
                                                              labels=training_labels,
                                                              savedir=savedir),
                lambda savedir: util.plot_class_average_layer(preds, flags.height, flags.width,
                                                              layer_name='ising_layer',
                                                              labels=training_labels,
                                                              normalize=True,
                                                              savedir=savedir),
                lambda savedir: util.plot_layer_examples(preds, flags.height, flags.width,
                                                         layer_name='ising_layer',
                                                         labels=training_labels,
                                                         is_classifier=flags.use_classifier,
                                                         savedir=savedir),
                lambda savedir: util.plot_layer_examples(preds, flags.height, flags.width,
                                                         layer_name='ising_layer',
                                                         labels=training_labels,
                                                         is_classifier=flags.use_classifier,
                                                         threshold=True,
                                                         savedir=savedir),
            ]
            for p in plots: p(flags.model_dir)
            if not flags.use_classifier:
                util.plot_output(preds, savedir=flags.model_dir)
            #self.plot_nodes_in_range(preds)
            #self.plot_nodes_in_range(preds, title='activation_embedding_same_scale', same_scale=True)
            #util.plot_class_fourier(preds, training_labels, flags.height, flags.width, savedir=flags.model_dir)
    def input_width(self):
        return 28 * 28
    def get_embedding(self):
        def do_phate():
            import phate
            phate_op = phate.PHATE(k=5, a=None, t=52, n_jobs=-2)
            #phate_op = phate.PHATE(k=5, a=None, t=105, n_jobs=-2)
            return phate_op.fit_transform(self.data)
        return util.npdo(do_phate, self.flags.data_dir + '/embedding.npy')

    def plot_embed(self, labels = None, title=None, fig = None, ax = None, set_legend=True, vmin=None, vmax=None):
        if labels is None: labels = self.labels[:60000]
        sc = util.scatter_plot2d(self.get_embedding()[:60000], labels, 'mnist_embedding' if title is None else title, set_legend=set_legend, fig=fig, ax = ax, vmin=vmin, vmax=vmax)
        if fig is None: util.save_show(self.flags.model_dir, 'input_embed' if title is None else title)
        return sc

    def plot_nodes_in_range(self, preds, separate = False, title = 'activation_embedding', same_scale=False):
        h,w = (self.flags.height, self.flags.width)
        if separate:
            for i in range(h*w):
                labels = preds['ising_layer'][:,i]
                self.plot_embed(labels = labels,
                                title  = 'Node %d embedding' % i)
        else:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(h, w, figsize=(5*w,5*h))
            fig.suptitle('ising layer node activations')
            minl = 1000000
            maxl = -1000000
            for i in range(h*w):
                labels = preds['ising_layer'][:,i]
                minl = min(np.percentile(labels, 1), minl)
                maxl = max(np.percentile(labels, 99), maxl)
            print(minl, maxl)
            for i in range(h*w):
                ax = axes[i // w, i % w] if h > 1 and w > 1 else axes[i]
                labels = preds['ising_layer'][:,i]
                #std_labels = (labels + np.min(labels)) / np.max(labels)
                if same_scale:
                    sc = self.plot_embed(labels = labels,
                                         title  = 'Node %d embedding' % i,
                                         fig = fig, ax = ax, set_legend=False,
                                         vmin = minl, vmax=maxl
                                         )
                else:
                    sc = self.plot_embed(labels = labels,
                                         title  = 'Node %d embedding' % i,
                                         fig = fig, ax = ax, set_legend=False,
                                         #vmin = minl, vmax=maxl
                                         )
            if same_scale:
                fig.colorbar(sc, ticks=np.linspace(minl, maxl, 5), ax=axes.ravel().tolist())
            util.save_show(self.flags.model_dir, title)

class Wishbone_Dataset(Dataset):
    def __init__(self, flags):
        super().__init__(flags)
        self.data_dir = '/home/atong/data/wishbone_thymus_panel1_rep1.fcs'
        _, self.full_data = fcsparser.parse(self.data_dir)
        self.data = self.filter_data(self.full_data)
        self.test_split = (self.data.shape[0] * 4) // 5
        self.train_data = self.data.iloc[self.test_split:]
        self.test_data  = self.data.iloc[:self.test_split]
    def filter_data(self, data):
        # Filer for important cells
        data = data[np.concatenate([data.columns[:13], data.columns[14:21]])]
        data = np.arcsinh(data / 5)
        return data
    def get_train_input(self):
        t = tf.data.Dataset.from_tensor_slices((self.train_data))
        return t.cache().shuffle(buffer_size=100000).batch(self.flags.batch_size).repeat(self.flags.epochs_between_evals)
    def get_eval_input(self):
        t = tf.data.Dataset.from_tensor_slices((self.test_data))
        return t.cache().batch(self.flags.batch_size).make_one_shot_iterator().get_next()
    def get_all_train_input(self):
        t = tf.data.Dataset.from_tensor_slices((self.train_data))
        return t.make_one_shot_iterator().get_next()
    def get_predict_input(self):
        t = tf.data.Dataset.from_tensor_slices((self.data))
        return t.batch(self.flags.batch_size).make_one_shot_iterator().get_next()
    def predict(self, preds):
        flags = self.flags
        if flags.predict == 'heatmap':
            util.plot_wishbone_heatmap(preds, self, savedir=flags.model_dir)
        elif flags.predict == 'scatter':
            util.plot_wishbone_embedding_scatter_plot(preds, self, savedir=flags.model_dir)
        elif flags.predict == 'pca':
            util.plot_wishbone_output_pca(preds, self, savedir=flags.model_dir)
        elif flags.predict == 'cls':
            util.wishbone_plot_pop_split(preds, self, flags.height, flags.width, savedir=flags.model_dir)
        elif flags.predict == 'cd8':
            util.plot_cd8_cd4(preds, savedir=flags.model_dir)
        elif flags.predict == 'corr':
            pt.corrolate_layer(preds, self, savedir=flags.model_dir)
        elif flags.predict == 'mi':
            self.mutual_information(preds, savedir=flags.model_dir)
        elif flags.predict == 'all':
            util.plot_wishbone_heatmap(preds, self, savedir=flags.model_dir)
            util.plot_wishbone_heatmap(preds, self, layer_name='embedding', savedir=flags.model_dir)
            util.plot_wishbone_output_pca(preds, self, savedir=flags.model_dir)
            util.plot_cd8_cd4(preds, self, savedir=flags.model_dir)
            util.wishbone_plot_pop_split(preds, self, flags.height, flags.width, savedir=flags.model_dir)
            util.plot_wishbone_heatmap2(preds, self, savedir=flags.model_dir)
        else:
            raise RuntimeError('Unknown predict parameter %s' % flags.predict)
    def input_width(self):
        return 20
    def get_dataset_default_flags(self):
        return  {   'batch_size': 500,
                    'train_epochs': 200,
                    'epochs_between_evals': 20,
                }
    def mutual_information(self, preds, savedir=None):
        layer_outs = preds['ising_layer']
        df = self.data
        fdf = self.full_data
        
        all_data = pd.concat([pd.DataFrame(layer_outs), df, fdf.Branch, fdf.Trajectory], axis=1)
        print(layer_outs.shape)
        d = []
        def plot_heatmap(sampled, title):
            embedding_df = sampled.iloc[:,:20]
            gene_df      = sampled.iloc[:,20:-2]
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
                    col_cluster = False,
                    cmap = "RdBu_r",
                    center=0,
            )
            cg.ax_row_dendrogram.set_visible(False)
            plt.savefig(title)
        for b,g in all_data.groupby('Branch'):
            tmp = g.sample(8000)
            plot_heatmap(pd.DataFrame(tmp), 'Branch_%d_heatmap.png' % b)
            
            d.append(tmp)
        plot_heatmap(pd.concat(d), 'All_heatmap.png')



    def create_plot(self):
        phate_op = phate.PHATE()
        embed = phate_op.fit_transform(self.data)
        #print(self.full_data.Branch.loc[:1000])
        phate.plot.scatter2d(embed,c = np.array(self.full_data.Branch))
        plt.show()
    def plot_phate(self):
        df = np.array(pd.read_pickle('wishbone_phate.pickle'))
        lut = dict(zip([1,2,3], 'rbg'))
        colors = self.full_data.Branch.map(lut)
        for i,c in enumerate('rbg'):
            dff = df[self.full_data.Branch == i+1]
            plt.scatter(dff[:,0], dff[:,1], c=c, s=1, alpha=0.5)
        #plt.legend([1,2,3], ('Branch 0', 'Branch 1', 'Branch 2'))
        plt.xlabel('PHATE 1')
        plt.ylabel('PHATE 2')
        plt.xticks([])
        plt.yticks([])
        plt.show()


def PCA(data, n_components=2, **kwargs):
    np.random.seed(42)
    return decomposition.PCA(n_components=n_components).fit_transform(data)
def _process_image(img, label):
    img = tf.image.resize_images(img, [28,28])
    img = tf.reshape(img, [-1])
    return img,label

class Teapot_Dataset(Dataset):
    """ Access to the rotating teapot dataset """
    # TODO should we be doing PCA on this? Doubtful
    def __init__(self, flags, pca=False):
        super().__init__(flags)
        self.n_components = 100
        self.pca = pca
        self.data = self.load_teapot()
    def load_teapot(self):
        x = loadmat(os.path.join(
            os.environ['HOME'], "data", "datasets", "tea.mat"))
        data = x["Input"][0, 0][0].transpose().reshape(-1, 3, 101, 76).transpose(
            0, 3, 2, 1) / 255
        if self.pca:
            data = data.reshape(len(data), -1)
            data = PCA(data, n_components=self.n_components)
            #data = skimage.transform.resize(data, 
        return (data.astype(np.float32), np.arange(400, dtype=np.float32))
    def as_dataset(self):
        return tf.data.Dataset.from_tensor_slices((self.data)).map(_process_image)
    def get_train_input(self):
        return self.as_dataset().cache().shuffle(buffer_size=100000).batch(self.flags.batch_size).repeat(
                self.flags.epochs_between_evals)
    def get_eval_input(self):
        return self.as_dataset().cache().batch(self.flags.batch_size).make_one_shot_iterator().get_next()
    def get_all_train_input(self):
        return self.as_dataset().make_one_shot_iterator().get_next()
    def get_predict_input(self):
        return self.as_dataset().batch(self.flags.batch_size).make_one_shot_iterator().get_next()
    def predict(self, preds):
        flags = self.flags
        savedir = flags.model_dir
        print(flags.predict)
        if flags.predict == 'mds':
            util.plot_mds(preds, savedir=savedir)
        elif flags.predict == 'heatmap':
            util.plot_heatmap(preds, repeat = True, savedir=savedir)
            #util.plot_heatmap(preds, layer_name='embedding', savedir=savedir)
        elif flags.predict == 'polar':
            util.plot_polar(preds, savedir=savedir)
            util.plot_polar(preds, transpose=True, savedir=savedir)
        elif flags.predict == 'regressor':
            util.plot_heatmap(preds, savedir=savedir)
        elif flags.predict == 'all':
            util.plot_output(preds, ishape=(28,28), num_channels=3, savedir=savedir, gshape=(20,20)) # h,w for gshape
#            util.plot_average_layer(preds, flags.height, flags.width, layer_name='ising_layer')
            util.plot_polar(preds, savedir=savedir)
            util.plot_heatmap(preds, savedir=savedir)
            util.plot_heatmap(preds, layer_name='embedding', savedir=savedir)
            util.plot_fourier(preds, layer_name='ising_layer', savedir=savedir)
            util.plot_heatmap_3color(preds, savedir=savedir)
        else:
            raise RuntimeError('Unknown predict parameter %s' % flags.predict)
    def input_width(self):
        if self.pca:
            return self.n_components
        return 28 * 28  * 3
    def get_dataset_default_flags(self):
        return  {   'batch_size': 40,
                    'train_epochs': 5000,
                    'epochs_between_evals': 1000,
                }
    def print_teapot(self, path, index = 0):
        import imageio
        x = loadmat(os.path.join(
            os.environ['HOME'], "data", "datasets", "tea.mat"))
        data = x["Input"][0, 0][0].transpose().reshape(-1, 3, 101, 76).transpose(
            0, 3, 2, 1)
        imageio.imwrite('%s%d.png' %(path, index), data[index,:,:,:])
    def print_pca(self):
        x = loadmat(os.path.join(
            os.environ['HOME'], "data", "datasets", "tea.mat"))
        data = x["Input"][0, 0][0].transpose().reshape(-1, 3, 101, 76).transpose(
            0, 3, 2, 1) / 255
        #mds = manifold.MDS().fit_transform(data)
        pca = decomposition.PCA(n_components = self.n_components).fit(data.reshape(len(data), -1))
        print(pca.singular_values_)
        print(pca.explained_variance_ratio_)
        #imageio.imwrite('%s%d.png' %(path, index), data[index,:,:,:])
            

class Generated_Dataset(Dataset):
    def __init__(self, flags, dims, n = 60000, noise = 0.0, random_state = 42):
        super().__init__(flags)
        self.dims = dims
        self.n = n
        self.noise = noise
        self.random_state = random_state
        self.include_labels = False
        np.random.seed(random_state)
        self.x, self.y = self.generate_full()
        assert self.x.shape[0] == n
        assert self.x.shape[1] == dims
        assert self.y.shape[0] == n
    def generate_full(self):
        raise NotImplementedError
    def get_train_input(self):
        return self.as_dataset().cache().shuffle(buffer_size=100000).batch(self.flags.batch_size).repeat(
                self.flags.epochs_between_evals)
    def as_dataset(self):
        if self.include_labels:
            return tf.data.Dataset.from_tensor_slices((self.x.astype(np.float32), self.y.astype(np.int32)))
        return tf.data.Dataset.from_tensor_slices((self.x.astype(np.float32)))
    def get_eval_input(self):
        return self.as_dataset().cache().batch(self.flags.batch_size).make_one_shot_iterator().get_next()
    def get_all_train_input(self):
        return self.as_dataset().make_one_shot_iterator().get_next()
    def get_predict_input(self):
        return self.as_dataset().batch(self.flags.batch_size).make_one_shot_iterator().get_next()
    def input_width(self):
        return self.dims
    def show(self, savedir=None):
        y = self.y
        if len(y.shape) > 1:
            y = y[:,0]
        phate.plot.scatter([self.x[:,0], self.x[:,1], self.x[:,2]], c=y)
        util.save_show(savedir, 'input_data')
    def predict(self, preds):
        savedir = self.flags.model_dir
        #self.plot_output(preds, savedir=savedir)
        util.plot_heatmap_d3d(preds, self.y, savedir=savedir)
        #util.plot_heatmap_d3d(preds, self.y, layer_name='embedding', savedir=savedir)
    def plot_output(self, preds, savedir=None):
        y = self.y
        if len(y.shape) > 1:
            y = y[:,0]
        ins = preds['input']
        outs = preds['prediction']
        fig = plt.figure(figsize = (10,5))
        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        print(ins.shape)
        phate.plot.scatter(ins[:,0], ins[:,1], ins[:,2], c=y, ax=ax0,legend=False)
        phate.plot.scatter(outs[:,0], outs[:,1], outs[:,2], c=y, ax=ax1, legend=False)
        util.save_show(savedir, 'output')

class Hump_Dataset(Generated_Dataset):
    def generate_full(self):
        num_humps = 10
        y = np.random.uniform(0,num_humps, size = self.n)
        x = [scipy.stats.norm.pdf(y, i,1) for i in range(num_humps)]
        x = np.transpose(np.array(x))
        x = np.repeat(x,self.dims // num_humps, axis=1)
        print(x.shape)
        z = np.random.normal(0,0.1,size=(self.n, self.dims))
        #z = np.random.normal(0,0.01,size=(self.n, self.dims))
        x += z
        #x = np.concatenate((x,z), axis=1)
        return x,y
    def plot_pca(self):
        self.pca = PCA(self.x)
        plt.scatter(self.pca[:,0], self.pca[:,1], c = self.y, s=1)
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.xticks([])
        plt.yticks([])
        plt.show()



class Dataset_3d(Generated_Dataset):
    """
    Dataset of any 3d dataset (like swiss roll, S curve, gaussian line)
    """
    def __init__(self, flags, name = 'swiss_roll', n = 60000, noise = 0.0, random_state=42):
        self.name = name
        super().__init__(flags, dims=3, n=n, noise=noise, random_state=random_state)
    def generate_full(self):
        if self.name == 'swiss_roll':
            f = datasets.make_swiss_roll
        if self.name == 's_curve':
            f = datasets.make_s_curve
        return f(self.n, noise=self.noise, random_state=self.random_state)
    def predict(self, preds):
        savedir = self.flags.model_dir
        self.plot_output(preds, savedir=savedir)
        util.plot_heatmap_d3d(preds, self.y, savedir=savedir)
    def plot_output(self, preds, savedir=None):
        y = self.y
        if len(y.shape) > 1:
            y = y[:,0]
        ins = preds['input']
        outs = preds['prediction']
        fig = plt.figure(figsize = (10,5))
        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        print(ins.shape)
        phate.plot.scatter(ins[:,0], ins[:,1], ins[:,2], c=y, ax=ax0,legend=False)
        phate.plot.scatter(outs[:,0], outs[:,1], outs[:,2], c=y, ax=ax1, legend=False)
        util.save_show(savedir, 'output')

class Hidden_Linear(Generated_Dataset):
    def __init__(self, flags, n = 60000, dims = 50, noise = 1.0, scale = 1.0, random_state = 42):
        self.scale = scale
        super().__init__(flags, dims, n=n, noise=noise, random_state=random_state)

    def generate_full(self):
        y = np.sort(np.random.uniform(0, self.scale * 10, size = (self.n, 1)))
        x = np.random.normal(scale = 1, size = (self.n, self.dims - 1))
        x = np.concatenate((y,x), axis = 1)
        return x,y

    def predict(self, preds):
        super().predict(preds)
        #self.plot_activations_over_y(preds, savedir=self.flags.model_dir)

    def plot_activations_over_y(self, preds, layer_name = 'ising_layer', savedir=None):
        # These are the colors that will be used in the plot
        color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                          '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                          '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                          '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
        outs = pd.concat([pd.DataFrame(preds[layer_name]), 
                          pd.DataFrame(self.y, columns = ['y'])], axis=1)
        outs = outs.sort_values(by=['y'])
        import matplotlib.cm as cm
        import seaborn as sns
        pal = sns.color_palette("tab10", n_colors=10)
        for rank, column in enumerate(outs):
            if column == 'y': continue
            line = plt.plot(outs['y'], 
                            outs[column],
                            lw=2.5,
                            color=color_sequence[rank])
        util.save_show(savedir, 'lineplot')


class Linear_Dataset(Generated_Dataset):
    """ Generates a line with gaussian noise in high dimensional space.
    Y ~ U(0,1) represents true position on the line. X | Y Gaussian(y_embed,1)
    """
    def __init__(self, flags, n = 60000, dims = 50, noise = 1.0, scale = 1.0, random_state = 42):
        self.scale = scale
        super().__init__(flags, dims, n=n, noise=noise, random_state=random_state)
    def generate_full(self):
        y = np.sort(np.random.uniform(0, self.scale, self.n))
        self.random_dir = scipy.stats.special_ortho_group.rvs(self.dims)[:,0]
        x = np.random.normal(np.outer(np.ones(self.n) * y, self.random_dir), scale= self.noise / (self.dims ** 2))
        return x,y
    def predict(self, preds):
        super().predict(preds)
        self.plot_activations_over_y(preds, savedir=self.flags.model_dir)
    def plot_activations_over_y(self, preds, layer_name = 'ising_layer', savedir=None):
        # These are the colors that will be used in the plot
        color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                          '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                          '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                          '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
        outs = pd.concat([pd.DataFrame(preds[layer_name]), 
                          pd.DataFrame(self.y, columns = ['y'])], axis=1)
        outs = outs.sort_values(by=['y'])
        import matplotlib.cm as cm
        import seaborn as sns
        pal = sns.color_palette("tab10", n_colors=10)
        for rank, column in enumerate(outs):
            if column == 'y': continue
            line = plt.plot(outs['y'], 
                            outs[column],
                            lw=2.5,
                            color=color_sequence[rank])
        util.save_show(savedir, 'lineplot')

class Blend_Dataset(Generated_Dataset):
    def __init__(self, flags, n = 60000, dims=60, noise=1.0, random_state=42):
        super().__init__(flags, dims, n, noise, random_state)
    def generate_full(self):
        ld = Linear_Dataset(self.flags, self.n // 2, self.dims, self.noise, 10, self.random_state)
        hd = Hub_Spoke(self.flags, self.n // 2, self.dims, self.noise, self.random_state)
        ldx, ldy = ld.generate_full()
        hdx, hdy = hd.generate_full()
        #print( np.concatenate((ldx, hdx), axis=0).shape, np.concatenate((ldy, hdy)).shape)
        return np.concatenate((ldx, hdx)), np.concatenate((ldy, hdy + 10))
    def predict(self, preds):
        super().predict(preds)
        self.plot_activations_over_y(preds, savedir=self.flags.model_dir)
        util.plot_heatmap_d3d(preds, self.y, title_suffix = 'line_only', filter_y_zero_one=True, savedir=self.flags.model_dir)
    def plot_activations_over_y(self, preds, layer_name = 'ising_layer', savedir=None, filter_line = False):
        # These are the colors that will be used in the plot
        color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                          '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                          '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                          '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
        outs = pd.concat([pd.DataFrame(preds[layer_name]), 
                          pd.DataFrame(self.y, columns = ['y'])], axis=1)
        outs = outs.sort_values(by=['y'])
        import matplotlib.cm as cm
        import seaborn as sns
        pal = sns.color_palette("tab10", n_colors=10)
        for rank, column in enumerate(outs):
            if column == 'y': continue
            line = plt.plot(outs['y'], 
                            outs[column],
                            lw=2.5,
                            color=color_sequence[rank])
        util.save_show(savedir, 'lineplot')


class Hub_Spoke(Generated_Dataset):
    """ Generate a hub of 5 genes """
    def __init__(self, flags, n = 60000, dims = 15, noise = 1.0, random_state = 42, perturb=False):
        self.perturb=perturb
        super().__init__(flags, dims, n=n, noise=noise, random_state=random_state)
    def generate_full(self):
        num_modules = 3
        sub_per_module = 2
        m = num_genes_per_module = self.dims // num_modules
        self.dims = num_modules * m
        pairwise_correlation = 0.6
        sigma = np.ones((m, m)) * pairwise_correlation
        for i in range(m):
            sigma[i,i] = 1
        sigma = scipy.linalg.block_diag(*([sigma] * num_modules))
        x = np.random.multivariate_normal(np.zeros(self.dims), self.noise * sigma, self.n)

        """
        y = np.random.binomial(1, 0.5, size=(self.n, num_modules))
        active = np.zeros((self.n, self.dims))
        for ind,i in enumerate(y):
            active[ind] = np.repeat(i, num_genes_per_module) * 10
        print(active[:10, :])
        """
        if self.perturb:
            y = np.random.choice(num_modules * sub_per_module, size=self.n)
            active = np.zeros((self.n, num_modules))
            for i,yi in enumerate(y):
                active[i, yi // sub_per_module] = 10
            active = np.repeat(active, m, axis=1)
            for i,yi in enumerate(y):
                yii = m*(yi // sub_per_module) + (m-1)
                if  yi % sub_per_module == 1:
                    active[i, yii] += 10
                if yi % sub_per_module == 2:
                    active[i, yii - 1] += 10
        else:
            y = np.random.choice(num_modules, size=self.n)
            active = np.zeros((self.n, num_modules))
            for i,yi in enumerate(y):
                active[i, yi] = 10
            active = np.repeat(active, m, axis=1)
        x = x + active
        print(np.unique(y))
        """
        y = np.random.uniform(0, 1, size=(self.n, 10))
        tmp = y.copy()
        tmp[:,1:5] += 100 * (y[:,0][:, np.newaxis])
        tmp[:,6:] += 100 * (y[:,5][:, np.newaxis])
        tmp[:,0] *= 100
        tmp[:,5] *= 100
        x = np.random.normal(tmp, scale = self.noise)
        """
        return x,y

    def generate_full_v1(self): # Saved for replication
        num_modules = 3
        m = num_genes_per_module = 5
        self.dims = num_modules * m
        pairwise_correlation = 0.6
        sigma = np.ones((m, m)) * pairwise_correlation
        for i in range(m):
            sigma[i,i] = 1
        sigma = scipy.linalg.block_diag(*([sigma] * num_modules))
        x = np.random.multivariate_normal(np.zeros(self.dims), sigma, self.n)
        y = np.random.binomial(1, 0.5, size=(self.n, num_modules))
        active = np.zeros((self.n, self.dims))
        for ind,i in enumerate(y):
            active[ind] = np.repeat(i, num_genes_per_module) * 10
        print(active[:10, :])
        x = x + active
        print(x[:10,:])
        """
        y = np.random.uniform(0, 1, size=(self.n, 10))
        tmp = y.copy()
        tmp[:,1:5] += 100 * (y[:,0][:, np.newaxis])
        tmp[:,6:] += 100 * (y[:,5][:, np.newaxis])
        tmp[:,0] *= 100
        tmp[:,5] *= 100
        x = np.random.normal(tmp, scale = self.noise)
        """
        return x,y
    def predict(self, preds):
        savedir = self.flags.model_dir
        #util.plot_heatmap_d3d_cat(preds, self.y, savedir=savedir, title_suffix='_y')
        util.plot_heatmap_d3d_cat(preds, self.y, savedir=savedir, title_suffix='_y_abs', abs_val = True)
        self.plot_output(preds, savedir=savedir)
        self.plot_output_pca(preds, savedir=savedir)
        #util.plot_heatmap_d3d_cat(preds, self.y, savedir=savedir, title_suffix='y1')
        #util.plot_heatmap_d3d_cat(preds, self.y, savedir=savedir, title_suffix='y2')
        """
        util.plot_heatmap_d3d(preds, self.x[:,0], savedir=savedir)
        util.plot_heatmap_d3d(preds, self.x[:,5], savedir=savedir, title_suffix='y5')
        util.plot_heatmap_d3d(preds, self.x[:,10], savedir=savedir, title_suffix='y10')
        """
    def plot_pca(self):
        self.pca = PCA(self.x)
        print(np.unique(self.y))
        cmap = plt.get_cmap('tab10')
        for i in np.unique(self.y):
            d = self.pca[self.y == i]
            plt.scatter(d[:,0], d[:,1], c = cmap(i), label=i, s=1)
            plt.xlabel('PCA 1')
            plt.ylabel('PCA 2')
            plt.xticks([])
            plt.yticks([])
        plt.show()
    def plot_output_pca(self, preds, savedir=None):
        y = self.y
        if len(y.shape) > 1:
            y = y[:,0]
        ins = preds['input']
        outs = preds['prediction']
        fig = plt.figure(figsize = (10,5))
        ax0 = fig.add_subplot(1, 2, 1)#, projection='3d')
        ax1 = fig.add_subplot(1, 2, 2)#, projection='3d')
        print(ins.shape)
        pins = PCA(ins, n_components=2)
        pouts = PCA(outs, n_components=2)
        phate.plot.scatter(pins[:,0], pins[:,1],  \
                c=y, ax=ax0, xlabel = 'PCA 1', ylabel = 'PCA 2', legend=False)
        phate.plot.scatter(pouts[:,0], pouts[:,1],  \
                c=y, ax=ax1, xlabel = 'PCA 1', ylabel = 'PCA 2', legend=False)
        util.save_show(savedir, 'output_pca')



class Gene_Interaction(Generated_Dataset):
    """
    N gene modules of M nodes in input data with N disconnected ising components
    """
    def __init__(self, flags, n = 500, dims = 100, noise = 1.0, wbias = 0.5, random_state = 42):
        self.wbias = wbias
        super().__init__(flags, dims, n=n, noise=noise, random_state=random_state)
        self.include_labels = True

    def generate_full(self):
        self.w = np.zeros(self.dims)
        lim = int(self.dims * 0.4)
        self.w[:lim // 2] = self.wbias
        self.w[lim // 2 : lim] = -self.wbias
        self.w[:lim] = np.random.normal(self.w[:lim], self.noise)
        print(self.w)
        sigma = np.eye(self.dims)
        for i in range(lim):
            for j in range(lim):
                if i == j:
                    continue
                sigma[i,j] = 0.6
        self.A = sigma
        x = np.random.multivariate_normal(np.zeros(self.dims), sigma, 500)
        ps = []
        for xi in x:
            p = 1 / (1 + np.exp(-np.dot(self.w, xi))) # sigmoid(wx)
            ps.append(p)
        ps = np.array(ps)
        y = np.random.binomial(1, ps).astype(np.int32)
        #y = (ps > 0.5).astype(np.int32) # Should this be bernolli?
        A = np.random.binomial(1, 0.1, (self.dims, self.dims))
        A[:lim, :lim] = np.random.binomial(1, 0.3, (lim, lim))
        x = x + 3 * np.random.normal(0, 1, (500, 100))
        x = x.astype(np.float32)
        y = y.astype(np.int32)
        self.xtrain, self.xtest = x[:300], x[300:]
        self.ytrain, self.ytest = y[:300], y[300:]
        return x,y
    """
    def get_train_input(self):
        return self.as_dataset(self.xtrain, self.ytrain, self.flags.epochs_between_evals)
    def as_dataset(self, x, y, num_epochs):
        return tf.estimator.inputs.pandas_input_fn(
            pd.DataFrame(x, columns = [str(i) for i in range(100)]),
            y = pd.Series(y),
            num_epochs = num_epochs,
            batch_size = self.flags.batch_size,
            num_threads = 1,
            shuffle = True,
        )()
    def get_eval_input(self):
        return self.as_dataset(self.xtest, self.ytest, 1)
    def get_predict_input(self):
        return self.as_dataset(self.xtest, self.ytest, 1)
    """
    def get_train_input(self):
        return self.as_dataset(self.xtrain, self.ytrain).cache().shuffle(
                    buffer_size=100000).batch(self.flags.batch_size).repeat(
                    self.flags.epochs_between_evals)
    def as_dataset(self, x, y = None):
        if self.include_labels:
            return tf.data.Dataset.from_tensor_slices((x.astype(np.float32), y.astype(np.int32)))
        return tf.data.Dataset.from_tensor_slices((x.astype(np.float32)))
    def get_eval_input(self):
        return self.as_dataset(self.xtest, self.ytest).batch(self.flags.batch_size)
    def get_predict_input(self):
        return self.as_dataset(self.xtest, self.ytest).batch(self.flags.batch_size)
    def get_network(self):
        return (self.A > 0).astype(np.int32)
        return self.A

def gen_teapot_pca(flags):
    path = 'teapot_pca/'
    for i in range(100):
        d = Teapot_Dataset(flags, pca=True)
        x,y = d.data[:,0], d.data[:,1]
        np.save('%s%d.npy' % (path, i), d.data)
        fig, ax = plt.subplots(1)
        fig.set_size_inches(8,6)
        ax.scatter(x,y,c=range(400))
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig('%s%d.png' % (path, i))
        plt.close()

def plot_teapot_pca(flags):
    data = np.load('teapot_pca/97.npy')
    x,y = data[:,0], data[:,1]
    fig, ax = plt.subplots(1)
    fig.set_size_inches(8,6)
    ax.scatter(x,y,c=range(400))
    marked = range(0,400, 80)
    xm = x[marked]
    ym = y[marked]
    ax.scatter(xm, ym, c='r')
    print(marked)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig('teapot_pca.png')
    plt.close()
    #plt.show()

def plot_teapot_mds():
    x = loadmat(os.path.join(
        os.environ['HOME'], "data", "datasets", "tea.mat"))
    data = x["Input"][0, 0][0].transpose().reshape(-1, 3, 101, 76).transpose(
        0, 3, 2, 1) / 255
    data = manifold.MDS(n_jobs = 1, n_init = 40).fit_transform(data.reshape(len(data),-1))
    x,y = data[:,0], data[:,1]
    fig, ax = plt.subplots(1)
    fig.set_size_inches(8,6)
    ax.scatter(x,y,c=range(400))
    marked = range(0,400, 80)
    xm = x[marked]
    ym = y[marked]
    ax.scatter(xm, ym, c='r')
    print(marked)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig('teapot_mds.png')
    #plt.close()
    plt.show()

class Noonan_Dataset(Dataset):
    def __init__(self, flags):
        path = 'RM3'
        self.data = pd.read_pickle(path).astype('float32').to_dense()
        self.proj = np.load('14.5_2dprojection.npy')[:,:self.data.shape[0]]
        print(self.proj)
        
    def as_dataset(self):
        return tf.data.Dataset.from_tensor_slices((self.data))
    def get_train_input(self):
        return self.as_dataset().cache().shuffle(buffer_size=100000).batch(self.flags.batch_size).repeat(
                self.flags.epochs_between_evals)
    def get_eval_input(self):
        return self.as_dataset().cache().batch(self.flags.batch_size).make_one_shot_iterator().get_next()
    def get_all_train_input(self):
        return self.as_dataset().make_one_shot_iterator().get_next()
    def get_predict_input(self):
        return self.as_dataset().batch(self.flags.batch_size).make_one_shot_iterator().get_next()
    def input_width(self):
        return self.data.shape[1]
    def get_dataset_default_flags(self):
        """ Dictionary containing default flag overrides """
        return {
                'batch_size': 50,
        }
    def predict(self, preds):
        savedir = self.flags.model_dir
        
        print(self.data['Tbr1'].values)
        util.plot_heatmap_d3d(preds, self.proj[:,0], savedir=savedir, title_suffix='Tbr1')

class Mnist_Variation(Dataset):
    def __init__(self, flags):
        super().__init__(flags)
        self.url = 'http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_back_image_new.zip'
        self.fpath = self.download(flags.data_dir)
    def amat_to_numpy(self, directory, fname, np_name):
        fpath = os.path.join(directory, fname)
        with open(fpath, 'r') as f:
            arr = np.loadtxt(f, dtype=np.float32)
        np.save(os.path.join(directory, np_name), arr)
    def download(self, directory):
        # This seems backwards to me, but it seems to be right based on file size
        train_path = 'mnist_all_background_images_rotation_normalized_test.amat'
        test_path = 'mnist_all_background_images_rotation_normalized_train_valid.amat'
        filepath = os.path.join(directory, 'mnist_rotation_back_image_new')
        if tf.gfile.Exists(filepath):
            return filepath
        if not tf.gfile.Exists(directory):
            tf.gfile.MakeDirs(directory)
        _, zipped_filepath = tempfile.mkstemp(suffix='.zip')
        print('Downloading %s to %s' % (self.url, zipped_filepath))
        urllib.request.urlretrieve(self.url, zipped_filepath)
        with zipfile.ZipFile(zipped_filepath, 'r') as f_in:
            f_in.extractall(filepath)
        self.amat_to_numpy(filepath, train_path, 'train.npy')
        self.amat_to_numpy(filepath, test_path, 'test.npy')
        return filepath
    def parse_to_dataset(self, name):
        print(self.fpath, name, os.path.join(self.fpath, name))
        data = np.load(os.path.join(self.fpath, name))
        features = data[:, :-1]
        labels = data[:,-1].astype(np.int32)
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        return dataset
    def get_data(self, dataset):
        assert dataset in ['train', 'test']
        return self.parse_to_dataset(dataset + '.npy')
    def get_train_input(self):
        return self.get_data('train').shuffle(buffer_size=100000).batch(self.flags.batch_size).repeat(
                self.flags.epochs_between_evals)
    def get_eval_input(self):
        return self.get_data('train').batch(self.flags.batch_size).make_one_shot_iterator().get_next()
    def get_all_train_input(self):
        return self.get_data('train').make_one_shot_iterator().get_next()
    def get_predict_input(self):
        return self.get_data('test').batch(self.flags.batch_size).make_one_shot_iterator().get_next()
    def input_width(self):
        return 28*28
#    def predict

class Affnist(Dataset):
    def load_records(self, directory, return_transform=False, get_test=False):
        files = [os.path.join(directory, '%d.tfrecords' %i) for i in range(32)]
        path = os.path.join(directory, '1.tfrecords')
        def _parse_function(example_proto):
            keys_to_features = {
                    'image': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.int64),
                    'human_readable_transform': tf.FixedLenFeature([], tf.string)
            }
            parsed_record = tf.parse_single_example(example_proto, keys_to_features)
            image = tf.cast(tf.decode_raw(parsed_record['image'], tf.uint8), tf.float32) / 255.0
            label = tf.cast(parsed_record['label'], tf.int32)
            transform = tf.cast(tf.decode_raw(parsed_record['human_readable_transform'], tf.float64), tf.float32)
            if return_transform:
                return image, label, transform
            return image, label
        dataset = tf.data.TFRecordDataset(path).map(_parse_function)
        sess = tf.Session()
        value = dataset.make_one_shot_iterator().get_next()
        print(sess.run(value))
        print(sess.run(value)[0].shape)
        #print(dataset)
        #print(dataset.output_shapes)
        return dataset
    def get_data(self, dataset):
        assert dataset in ['train', 'test']
        DIR = './affnist/transformed/'
        TRAIN_DIR = 'training_and_validation_batches'
        TEST_DIR = 'test_batches'
        dir = os.path.join(DIR, TEST_DIR if dataset == 'test' else TRAIN_DIR)
        return self.load_records(dir)
    def get_train_input(self):
        return self.get_data('train').shuffle(buffer_size=100000).batch(self.flags.batch_size).repeat(
                self.flags.epochs_between_evals)
    def get_eval_input(self):
        return self.get_data('train').batch(self.flags.batch_size).make_one_shot_iterator().get_next()
    def get_all_train_input(self):
        return self.get_data('train').make_one_shot_iterator().get_next()
    def get_predict_input(self):
        return self.get_data('test').batch(self.flags.batch_size).make_one_shot_iterator().get_next()
    def input_width(self):
        return 40*40

if __name__ == '__main__':
    import args
    flags = args.Namespace(**{
        "args_from_file": "teaargs",
        "batch_regularization": 1e-05,
        "batch_size": 500,
        "data_dir": "/tmp/mnist_data",
        "data_format": "channels_last",
        "dataset": "teapot",
        "decay": 1,
        "epochs_between_evals": 10,
        "export_dir": "./exported",
        "height": 1,
        "ising_graph_type": "rings",
        "ising_regularization": 0.001,
        "layer_num": 3,
        "learning_rate": 0.0001,
        "middle_width": 3,
        "model_dir": "./",
        "predict": False,
        "train_and_predict": False,
        "train_epochs": 40,
        "use_classifier": False,
        "width": 10
    })
    #d = Mnist_Dataset(flags)
    #d.get_embedding()
    #d.plot_embed()
    #print('phating')
    #d.plot_embed()
    #exit(0)
    #d = Teapot_Dataset(flags)
    #print(d.data.shape)
    d = Wishbone_Dataset(flags)
    for c in d.data.columns:
        print(c)

    #d.create_plot()
