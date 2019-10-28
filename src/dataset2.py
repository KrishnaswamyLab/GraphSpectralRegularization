import tensorflow.keras.datasets as kd
import util
import numpy as np
import graphtools
import time
import pygsp

def keras_image_format_to_std(data):
    data = (data.astype(np.float32) - 127.5) / 127.5
    if len(data.shape) == 3:
        data = np.expand_dims(data, axis=3)
    return data

class Dataset(object):
    def __init__(self):
        self.embed_dir = './embeddings'
    def get_train(self): raise NotImplementedError
    def get_test(self): raise NotImplementedError
    def get_shape(self): raise NotImplementedError 
    def is_image_type(self): return False
    def get_embedding(self): raise NotImplementedError
    def factory(name):
        if name == 'mnist': return Keras_Dataset(kd.mnist, name)
        if name == 'cifar10': return Keras_Dataset(kd.cifar10, name)
        if name == 'cifar_ship_deer': return Keras_Dataset(kd.cifar10, name, label_subset = [4,8])
        if name == 'fashion_mnist': return Keras_Dataset(kd.fashion_mnist, name)
        if name == 'fashion_mnist_shirt_boot': return Keras_Dataset(kd.fashion_mnist, label_subset = [6,9], name=name)
        if name == 'cifar100': return Keras_Dataset(kd.cifar100, name)
        if name == 'mnist5': return Mnist_Fives_Dataset()
        if name == 'mnist5_small7': return Mnist_Fives_Small_Sevens_Dataset()
        if name == 'mnist05': return Mnist_Digit_Dataset([0,5], 'zeros_and_fives')
        if name == 'mnist45': return Mnist_Digit_Dataset([4,5], 'fours_and_fives')
        raise AssertionError('Bad Dataset Creation: %s' % name)
    factory = staticmethod(factory)
    def get_dataset_default_flags(self): return {}


class Keras_Dataset(Dataset):
    def __init__(self, data_loader, name, label_subset=None):
        super().__init__()
        self.data = data_loader.load_data()
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self.data
        self.embed_name = ('%s_phate' % name)
        self.name = name
        self.label_subset = np.unique(self.train_labels)
        if label_subset is not None:
            self.subset_labels(label_subset)


    def subset_labels(self, label_subset): 
        if not (isinstance(label_subset, list) or isinstance(label_subset, tuple)):
            label_subset = (label_subset,)
        def filter_labels(labels, cl):
            mask = np.zeros_like(labels, dtype=np.bool)
            for c in cl:
                mask |= labels == c
            return mask
        train_idx = filter_labels(self.train_labels, label_subset)
        test_idx  = filter_labels(self.test_labels, label_subset)
        self.train_images = self.train_images[train_idx]
        self.train_labels = self.train_labels[train_idx]
        self.test_images = self.test_images[test_idx]
        self.test_labels = self.test_labels[test_idx]
        self.label_subset = label_subset
        print('%s dataset has %d training images and %d test images' %
              (self.name, len(self.train_labels), len(self.test_labels)))

    def get_train(self):
        return keras_image_format_to_std(self.train_images)
    def get_test(self):
        return keras_image_format_to_std(self.test_images)
    def get_shape(self):
        keras_image_format_to_std(self.test_images).shape[1:]
        
        return keras_image_format_to_std(self.test_images).shape[1:]
    def is_image_type(self):
        return True
    def get_train_labels(self):
        return self.train_labels
    def get_embedding(self):
        def do_phate():
            import phate
            phate_op = phate.PHATE(k=5, a=None, t=52, n_jobs=-2)
            #phate_op = phate.PHATE(k=5, a=None, t=105, n_jobs=-2)
            return phate_op.fit_transform(np.reshape(self.train_images, (-1, 28*28)))
        return util.npdo(do_phate, self.embed_dir + '%s/%s.npy' %(self.embed_dir, self.embed_name))
    def plot_embed(self, labels = None, title=None, fig = None, ax = None, set_legend=True, vmin=None, vmax=None):
        if labels is None: labels = self.train_labels.astype(np.int32)
        sc = util.scatter_plot2d(self.get_embedding(), 
                                 labels, 'PHATE k=5 a=None, t=52' if title is None else title, 
                                 set_legend=set_legend, fig=fig, ax = ax, vmin=vmin, vmax=vmax)
        if fig is None: util.save_show(self.embed_dir, self.embed_name if title is None else title)
        return sc
    def subset(self, num):
        self.train_images = self.train_images[:num]
        self.train_labels = self.train_labels[:num]
        self.test_images = self.test_images[:num]
        self.test_labels = self.test_labels[:num]

class Mnist_Dataset(Keras_Dataset):
    def __init__(self):
        super().__init__(kd.mnist, 'mnist')
    def build_graph(self):
        # TODO incomplete
        start = time.time()
        G = graphtools.Graph(np.reshape(self.train_images, (-1, 28*28))[:2000],
                             n_pca=100,
                             n_jobs = -2,
                             use_pygsp=True,
                            )
        # G.diff_op is sparse, make it dense before matrix power
        #print(np.linalg.matrix_power(G.diff_op.toarray(), 5))
        #print(pygsp.utils.resistance_distance(G))
        end = time.time()
        print('time in seconds: %0.4f' % (end - start))

class Mnist_Digit_Dataset(Mnist_Dataset):
    def __init__(self, label_subset, subset_name = 'digit', verbose=False):
        """ Initializes Mnist with subset of digits.
        Takes a digit or list of digits to subset.
        """
        super().__init__()
        self.subset_name = subset_name
        if not (isinstance(label_subset, list) or isinstance(label_subset, tuple)):
            label_subset = (label_subset,)
        def filter_labels(labels, cl):
            mask = np.zeros_like(labels, dtype=np.bool)
            for c in cl:
                mask |= labels == c
            return mask

        train_idx = filter_labels(self.train_labels, label_subset)
        test_idx  = filter_labels(self.test_labels, label_subset)
        self.train_images = self.train_images[train_idx]
        self.train_labels = self.train_labels[train_idx]
        self.test_images = self.test_images[test_idx]
        self.test_labels = self.test_labels[test_idx]
        self.embed_name = 'mnist_%s_phate' % self.subset_name
        self.label_subset = label_subset
        if verbose:
            print('Mnist %s dataset has %d training images and %d test images' %
                    (self.subset_name, len(self.train_labels), len(self.test_labels)))

class Mnist_Fives_Dataset(Mnist_Digit_Dataset):
    def __init__(self):
        super().__init__(5, subset_name = 'fives')

class Mnist_Fives_Small_Sevens_Dataset(Mnist_Digit_Dataset):
    def __init__(self, num_fives=5000, num_sevens=100):
        super().__init__([5,7], subset_name='five_small_seven')
        self.num_fives = num_fives
        self.num_sevens = num_sevens
        d_fives = Mnist_Digit_Dataset(5)
        d_fives.subset(num_fives)
        d_sevens = Mnist_Digit_Dataset(7)
        d_sevens.subset(num_sevens)
        self.train_labels = np.concatenate([d_fives.train_labels, d_sevens.train_labels])
        self.train_images = np.concatenate([d_fives.train_images, d_sevens.train_images])
        self.test_labels = np.concatenate([d_fives.test_labels, d_sevens.test_labels])
        self.test_images = np.concatenate([d_fives.test_images, d_sevens.test_images])
        print('Mnist %s dataset has %d training images and %d test images' %
                (self.subset_name, len(self.train_labels), len(self.test_labels)))

if __name__ == '__main__':
    #d = Keras_Dataset(kd.cifar10, name='cifar10')
    d = Mnist_Fives_Small_Sevens_Dataset()
    
    import tensorflow.keras.backend as K
    #d = Mnist_Digit_Dataset([0,5])
    #d.get_train()
    #d.plot_embed()
