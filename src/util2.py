from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.datasets import mnist
from tensorflow.keras.backend import set_session
import tensorflow as tf
import  matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition
import time

def PCA(data, n_components=2, **kwargs):
    pca = decomposition.PCA(n_components=n_components).fit(data)
    return pca, pca.transform(data)

def save(model, path):
    mjson = model.to_json()
    with open(path + '.json', "w") as f:
        f.write(mjson)
    model.save_weights(path+'.h5')

def load_model(path):
    with open(path + '.json', 'r') as f:
        json = f.read()
    m = model_from_json(json)
    #import json as js
    #d = js.loads(json)
    #print(js.dumps(d, sort_keys=True, indent=4))
    m.load_weights(path + '.h5')
    return m

def perform_steps(model, test, stepsize = 1, steps = 100):
    """ Step size is between 0 and 1 where 1 is a full step and zero is no step by linear interpolation."""
    imgs = [test]
    for i in range(steps):
        imgs.append(imgs[-1] * (1 - stepsize) + model.predict(imgs[-1]) * stepsize)
    return imgs

def plot_model(path, steps=10000, stepsize=0.001, suffix = None):
    _, x_test = load_data()
    model = load_model(path)
    imgs = perform_steps(model, np.repeat(x_test[:20], 2, axis=0), steps=10000, stepsize=0.001)
    imgs = imgs[::1000]
    plot(imgs, savedir='imgs', title='small_steps' + suffix)
    #imgs = perform_steps(model, x_test[:20], steps=30, stepsize=1)
    #plot(imgs, savedir='imgs', title = 'plot' + suffix)

def plot(imgs, title = 'plot', savedir=None, n=20):
    plt.figure(figsize=(40, 2*len(imgs)))
    for i in range(n):
        for j in range(len(imgs)):
            # display original
            ax = plt.subplot(len(imgs), n, i+1+j*n)
            plt.imshow(imgs[j][i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    save_show(savedir, title)

def save_show(savedir, name):
    if savedir is not None:
        plt.savefig(savedir + '/' + name)
        plt.close()
    else:
        plt.show()

def load_data(return_labels=False):
    (x_train, x_train_lab), (x_test, x_test_lab) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
    if return_labels:
        return x_train, x_train_lab, x_test, x_test_lab
    return x_train, x_test

def load_data_fives():
    train, train_lab, test, test_lab = load_data(return_labels=True)
    return train[train_lab == 5], test[test_lab == 5]

def set_config():
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth=True
    conf.gpu_options.per_process_gpu_memory_fraction=0.4
    set_session(tf.Session(config=conf))
    np.random.seed(42)
    tf.set_random_seed(42)
    tf.logging.set_verbosity('ERROR')

def scatter_plot2d(data, labels, title = None, fig = None, ax = None, xlabel = None, ylabel = None, set_legend = False, vmin=None, vmax=None):
    discrete_labels = (labels.dtype == np.int32 or labels.dtype == np.int64)
    if fig is None or ax is None: 
        fig, ax = plt.subplots(1)
        fig.set_size_inches(10,8)
    if discrete_labels:
        cmap = plt.get_cmap('tab10')
        if len(np.unique(labels)) > 10:
            cmap = plt.get_cmap('tab20')
        for i in np.unique(labels):
            d = data[labels == i]
            sc = ax.scatter(d[:,0], d[:,1], c=cmap(i), label=i, s=1, vmin=vmin, vmax=vmax)
    else:
        sc = ax.scatter(data[:,0], data[:,1], c=labels, s=1, vmin=vmin, vmax=vmax)

    ax.set_xticks([])
    ax.set_yticks([])
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=15)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=15)
    if title is not None:
        ax.set_title(title)
    if set_legend:
        if discrete_labels:
            ax.legend(fontsize='xx-large', markerscale=10)
        else:
            fig.colorbar(sc, ticks=np.linspace(np.min(labels), np.max(labels), 5))
    return sc

def npdo_list(f, paths):
    assert all([path.endswith('.npy') for path in paths])
    try:
        tmp = [np.load(p) for p in paths]
        print('Successfully loaded from file')
        return tmp
    except FileNotFoundError as inst:
        print('File not found, running given function and storing')
        start = time.time()
        out = f()
        end = time.time()
        print('Took: %d seconds.' % (end - start))
        [np.save(p,o) for o,p in zip(out, paths)]
        print([p for o,p in zip(out, paths)])
        return out

def npdo(f, path):
    assert path.endswith('.npy')
    try:
        tmp = np.load(path)
        print('Successfully loaded from file')
        return tmp
    except FileNotFoundError as inst:
        print('File not found, running given function and storing')
        start = time.time()
        out = f()
        end = time.time()
        print('Took: %d seconds.' % (end - start))
        np.save(path, out)
        return out
