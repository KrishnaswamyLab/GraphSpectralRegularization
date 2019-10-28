import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import time
import math
import sklearn

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

def process_predictions(predictions):
    preds = {}
    for p in predictions:
        for k,v in p.items():
            if k not in preds:
                preds[k] = []
            preds[k].append(v)
    preds = {k : np.concatenate(v) for k,v in preds.items()}
    return preds

def save_show(savedir, name):
    if savedir is not None:
        plt.savefig(savedir + '/' + name)
        plt.close()
    else:
        plt.show()

def plot_heatmap_3color(preds, layer_name='ising_layer', repeat = False, savedir=None):
    layer_outs = preds[layer_name]
    g = sns.heatmap(layer_outs, 
            yticklabels=False, 
            cbar=True,
    )
    #g.set_title('Ising Layer Heatmap')
    if repeat:
        save_show(savedir, 'repeat_heatmap_' + layer_name)
    else:
        save_show(savedir, 'heatmap_' + layer_name)

def plot_heatmap(preds, layer_name='ising_layer', repeat = False, savedir=None):
    layer_outs = preds[layer_name]
    if repeat:
        layer_outs = pd.DataFrame(np.tile(layer_outs, [2,2]))
        row_colors = pd.DataFrame(np.repeat(['r', 'g'], 400), columns=['run'])
        g = sns.clustermap(layer_outs,
                #row_colors = pd.concat([row_colors, row_colors2], axis=1),
                row_colors = row_colors if repeat else None,
                row_cluster=False,
                col_cluster=False,
                yticklabels=False,
        )
        g.ax_heatmap.set_title('Ising Layer Heatmap')
    else:
        g = sns.heatmap(layer_outs, 
                yticklabels=False, 
                cbar=True,
        )
        #g.set_title('Ising Layer Heatmap')
    if repeat:
        save_show(savedir, 'repeat_heatmap_' + layer_name)
    else:
        save_show(savedir, 'heatmap_' + layer_name)

def plot_polar(preds, layer_name='ising_layer', transpose = False, savedir=None):
    layer_outs = pd.DataFrame(preds[layer_name])
    if transpose:
        layer_outs = layer_outs.sample(20).T
        layer_outs = layer_outs.append(layer_outs.iloc[0])
        colors = list(map(cm.autumn, np.arange(400) / 400))
        for i,d in layer_outs.T.iterrows():
            plt.polar(np.arange(layer_outs.shape[1] + 1)  / 10 * 2 * math.pi, d, c=colors[i])
        save_show(savedir, 'polart_' + layer_name)
    else:
        plt.polar(np.arange(400)  / 400 * 2 * math.pi, layer_outs)
        save_show(savedir, 'polar_' + layer_name)
    #plt.show()

def plot_mds(preds, layer_name='ising_layer', savedir=None):
    layer_outs = pd.DataFrame(preds[layer_name]).astype(np.float64)
    mds = sklearn.manifold.MDS().fit_transform(layer_outs)
    plt.scatter(mds[:,0], mds[:,1], c = range(400))
    save_show(savedir, 'mds_' + layer_name)

def plot_mnist_heatmap(preds, training_labels, layer_name='ising_layer', savedir=None):
    layer_outs = preds[layer_name]
    tl = pd.DataFrame(training_labels, columns = ['labels'])
    outs = pd.concat([pd.DataFrame(layer_outs), tl], axis=1).sort_values(by='labels')
    import matplotlib.cm as cm
    pal = sns.color_palette("tab10", n_colors=10)
    lut = dict(zip(outs.labels.unique(), pal))
    row_colors = outs.labels.map(lut)
    g = sns.clustermap(outs.drop(columns=['labels']),
            row_colors=row_colors,
            row_cluster=False,
            col_cluster=False,
            yticklabels=False,
    )
    for label in range(10):
        g.ax_col_dendrogram.bar(0,0, color=lut[label], label=label, linewidth=0)
    g.ax_col_dendrogram.legend(loc='center', ncol=5)
    g.ax_heatmap.set_title('Mnist Heatmap')
    save_show(savedir, 'heatmap_' + layer_name)

def plot_heatmap_d3d(preds, y, layer_name='ising_layer', title_suffix='', filter_y_zero_one = False, savedir=None):
    outs = pd.concat([pd.DataFrame(preds[layer_name]), 
                      pd.DataFrame(y, columns = ['y'])], axis=1)
    outs = outs.sample(min(outs.shape[0],1000))
    outs = outs.sort_values(by=['y'])
    if filter_y_zero_one:
        outs = outs.loc[outs.y < 1].loc[outs.y > 0]
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    cmap = cm.Reds
    norm = Normalize(vmin=outs.y.min(), vmax=outs.y.max())
    #norm = Normalize(vmin=outs.y.min(), vmax=outs.y.max())
    row_colors = outs.y.apply(lambda x: cmap(norm(x)))
    g = sns.clustermap(outs.drop(columns=['y']), 
            row_colors=row_colors,
            row_cluster=False,
            col_cluster=False,
            yticklabels=False,
    #        vmin = -40,
    #        vmax = 40,
    )
    g.ax_heatmap.set_title('Activation Heatmap')
    save_show(savedir, 'heatmap_' + layer_name + title_suffix)

def transform_outputs(df):
    #transformed = df / (df.max() - df.min())
    transformed = df.applymap(lambda x: np.tanh(x/2))
    return transformed

def plot_heatmap_d3d_cat(preds, y, layer_name='ising_layer', title_suffix='', abs_val=False, savedir=None):
    output = pd.DataFrame(preds[layer_name])
    if abs_val:
        output = transform_outputs(output)
    outs = pd.concat([output, 
                      pd.DataFrame(y, columns = ['y'])], axis=1)
    outs = outs.sample(min(outs.shape[0],1000))
    outs = outs.sort_values(by=['y'])
    import matplotlib.cm as cm
    pal = sns.color_palette("tab10", n_colors=10)
    lut = dict(zip(outs.y.unique(), pal))
    row_colors = outs.y.map(lut)
    g = sns.clustermap(outs.drop(columns=['y']), 
            row_colors=row_colors,
            row_cluster=False,
            col_cluster=False,
            yticklabels=False,
            #center=0,
            #cmap = "RdBu_r",
    )
    g.ax_heatmap.set_title('Activation Heatmap')
    save_show(savedir, 'heatmap_' + layer_name + title_suffix)

def plot_wishbone_heatmap(preds, data, layer_name='ising_layer', even_trajectory=True, savedir=None):
    layer_outs = preds[layer_name]
    df = data.full_data
    outs = pd.concat([pd.DataFrame(layer_outs), df.Trajectory, df.Branch], axis=1)
    if even_trajectory:
        d = []
        for b,g in outs.groupby('Branch'):
            d.append(g.sample(333))
        outs = pd.concat(d)
    else:
        outs = outs.sample(1000)
    outs = outs.sort_values(by=['Branch', 'Trajectory'])
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
            col_cluster=False,
            yticklabels=False,
            cmap = "RdBu_r",
            #cmap = cm.Reds,
    )
    g.ax_heatmap.set_title('Ising Layer Wishbone Heatmap')
    save_show(savedir, 'heatmap_' + layer_name)

def plot_wishbone_heatmap2(preds, data, layer_name='ising_layer', even_trajectory=True, savedir=None):
    layer_outs = preds[layer_name]
    df = data.full_data
    vmax = np.max(layer_outs)
    vmin = np.minimum(0.0, np.min(layer_outs))
    outs = pd.concat([pd.DataFrame(layer_outs), df.Trajectory, df.Branch], axis=1)
    if even_trajectory:
        d = []
        for b,g in outs.groupby('Branch'):
            d.append(g.sample(333))
        outs = pd.concat(d)
    else:
        outs = outs.sample(1000)

    def plot_outs(outs, i):
        outs = outs.sort_values(by=['Branch', 'Trajectory'])
        lut = dict(zip([1,2,3], 'rbg'))
        row_colors = outs.Branch.map(lut)
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm
        cmap = cm.Reds
        norm = Normalize(vmin=outs.Trajectory.min(), vmax=outs.Trajectory.max())
        row_colors2 = outs.Trajectory.apply(lambda x: cmap(norm(x)))
        g = sns.clustermap(outs.drop(columns=['Trajectory', 'Branch']), 
                row_colors = pd.concat([row_colors, row_colors2], axis=1),
                row_cluster=False,
                col_cluster=False,
                yticklabels=False,
                vmax = vmax,
                vmin = vmin,
                cmap = cm.Reds,
        )
        g.ax_heatmap.set_title('Ising Layer Wishbone Heatmap')
        save_show(savedir, 'heatmap%d_' %i  + layer_name)
    
    print('aaaaaaaaaaaaaaaaaaaa')
    print(outs.Branch.unique())
    plot_outs(outs[outs.Branch != 2], 3)
    plot_outs(outs[outs.Branch != 3], 2)

def plot_wishbone_embedding_scatter_plot(preds, data, savedir=None):
    from mpl_toolkits.mplot3d import axes3d
    outs = preds['embedding']
    df = data.full_data
    #outs = pd.concat([pd.DataFrame(outs), df.Trajectory, df.Branch], axis=1)
    fig = plt.figure()
    if outs.shape[1] == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(outs[:,0], outs[:,1], outs[:,2], c=df.Branch, s=1)
    if outs.shape[1] == 2:
        ax = fig.add_subplot(111)
        ax.scatter(outs[:,0], outs[:,1], c = df.Branch, s=1)
    save_show(savedir, 'scatter')

def plot_fourier(preds, layer_name = 'ising_layer', savedir=None):
    fig = plt.figure()
    ax = fig.add_subplot(1,1, 1)
    



def plot_class_fourier(preds, labels, height, width, layer_name = 'ising_layer', ignore_mean = True, savedir=None):
    h,w = (height, width)
    fig = plt.figure(figsize = (10,3))
    cols, rows = (2, 5)
    if labels is None: # Use output instead of input if not supplied
        labels = preds['classes']
    imgs = []
    stds = []
    for cls in range(10):
        matches = np.argwhere(labels == cls).flatten()
        matched_layers = preds[layer_name][matches]
        matched_layers = np.reshape(matched_layers, [-1, h, w])
        #matched_layers = np.reshape(matched_layers, [-1, w, h])
        fourier_amplitudes = np.real(np.fft.rfft(matched_layers))
        #fourier_amplitudes = np.real(np.fft.rfft(matched_layers, axis=1))
        print(np.mean(fourier_amplitudes, axis=0).shape)
        #imgs.append(np.reshape(np.mean(fourier_amplitudes, axis=0),[h,-1]))
        mean_fourier = np.mean(fourier_amplitudes, axis=0)
        if ignore_mean:
            mean_fourier = mean_fourier[:,1:]
        imgs.append(mean_fourier)
        stds.append(np.reshape(np.std(fourier_amplitudes, axis=0), [h,-1]))
    imgs = np.array(imgs)
    stds = np.array(stds)
    for cls,(img,std) in enumerate(zip(imgs, stds)):
        ax = fig.add_subplot(cols, rows, cls+1)
        ax.set_title(cls)
        ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)
        if height == 1: # plot curves instead
            img = img.flatten()
            std = std.flatten()
            n = len(img)
            sc = ax.errorbar(np.arange(n), img, yerr=std, linestyle='-')
            #plt.ylim = (np.min(imgs) - 1, np.max(imgs + 1))
        elif height == 2:
            for i in range(height):
                sc = ax.errorbar(np.arange(img.shape[1]), img[i,:], yerr=std[i,:], linestyle='-')
        else: # Colorbar plots
            sc = ax.imshow(img, vmin=np.min(imgs), vmax=np.max(imgs))
            fig.colorbar(sc)
    save_show(savedir, 'fourier averages')

def plot_misses(preds, height, width, layer_name = 'ising_layer', 
        threshold=False, labels=None, is_classifier=False, savedir=None):
    cols, rows = (17,20)
    h,w = (height, width)
    fig = plt.figure(figsize = (40,34))
    
    miss_idx = np.where((preds['classes'] - labels) != 0)[0]
    matched_layers = preds[layer_name][miss_idx]
    if threshold:
        thresh = np.percentile(matched_layers, 90, axis = 1, keepdims=True)
        matched_layers = matched_layers > thresh
    matched_images = 1-preds['input'][miss_idx]
    for i,idx in enumerate(miss_idx):
        img = np.reshape(matched_layers[i], [h,w])
        inp = np.reshape(matched_images[i], [28,28])
        ax = fig.add_subplot(cols, rows, 2*i+1)
        ax.text(0,0.5,labels[idx], color='g')
        ax.text(3,0.5,preds['classes'][idx], color='r')
        ax1 = fig.add_subplot(cols, rows, 2*i+2)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        if threshold:
            sc = ax.imshow(img, vmin=np.min(matched_layers), vmax=np.max(matched_layers), cmap = 'Greys')
        else:
            sc = ax.imshow(img, vmin=np.min(matched_layers), vmax=np.max(matched_layers))
        sc = ax1.imshow(inp, cmap='Greys')
    save_show(savedir, 'misses')


def plot_layer_examples(preds, height, width, layer_name = 'ising_layer', 
        threshold=False, labels=None, is_classifier=False, savedir=None):
    cols, rows = (10, 15)
    h,w = (height, width)
    for cls in range(10):
        matches = np.argwhere(labels == cls).flatten()
        matched_layers = preds[layer_name][matches][:cols*rows] # [10, h x w]
        if threshold:
            #thresh = np.percentile(matched_layers, 90, axis = 0, keepdims=True)
            thresh = np.percentile(matched_layers, 90, axis = 1, keepdims=True)
            matched_layers = matched_layers > thresh
        matched_inputs = 1.0 - preds['input'][matches][:cols*rows]
        pred_classes = preds['classes']
        print(np.where((pred_classes - labels) != 0))
        i = 494
        print(pred_classes[i], labels[i])
        plt.imshow(np.reshape(preds['input'][i], [28,28]))
        plt.show()
        return
        if is_classifier:
            fig = plt.figure(figsize = (20,40))
            cols,rows = (20,10)
            for i in range(cols*rows // 2):
                img = np.reshape(matched_layers[i],[h,w])
                inp = np.reshape(matched_inputs[i],[28,28])
                ax = fig.add_subplot(cols, rows, 2*i+1)
                ax1 = fig.add_subplot(cols, rows, 2*i+2)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax1.get_xaxis().set_visible(False)
                ax1.get_yaxis().set_visible(False)
                if threshold:
                    sc = ax.imshow(img, vmin=np.min(matched_layers), vmax=np.max(matched_layers), cmap = 'Greys')
                else:
                    sc = ax.imshow(img, vmin=np.min(matched_layers), vmax=np.max(matched_layers))
                sc = ax1.imshow(inp, cmap='Greys')
        else:
            fig = plt.figure(figsize = (30,20))
            matched_outputs = preds['prediction'][matches][:cols*rows]
            for i in range(cols*rows // 3):
                img = np.reshape(matched_layers[i],[h,w])
                inp = np.reshape(matched_inputs[i],[28,28])
                outp = np.reshape(matched_outputs[i],[28,28])
                ax = fig.add_subplot(cols, rows, 3*i+1)
                ax1 = fig.add_subplot(cols, rows, 3*i+2)
                ax2 = fig.add_subplot(cols, rows, 3*i+3)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax1.get_xaxis().set_visible(False)
                ax1.get_yaxis().set_visible(False)
                ax2.get_xaxis().set_visible(False)
                ax2.get_yaxis().set_visible(False)
                sc = ax.imshow(img, vmin=np.min(matched_layers), vmax=np.max(matched_layers))
                sc = ax1.imshow(inp, cmap='Greys')
                sc = ax2.imshow(outp, cmap='Greys')
        if threshold:
            save_show(savedir, 'threshold_%d' % cls)
        else:
            save_show(savedir, 'examples_%d' % cls)

def segment_class_average_layer(preds, height, width, layer_name, labels=None, savedir=None):
    h,w = (height, width)
    fig = plt.figure()
    if labels is None:
        labels = preds['classes']
    imgs = []
    for cls in range(10):
        matches = np.argwhere(labels == cls).flatten()
        matched_layers = preds[layer_name][matches]
        cimg = np.reshape(np.nanmean(matched_layers, axis=0),[h, w])
        imgs.append(cimg)
    imgs = np.array(imgs)
    imgs = imgs / np.linalg.norm(imgs, ord = 1, axis = 1, keepdims=True)
    segments = np.argmax(imgs, axis=0)
    ax =fig.add_subplot(1,1,1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    cax = ax.imshow(segments, cmap = 'tab10')
    cbar = fig.colorbar(cax, ticks=range(10))
    cax.set_clim(-0.5, 9.5)
    save_show(savedir, 'segments')




def plot_class_average_layer(preds, height, width,layer_name, labels = None, savedir=None, normalize=False):
    h,w = (height, width)
    fig = plt.figure(figsize = (10,3))
    cols, rows = (2, 5)
    if labels is None:
        labels = preds['classes']
    if normalize: # Normalize each neuron by mean activation across test set
        img = np.reshape(np.mean(preds[layer_name], axis=0), [h,w])
    imgs = []
    stds = []
    for cls in range(10):
        matches = np.argwhere(labels == cls).flatten()
        matched_layers = preds[layer_name][matches]
        cimg = np.reshape(np.nanmean(matched_layers, axis=0),[h, w])
        if normalize:
            cimg -= img
        imgs.append(cimg)
        stds.append(np.reshape(np.nanstd(matched_layers, axis=0), [h,w]))
    imgs = np.array(imgs)
    stds = np.array(stds)
    for cls,(img,std) in enumerate(zip(imgs, stds)):
        ax = fig.add_subplot(cols, rows, cls+1)
        ax.set_title(cls)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if height == 1: # plot curves instead
            img = img.flatten()
            std = std.flatten()
            n = len(img)
            sc = ax.errorbar(np.arange(2*n), np.tile(img,2), yerr=np.tile(std,2), linestyle='-')
            #plt.ylim = (np.min(imgs) - 1, np.max(imgs + 1))
        elif height == 2:
            for i in range(height):
                sc = ax.errorbar(np.arange(2*(img.shape[1])), np.tile(img[i,:],2), yerr=np.tile(std[i,:],2), linestyle='-')
        else: # Colorbar plots
            if normalize:
                sym_max = np.maximum(-np.min(imgs), np.max(imgs))
                sc = ax.imshow(img, vmin=-sym_max, vmax = sym_max)
            else:
                sc = ax.imshow(img, vmin=np.min(imgs), vmax=np.max(imgs))
            #fig.colorbar(sc)
    save_show(savedir, 'norm_averages' if normalize else 'class_averages')

def wishbone_plot_pop_split(preds, data, h, w, layer_name='ising_layer', savedir=None, ythresh=0.5, xthresh=0.5):
    fig = plt.figure(figsize =(10,10))
    cols, rows = (2,2)
    col = data.data.columns
    cd8_loc = col.get_loc('CD8')
    cd4_loc = col.get_loc('CD4')
    outs = preds['prediction']
    outs = outs[:,(cd4_loc,cd8_loc)]
    cd4m = outs[:,0] > 0.5
    cd8m = outs[:,1] > 0.5
    pp = cd4m & cd8m
    pm = cd4m & ~cd8m
    mp = ~cd4m & cd8m
    mm = ~cd4m & ~cd8m
    for i,(mask,name) in enumerate(zip([pp, pm, mp, mm], ['+/+', '+/-', '-/+', '-/-'])):
        ax = fig.add_subplot(cols, rows, i+1)
        ax.set_title(name)
        means = np.reshape(np.nanmean(preds[layer_name][mask], axis=0), [h,w])
        stds  = np.reshape(np.nanstd(preds[layer_name][mask], axis=0), [h,w])
        for j in range(h):
            sc = ax.errorbar(np.arange(2*(means.shape[1])), np.tile(means[j,:],2), yerr=np.tile(stds[j,:],2), linestyle='-')
    save_show(savedir, 'split')

def plot_average_layer(preds, height, width, layer_name='ising_layer', savedir=None):
    fig = plt.figure(figsize = (10,3))
    ax = fig.add_subplot(1,1,1)
    h,w = (height, width)
    img = np.reshape(np.mean(preds[layer_name], axis = 0), [h,w])
    std = np.reshape(np.std(preds[layer_name], axis=0), [h,w])
    if height == 1: # plot curves instead
        img = img.flatten()
        std = std.flatten()
        n = len(img)
        sc = ax.errorbar(np.arange(2*n), np.tile(img,2), yerr=np.tile(std,2), linestyle='-')
        #plt.ylim = (np.min(imgs) - 1, np.max(imgs + 1))
    elif height == 2:
        for i in range(height):
            sc = ax.errorbar(np.arange(2*(img.shape[1])), np.tile(img[i,:],2), yerr=np.tile(std[i,:],2), linestyle='-')
    else:
        sc = plt.imshow(img)
        plt.colorbar(sc)
    save_show(savedir, 'averages')

def plot_embedding(preds, true_labels, savedir = None):
    data = preds['embedding']
    plt.scatter(data[:,0], data[:,1], 
            cmap=plt.get_cmap('tab10', 10), 
            c=true_labels, s=3)
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    save_show(savedir, 'embedding')

def plot_histogram(preds, layer_name='ising_layer', savedir=None):
    data = preds[layer_name]
    plt.hist(data.flatten())
    plt.xlabel('Activation')
    plt.ylabel('Count')
    plt.yticks([])
    save_show(savedir, 'activation_histogram_' + layer_name)

def pca(data):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    loads = pca.fit_transform(data)
    return loads

def plot_pca(data, labels, title = None, fig = None, ax = None): 
    loads = pca(data)
    scatter_plot(loads, labels, title=title, fig = fig, ax = ax, xlabel = 'PCA 1', ylabel = 'PCA2', hide_ticks = True)

def scatter_plot(data, labels, title = None, fig = None, ax = None, xlabel = None, ylabel = None, hide_ticks = False):
    if fig is None or ax is None: 
        fig, ax = plt.subplots(1)
        fig.set_size_inches(10,8)
    sc = ax.scatter(data[:,0], data[:,1], c=labels, s=1)
    x = data[:,0]
    y = data[:,1]
    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=15)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=15)
    if title is not None:
        ax.set_title(title)

def plot_wishbone_output_pca(preds, data, savedir=None):
    fig = plt.figure(figsize = (6,3))
    ins = preds['input']
    outs = preds['prediction']
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2)
    cols = data.data.columns
    cd3_loc = cols.get_loc('CD3')
    plot_pca(ins, ins[:,cd3_loc], fig=fig, ax=ax0)
    plot_pca(outs, outs[:,cd3_loc], fig=fig, ax=ax1)
    save_show(savedir, 'pca')

def plot_cd8_cd4(preds, data, savedir=None):
    fig = plt.figure(figsize = (6,3))
    ins = preds['input']#[:1000]
    outs = preds['prediction']#[:1000]
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2)
    ax0.set_title('Input')
    ax1.set_title('Output')
    ax0.set_xlabel('CH4')
    ax0.set_ylabel('CH8')
    ax1.set_xlabel('CH4')
    ax1.set_ylabel('CH8')
    cols = data.data.columns
    cd8_loc = cols.get_loc('CD8')
    cd4_loc = cols.get_loc('CD4')
    cd3_loc = cols.get_loc('CD3')
    ins = ins[:,(cd4_loc,cd8_loc)]
    outs = outs[:,(cd4_loc,cd8_loc)]
    scatter_plot(ins, preds['input'][:,cd3_loc], fig=fig, ax=ax0)
    ax0.axhline(y=0.5)
    ax0.axvline(x=0.5)
    scatter_plot(outs, preds['prediction'][:,cd3_loc], fig=fig, ax=ax1)
    ax1.axhline(y=0.5)
    ax1.axvline(x=0.5)
    save_show(savedir, '/cd8_cd4')


def plot_output(preds, ishape=(28,28), num_channels=1, savedir=None, gshape =(10,10)):
    fig = plt.figure(figsize = (10,5))
    n = gshape[0] * gshape[1]
    ins  = preds['input'][:n]
    outs = preds['prediction'][:n]
    ins = np_to_img(ins, gshape, ishape, num_channels)
    outs = np_to_img(outs, gshape, ishape, num_channels)
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2)
    ax0.imshow(ins, cmap=plt.get_cmap('gray'), vmin = 0, vmax = 1)
    ax1.imshow(outs, cmap=plt.get_cmap('gray'), vmin = 0, vmax = 1)
    ax0.set_title('Input')
    ax1.set_title('Output')
    ax0.get_xaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax0.get_yaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    save_show(savedir, 'outputs')

def np_to_img(imgs, gshape, ishape, nc=1):
    gh, gw = gshape
    ih, iw = ishape
    h, w   = (gh*ih, gw*iw)
    if nc == 1:
        imgs = np.reshape(imgs, [gh,gw,ih,iw])
        imgs = np.transpose(imgs, [0,1,3,2])
        imgs = np.reshape(imgs, [gh,w,ih])
        imgs = np.transpose(imgs, [0,2,1])
        imgs = np.reshape(imgs, [h,w])
    else:
        imgs = np.reshape(imgs, [gh,gw,ih,iw,nc])
        imgs = np.transpose(imgs, [0,1,3,2,4])
        imgs = np.reshape(imgs, [gh,w,ih,nc])
        imgs = np.transpose(imgs, [0,2,1,3])
        imgs = np.reshape(imgs, [h,w,nc])
    return imgs

def get_input_shape(data_format):
    if data_format == 'channels_first':
        input_shape = [1,28,28]
    elif data_format == 'channels_last':
        input_shape = [28,28,1]
    else:
        raise Exception("Invalid data format")
    return input_shape

def layer_grid_summary(name, var, image_dims, batch_size, grid_dims = [10,10]):
    from magenta.models.image_stylization.image_utils import form_image_grid
    prod = np.prod(image_dims)
    grid = form_image_grid(tf.reshape(var, [batch_size, prod]), grid_dims, image_dims, 1)
    return tf.summary.image(name, grid, max_outputs = 1)

