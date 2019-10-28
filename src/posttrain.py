import tensorflow as tf
import numpy as np
import util
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

"""
Ising analysis post training
"""

def corrolate_layer(preds, data, layer_name='ising_layer', savedir=None):
    def get_reordered(show=True):
        outs = pd.DataFrame(preds[layer_name])
        g = sns.clustermap(outs.corr(),
                    vmin = -1, vmax = 1,
                    cmap = "RdBu_r")
        util.save_show(savedir, 'neuron_corr_' + layer_name)
        return g.dendrogram_row.reordered_ind
    order = get_reordered()

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
            col_cluster=True,
            yticklabels=False,
    )
    g.ax_heatmap.set_title('Ising Layer Wishbone Heatmap')
    save_show(savedir, 'heatmap_' + layer_name)
    


