"""
Parameter Exploration

Creates json parameter files which overwrite default parameters
"""
import args
import argparse
import json
import dataset as ds

args = args.get_default_args()

def explore_layer_shape():
    ising           = [1e-3, 1e-2, 1e-1, 0]
    middle_width    = [3]
    height          = [1,2]
    model_dir      = 'explore_shape/teapot/'
    graph_types     = ['rings', 'lines', 'torus']
    
    runs = []
    count = -1
    for i in ising:
        for mw in middle_width:
            for h in height:
                for gtype in graph_types:
                    count += 1
                    name = 'i%1.0e_mw%d_h%d_%s_%02d' % (i,mw,h,gtype,count)
                    runs.append({
                        'ising_regularization' : i,
                        'model_dir'            : model_dir + name,
                        'height'               : h,
                        'width'                : 10,
                        'batch_regularization' : 1e-5,
                        'middle_width'         : mw,
                        'learning_rate'        : 1e-4,
                        'dataset'              : 'teapot',
                        'train_and_predict'    : True,
                        'ising_graph_type'     : gtype,
                    })
    print(json.dumps(runs, indent=4, sort_keys = True))

def explore_lr():
    ising           = [1e-3]
    middle_width    = [2, 3]
    learning_rates  = [1e-4, 1e-5]
    batch_reg       = [1e-5]
    width           = [5,10,20]
    height          = [1,2,3,5]
    model_dir      = './auto/'
    runs = []
    count = -1
    for i in ising:
        for b in batch_reg:
            for h in height:
                for w in width:
                    for mw in middle_width:
                        for l in learning_rates:
                            count += 1
                            name = 'i%1.0e_b%1.0e_mw%d_h%d_w%d_lr%1.0e_%02d' % (i, b, mw, h, w, l, count)
                            runs.append({
                                'ising_regularization' : i,
                                'model_dir'            : model_dir + name,
                                'height'               : h,
                                'width'                : w,
                                'batch_regularization' : b,
                                'middle_width'         : mw,
                                'learning_rate'        : l,
#                                'train_and_predict'    : True,
                            })
    print(json.dumps(runs, indent=4, sort_keys = True))

def explore_ising_shape(dataset, ising_layer_num = None):
    shapes = ['grid', 'torus', 'rings', 'lines', 'wishbone']
    width  = [3, 5, 10]
    runs = []
    count = -1
    model_dir      = ('./explore_shape_layer_v3') + dataset + '/'
    for s in shapes:
        for w in width:
            count += 1
            name = '%s_%d_%d' % (s, w, count)
            runs.append({
                'ising_regularization'  : 1e-3,
                'model_dir'             : model_dir + name,
                'height'                : 2,
                'width'                 : w,
                'batch_regularization'  : 1e-5,
                'middle_width'          : 3,
                'ising_graph_type'      : s,
                'dataset'               : dataset,
            })
    shapes2 = ['star', 'kite']
    for s in shapes2:
        for h in [1,2]:
            for w in [5,10]:
                if w !=10 and s == 'kite': continue
                count += 1
                name = '%s_h%d_w%d_%d' % (s, h, w, count)
                runs.append({
                    'ising_regularization'  : 1e-3,
                    'model_dir'             : model_dir + name,
                    'height'                : h,
                    'width'                 : w,
                    'batch_regularization'  : 1e-5,
                    'middle_width'          : 3,
                    'ising_graph_type'      : s,
                    'dataset'               : dataset,
                })
    shapes3 = ['tree', 'lines', 'rings']
    for s in shapes3:
        for h in [1,2]:
            for w in [7,15]:
                count += 1
                name = '%s_h%d_w%d_%d' % (s, h, w, count)
                runs.append({
                    'ising_regularization'  : 1e-3,
                    'model_dir'             : model_dir + name,
                    'height'                : h,
                    'width'                 : w,
                    'batch_regularization'  : 1e-5,
                    'middle_width'          : 3,
                    'ising_graph_type'      : s,
                    'dataset'               : dataset,
                })
    for s in ['no_ising']:
        for h in [1]:
            for w in [3,5,7,10,15]:
                count += 1
                name = '%s_h%d_w%d_%d' % (s, h, w, count)
                runs.append({
                    'ising_regularization'  : 0,
                    'model_dir'             : model_dir + name,
                    'height'                : h,
                    'width'                 : w,
                    'batch_regularization'  : 0,
                    'middle_width'          : 3,
                    'dataset'               : dataset,
                })
    if ising_layer_num is not None:
        for r in runs:
            r['layer_num'] = ising_layer_num
        #runs = [r.update({'layer_num':ising_layer_num}) for r in runs]
    print(json.dumps(runs, indent=4, sort_keys = True))

def explore_training_time(dataset):
    train_epochs = 40 # Default value
    for te in range(train_epochs, train_epochs * 21, 3*train_epochs):
        print(te)

def explore_linear():
    runs = []
    count = -1
    model_dir      = 'paper_output/0/linear/'
    ising_weights = [0, 0.1, 0.01, 0.001, 0.0001]
    for i in ising_weights:
        #for br in [0, 1e-5]:
        for br in [0]:
            count += 1
            #name = 'i%1.0e_br%1.0e_%02d' % (i,br,count)
            name = 'i%1.0e_%02d' % (i,count)
            runs.append({
                'ising_regularization'  : i,
                'model_dir'             : model_dir + name,
                'height'                : 1,
                'width'                 : 10,
                'batch_regularization'  : br,
                'middle_width'          : 4,
                'dataset'               : 'linear',
                'train_and_predict'     : True,
                'train_epochs'          : 200,
                'decay'                 : 0.99,
                'ising_graph_type'      : 'lines',
                'laplacian_type'        : 'unnormalized',
                'layer_num'             : 2,
            })
    print(json.dumps(runs, indent=4, sort_keys = True))

def explore_teapot():
    runs = []
    count = -1
    model_dir      = 'paper_output/teapot/'
    ising_weights = [0, 0.1, 0.01, 0.001, 0.0001]
    for i in ising_weights:
        for br in [0, 1e-5]:
            count += 1
            name = 'i%1.0e_br%1.0e_%02d' % (i,br,count)
            runs.append({
                'ising_regularization'  : i,
                'model_dir'             : model_dir + name,
                'height'                : 1,
                'width'                 : 10,
                'batch_regularization'  : br,
                'middle_width'          : 3,
                'dataset'               : 'teapot',
                'train_and_predict'     : True,
                'train_epochs'          : 5000,
                'decay'                 : 0.99,
                'ising_graph_type'      : 'rings',
            })
    print(json.dumps(runs, indent=4, sort_keys = True))

def explore_hubspoke():
    runs = []
    count = -1
    model_dir      = 'paper_output/hubspoke/'
    ising_weights = [0, 0.1, 0.01, 0.001, 0.0001]
    for i in ising_weights:
        for shape in ['star', 'rings']:
            if i == 0 and shape == 'rings':
                continue
            for br in [0, 1e-5]:
                count += 1
                name = 'i%1.0e_%s_br%1.0e_%02d' % (i,shape, br,count)
                runs.append({
                    'ising_regularization'  : i,
                    'model_dir'             : model_dir + name,
                    'height'                : 1,
                    'width'                 : 10,
                    'batch_regularization'  : br,
                    'middle_width'          : 10,
                    'dataset'               : 'hub',
                    'train_and_predict'     : True,
#                    'decay'                 : 0.99,
                    'ising_graph_type'      : shape,
                    'layer_num'             : 2,
                    'train_epochs'          : 500,
                })
    print(json.dumps(runs, indent=4, sort_keys = True))

def explore_mnist_convolutional_decoder():
    runs = []
    count = -1
    model_dir      = 'conv_mnist/1/'
    ising_weights = [0, 0.1, 0.01, 0.001, 0.0001]
    for i in ising_weights:
        for br in [0, 1e-5]:
            for cd in [False, True]:
                count += 1
                name = 'i%1.0e_br%1.0e_%02d' % (i, br,count)
                if cd: name = 'conv_' + name
                runs.append({
                    'ising_regularization'  : i,
                    'model_dir'             : model_dir + name,
                    'height'                : 8,
                    'width'                 : 8,
                    'batch_regularization'  : br,
                    'middle_width'          : 10,
                    'dataset'               : 'mnist',
                    'conv_decoder'          : cd,
     #               'train_and_predict'     : True,
     #                   'decay'                 : 0.99,
                    'ising_graph_type'      : 'grid',
                    'train_epochs'          : 200,
                })
    print(json.dumps(runs, indent=4, sort_keys = True))

def create_warm_start_convolutional_decoder():
    runs = []
    count = -1
    model_dir      = 'warm/1/'
    ising_weights = [0, 0.1, 0.01, 0.001, 0.0001]
    for i in ising_weights:
        for br in [1e-5]:
            for cd in [False, True]:
                count += 1
                name = 'i%1.0e_br%1.0e_%02d' % (i, br,count) 
                if cd: name = 'conv_' + name
                runs.append({
                    'ising_regularization'  : i,
                    'model_dir'             : model_dir + name,
                    'height'                : 8,
                    'width'                 : 8,
                    'batch_regularization'  : br,
                    'middle_width'          : 10,
                    'dataset'               : 'mnist',
                    'conv_decoder'          : cd,
                    'warm_start_encoder'    : model_dir + prev_name if cd else False,
     #               'train_and_predict'     : True,
     #                   'decay'                 : 0.99,
                    'ising_graph_type'      : 'grid',
                    'train_epochs'          : 100,
                })
                prev_name = name
    print(json.dumps(runs, indent=4, sort_keys = True))

def explore_transfer():
    runs = []
    count = -1
    model_dir      = 'cifar_trans/00/'
    ising_weights = [0, 0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    for i in ising_weights:
        for br in [1e-5]:
            for cd in [False, True]:
                count += 1
                name = 'i%1.0e_br%1.0e_%02d' % (i, br,count) 
#                prev_name=name
                if cd: name = 'trans_' + name
                te = 50 #if cd else 2000
                runs.append({
                    'ising_regularization'  : 0 if cd else i,
                    'model_dir'             : model_dir + name,
                    'height'                : 8,
                    'width'                 : 8,
                    'batch_regularization'  : br,
                    'batch_size'            : 50,
                    'epochs_between_evals'  : 5,
                    'middle_width'          : 15,
                    'dataset'               : 'cifar10',
                    'learning_rate'         : 1e-4,
#                    'conv_decoder'          : cd,
                    'warm_start_encoder'    : model_dir + prev_name if cd else False,
                    'use_classifier'        : True,
     #               'train_and_predict'     : True,
     #                   'decay'                 : 0.99,
                    'ising_graph_type'      : 'grid',
                    'train_epochs'          : te,
#                    'early_stopping_thresh' : 0 if cd else 0.15,
                })
                prev_name=name
    print(json.dumps(runs, indent=4, sort_keys = True))

def explore_generalizability():
    """ Explore the generalizability of autoencoder using ising.
    This is in comparison to L1 and L2 regularization. Hopefully we can
    show similar properties and additional ones and argue that this is
    helpful in certain other circumstances.
    """
    runs = []
    count = -1
    model_dir      = 'general/0/'
    ising_weights = [0, 0.1, 0.01, 0.001, 0.0001]
    for i in ising_weights:
        for l2 in [0, 0.001, 0.0001]:
            count += 1
            name = 'i%1.0e_l2%1.0e_%02d' % (i, l2,count) 
#                prev_name=name
            #if cd: name = 'trans_' + name
            te = 100 #if cd else 2000
            runs.append({
                'ising_regularization'  : i,
                'model_dir'             : model_dir + name,
                'height'                : 8,
                'width'                 : 8,
                'batch_regularization'  : 0.,
                'l2_reg'                : l2,
                'batch_size'            : 50,
                'epochs_between_evals'  : 10,
                'middle_width'          : 3,
                'dataset'               : 'keras_mnist',
                'learning_rate'         : 1e-4,
#                    'conv_decoder'          : cd,
                'warm_start_encoder'    : False, #model_dir + prev_name if cd else False,
                'use_classifier'        : False,
 #               'train_and_predict'     : True,
 #                   'decay'                 : 0.99,
                'ising_graph_type'      : 'grid',
                'train_epochs'          : te,
#                    'early_stopping_thresh' : 0 if cd else 0.15,
            })
            prev_name=name
    print(json.dumps(runs, indent=4, sort_keys = True))

def explore_logistic_regression():
    runs = []
    count = -1
    model_dir      = 'logistic/0/'
    ising_weights = [0, 1000, 100, 10]
    for i in ising_weights:
        for l1 in [0, 0.1, 0.01]:
            for normalize in [False, True]:
                count += 1
                name = 'i%1.0e_l1%1.0e_%02d' % (i, l1, count) 
                if normalize: name = 'norm_' + name
                te = 1000 #if cd else 2000
                runs.append({
                    'ising_regularization'  : i,
                    'model_dir'             : model_dir + name,
                    'batch_regularization'  : 0.,
                    'l1_reg'                : l1,
                    'batch_size'            : 10,
                    'epochs_between_evals'  : 200,
                    'dataset'               : 'gene_synth',
                    'normalize_ising_reg'   : normalize,
                    'learning_rate'         : 0.1,
                    'decay'                 : 0.999,
                    'use_classifier'        : True,
                    'train_epochs'          : te,
                })
    print(json.dumps(runs, indent=4, sort_keys = True))
def explore_hats():
    runs = []
    count = -1
    model_dir      = 'hats/13/'
    for i in ['backwardshat', 'backwardsmeyer', 'hat1back', 'hat2back', 'hat3back', 'hat4back', 'meyers1back', 'meyers2back', 'meyers3back', 'meyers4back']:
        for hreg in [1e-6]:
            count += 1
            name = '%s_%1.0e_%02d' % (i, hreg, count) 
            runs.append({
                'heat_reg': hreg,
                'dataset': 'teapot',
                'softmax_embedding': True,
                'heat_kernel_path': 'wavelets_full.mat',
                'heat_kernel_name': i,
                'height': 1,
                "ising_graph_type": "rings",
                "layer_num": 2,
                "model_dir": model_dir + name,
                "width": 20,
                "train_epochs": 2000,
                "train_and_predict": True,
            })
    print(json.dumps(runs, indent=4, sort_keys = True))

def explore_mnist_convolutional_decoder2():
    runs = []
    count = -1
    model_dir      = '/data/atong/IsingAE/mnist/1/'
    ising_weights = [0, 0.1, 0.01, 0.001, 0.0001]
    for j in range(10):
        for i in ising_weights:
            for br in [0, 1e-5]:
                count += 1
                name = 'i%1.0e_br%1.0e_%02d' % (i, br,count)
                runs.append({
                    'ising_regularization'  : i,
                    'model_dir'             : model_dir + name,
                    'height'                : 8,
                    'width'                 : 8,
                    'batch_regularization'  : br,
                    'middle_width'          : 4,
                    'dataset'               : 'mnist',
                    'conv_decoder'          : True,
     #               'train_and_predict'     : True,
     #                   'decay'                 : 0.99,
                    'ising_graph_type'      : 'grid',
                    'train_epochs'          : 2000,
                })
    print(json.dumps(runs, indent=4, sort_keys = True))

def explore_hats2():
    runs = []
    count = -1
    model_dir      = 'hats/13/'
    for i in ['hat2back', 'hat3back', 'meyers2back', 'meyers3back','backwardsmeyer']:
        for hreg in [1e-6]:
            count += 1
            name = '%s_%1.0e_%02d' % (i, hreg, count) 
            runs.append({
                'heat_reg': hreg,
                'dataset': 'teapot',
                'softmax_embedding': True,
                'heat_kernel_path': 'wavelets_full.mat',
                'heat_kernel_name': i,
                'height': 1,
                "ising_graph_type": "rings",
                "layer_num": 2,
                "model_dir": model_dir + name,
                "width": 20,
                "train_epochs": 2000,
                "train_and_predict": True,
            })
    print(json.dumps(runs, indent=4, sort_keys = True))

def explore_hats3():
    runs = []
    count = -1
    model_dir      = 'hats/mnist/01/'
    for i in ['hat2back', 'hat3back', 'meyers2back', 'meyers3back','backwardsmeyer']:
        for hreg in [1e-6]:
            count += 1
            name = '%s_%1.0e_%02d' % (i, hreg, count) 
            runs.append({
                'heat_reg': hreg,
                'dataset': 'mnist',
                'softmax_embedding': True,
                'heat_kernel_path': 'wavelets_full.mat',
                'heat_kernel_name': i,
                'height': 1,
                "ising_graph_type": "rings",
                "layer_num": 2,
                "model_dir": model_dir + name,
                "width": 20,
                "train_and_predict": True,
            })
    print(json.dumps(runs, indent=4, sort_keys = True))

def explore_hubs_100():
    runs = []
    count = -1
    model_dir      = 'hubs/gen/1/'
    for l1_reg in [1e-3, 1e-4]:
        for i in range(10):
            count += 1
            name = '%1.0e_%02d' % (l1_reg, count) 
            runs.append({
                "dataset": "hub_perturb",
                "height": 3,
                "ising_regularization": 0.1,
                "l1_reg": l1_reg,
                "ising_graph_type": "lines",
                "layer_num": 2,
                "model_dir": model_dir + name,
                "train_and_predict": True,
                "learning_rate": 1e-4,
                "width": 2
            })
    print(json.dumps(runs, indent=4, sort_keys = True))


def main():
#    explore_hubspoke()
#    explore_mnist_convolutional_decoder2()
    #explore_mnist_convolutional_decoder()
#    explore_logistic_regression()
#    explore_transfer()
#    explore_generalizability()
    #create_warm_start_convolutional_decoder()
#    explore_teapot()
     #explore_linear()
#    explore_layer_shape()
#    explore_ising_shape('wishbone')
#    explore_ising_shape('mnist', ising_layer_num=None)
#    explore_training_time('mnist')
    #explore_hats3()
    explore_hubs_100()

if __name__ == '__main__':
    main()

