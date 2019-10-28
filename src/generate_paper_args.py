import args
import argparse
import json
import dataset as ds
import itertools

args = args.get_default_args()

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
    return runs

def explore_teapot():
    runs = []
    count = -1
    model_dir      = 'paper_output/0/teapot/'
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
    return runs

def explore_teapot_v2():
    runs = []
    count = -1
    model_dir      = 'paper_output/0/teapot_v2/'
    ising_weights = [0, 0.1, 0.01, 0.001, 0.0001]
    for i in ising_weights:
        for br in [0]:
            count += 1
            name = 'i%1.0e_br%1.0e_%02d' % (i,br,count)
            runs.append({
                'ising_regularization'  : i,
                'model_dir'             : model_dir + name,
                'height'                : 1,
                'width'                 : 20,
                'batch_regularization'  : br,
                'middle_width'          : 20,
                'dataset'               : 'teapot',
                #'train_and_predict'     : True,
                'train_epochs'          : 5000,
                #'decay'                 : 0.99,
                'ising_graph_type'      : 'rings',
                'layer_num'             : 2,
            })
    return runs

def explore_hubspoke():
    runs = []
    count = -1
    model_dir      = 'paper_output/0/hubspoke/'
    ising_weights = [0, 0.1, 0.01, 0.001, 0.0001]
    for i in ising_weights:
        for shape in ['lines']:
            for br in [0]:
                count += 1
                name = 'i%1.0e_%s_br%1.0e_%02d' % (i,shape, br,count)
                runs.append({
                    'ising_regularization'  : i,
                    'model_dir'             : model_dir + name,
                    'height'                : 3,
                    'width'                 : 2,
                    'batch_regularization'  : br,
                    'middle_width'          : 6,
                    'dataset'               : 'hub',
                    'train_and_predict'     : True,
#                    'decay'                 : 0.99,
                    'ising_graph_type'      : shape,
                    'layer_num'             : 2,
                    'train_epochs'          : 500,
                })
    return runs

def explore_mnist_2d_vis():
    runs = []
    count = -1
    model_dir      = 'paper_output/0/mnist_vis/'
    ising_weights = [0, 0.1, 0.01, 0.001, 0.0001]
    for i in ising_weights:
        for shape in ['grid']:
            for br in [0, 1e-5]:
                count += 1
                name = 'i%1.0e_%s_br%1.0e_%02d' % (i,shape, br,count)
                runs.append({
                    'ising_regularization'  : i,
                    'model_dir'             : model_dir + name,
                    'height'                : 5,
                    'width'                 : 10,
                    'batch_regularization'  : br,
                    'middle_width'          : 10,
                    'dataset'               : 'keras_mnist',
#                    'decay'                 : 0.99,
                    'ising_graph_type'      : shape,
                    'layer_num'             : 3,
                    'use_classifier'        : True,
                    #'train_epochs'          : 500,
                })
    return runs

def explore_mnist_2d_vis_v2():
    """
    There is a clear signal on classifier lets try the last layer of an autoencoder then
    """
    runs = []
    count = -1
    model_dir      = 'paper_output/0/mnist_vis/'
    ising_weights = [0, 0.1, 0.01, 0.001, 0.0001]
    for i in ising_weights:
        for shape in ['grid']:
            for br in [0, 1e-5]:
                count += 1
                name = 'i%1.0e_%s_br%1.0e_%02d' % (i,shape, br,count)
                runs.append({
                    'ising_regularization'  : i,
                    'model_dir'             : model_dir + name,
                    'height'                : 5,
                    'width'                 : 10,
                    'batch_regularization'  : br,
                    'middle_width'          : 10,
                    'dataset'               : 'mnist',
#                    'decay'                 : 0.99,
                    'ising_graph_type'      : shape,
                    'layer_num'             : 4,
                    #'use_classifier'        : True,
                    #'train_epochs'          : 500,
                })
    return runs

def explore_mnist_2d_vis_v3():
    """
    Ok worked for layer 4 trying layer 3
    """
    runs = []
    count = -1
    model_dir      = 'paper_output/0/mnist_vis/'
    ising_weights = [0, 0.1, 0.01, 0.001, 0.0001]
    for i in ising_weights:
        for shape in ['grid']:
            for br in [0, 1e-5]:
                count += 1
                name = 'i%1.0e_%s_br%1.0e_%02d' % (i,shape, br,count)
                runs.append({
                    'ising_regularization'  : i,
                    'model_dir'             : model_dir + name,
                    'height'                : 5,
                    'width'                 : 10,
                    'batch_regularization'  : br,
                    'middle_width'          : 10,
                    'dataset'               : 'mnist',
#                    'decay'                 : 0.99,
                    'ising_graph_type'      : shape,
                    'layer_num'             : 3,
                    #'use_classifier'        : True,
                    #'train_epochs'          : 500,
                })
    return runs

def explore_mnist_2d_vis_v4():
    """
    ok convolutional classifier on ising
    """
    runs = []
    count = -1
    model_dir      = 'paper_output/0/mnist_vis/'
    ising_weights = [0, 0.1, 0.01, 0.001, 0.0001]
    for i in ising_weights:
        for shape in ['grid']:
            for br in [0]:
                count += 1
                name = 'i%1.0e_%s_br%1.0e_%02d' % (i,shape, br,count)
                runs.append({
                    'ising_regularization'  : i,
                    'model_dir'             : model_dir + name,
                    'height'                : 8,
                    'width'                 : 8,
                    'batch_regularization'  : br,
                    'middle_width'          : 10,
                    'dataset'               : 'mnist',
#                    'decay'                 : 0.99,
                    'ising_graph_type'      : shape,
                    'layer_num'             : 3,
                    'use_classifier'        : True,
                    #'train_epochs'          : 500,
                    'conv_decoder'          : True,
                })
    return runs

def explore_mnist_2d_vis_v5():
    """
    ok convolutional classifier on ising
    """
    runs = []
    count = -1
    model_dir      = 'paper_output/0/mnist_vis_no_conv/'
    ising_weights = [0, 0.1, 0.01, 0.001, 0.0001]
    for i in ising_weights:
        for shape in ['grid']:
            for br in [0]:
                count += 1
                name = 'i%1.0e_%s_br%1.0e_%02d' % (i,shape, br,count)
                runs.append({
                    'ising_regularization'  : i,
                    'model_dir'             : model_dir + name,
                    'height'                : 8,
                    'width'                 : 8,
                    'batch_regularization'  : br,
                    'middle_width'          : 10,
                    'dataset'               : 'mnist',
#                    'decay'                 : 0.99,
                    'ising_graph_type'      : shape,
                    'layer_num'             : 3,
                    'use_classifier'        : True,
                    #'train_epochs'          : 500,
                    #'conv_decoder'          : True,
                })
    return runs

def explore_mnist_2d_vis_v6():
    """
    Autoencoder but 8x8
    """
    runs = []
    count = -1
    model_dir      = 'paper_output/0/mnist_vis_auto/'
    ising_weights = [0, 0.1, 0.01, 0.001, 0.0001]
    for i in ising_weights:
        for shape in ['grid']:
            for br in [0]:
                count += 1
                name = 'i%1.0e_%s_br%1.0e_%02d' % (i,shape, br,count)
                runs.append({
                    'ising_regularization'  : i,
                    'model_dir'             : model_dir + name,
                    'height'                : 8,
                    'width'                 : 8,
                    'batch_regularization'  : br,
                    'middle_width'          : 10,
                    'dataset'               : 'mnist',
#                    'decay'                 : 0.99,
                    'ising_graph_type'      : shape,
                    'layer_num'             : 4,
                    #'use_classifier'        : True,
                    #'train_epochs'          : 500,
                    #'conv_decoder'          : True,
                })
    return runs

def explore_mnist_2d_vis_v7():
    """
    Autoencoder but 8x8
    """
    runs = []
    count = -1
    model_dir      = 'paper_output/0/mnist_vis_auto_l3/'
    ising_weights = [0, 0.1, 0.01, 0.001, 0.0001]
    for i in ising_weights:
        for shape in ['grid']:
            for br in [0]:
                count += 1
                name = 'i%1.0e_%s_br%1.0e_%02d' % (i,shape, br,count)
                runs.append({
                    'ising_regularization'  : i,
                    'model_dir'             : model_dir + name,
                    'height'                : 8,
                    'width'                 : 8,
                    'batch_regularization'  : br,
                    'middle_width'          : 10,
                    'dataset'               : 'mnist',
#                    'decay'                 : 0.99,
                    'ising_graph_type'      : shape,
                    'layer_num'             : 3,
                    #'use_classifier'        : True,
                    #'train_epochs'          : 500,
                    #'conv_decoder'          : True,
                })
    return runs

def explore_mnist_2d_vis_v8():
    """
    Autoencoder but more very small ising weight values
    """
    runs = []
    count = -1
    model_dir      = 'paper_output/0/mnist_vis_auto_l4/'
    ising_weights = [0, 0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 1e-7]
    for i in ising_weights:
        for shape in ['grid']:
            for br in [0]:
                count += 1
                name = 'i%1.0e_%s_br%1.0e_%02d' % (i,shape, br,count)
                runs.append({
                    'ising_regularization'  : i,
                    'model_dir'             : model_dir + name,
                    'height'                : 8,
                    'width'                 : 8,
                    'batch_regularization'  : br,
                    'middle_width'          : 10,
                    'dataset'               : 'mnist',
#                    'decay'                 : 0.99,
                    'ising_graph_type'      : shape,
                    'layer_num'             : 4,
                    #'use_classifier'        : True,
                    #'train_epochs'          : 500,
                    #'conv_decoder'          : True,
                })
    return runs

def explore_wishbone():
    runs = []
    count = -1
    model_dir      = 'paper_output/0/wishbone/'
    ising_weights = [0, 0.1, 0.01, 0.001, 0.0001]
    for i in ising_weights:
        for shape in ['lines']:
            for br in [0]:
                count += 1
                name = 'i%1.0e_%s_br%1.0e_%02d' % (i,shape, br,count)
                runs.append({
                    'ising_regularization'  : i,
                    'model_dir'             : model_dir + name,
                    'height'                : 1,
                    'width'                 : 10,
                    'batch_regularization'  : br,
                    'middle_width'          : 10,
                    'dataset'               : 'wishbone',
#                    'decay'                 : 0.99,
                    'ising_graph_type'      : shape,
                    'layer_num'             : 2,
#                    'use_classifier'        : True,
                    #'train_epochs'          : 500,
                    #'conv_decoder'          : True,
                })
    return runs

def explore_blend():
    runs = []
    count = -1
    model_dir      = 'paper_output/0/blend/'
    ising_weights = [0, 0.1, 0.01, 0.001, 0.0001]
    for j in range(3):
        for i in ising_weights:
            for shape in ['lines']:
                for br in [0]:
                    count += 1
                    name = 'i%1.0e_%s_br%1.0e_%02d' % (i,shape, br,count)
                    runs.append({
                        'ising_regularization'  : i,
                        'model_dir'             : model_dir + name,
                        'height'                : 4,
                        'width'                 : 4,
                        'batch_regularization'  : br,
                        'middle_width'          : 10,
                        'dataset'               : 'wishbone',
#                    'decay'                 : 0.99,
                        'ising_graph_type'      : shape,
                        'layer_num'             : 2,
                        'l1_reg'                : 1e-5
#                    'use_classifier'        : True,
                        #'train_epochs'          : 500,
                        #'conv_decoder'          : True,
                    })
    return runs





def main():
#    runs =  list(itertools.chain(
#                explore_linear(),
#                explore_hubspoke(),
#                explore_teapot(),
#                explore_mnist_2d_vis(),
#            ))
    runs =  list(itertools.chain(
                #explore_mnist_2d_vis_v8(),
                #explore_teapot_v2(),
                explore_blend(),
                #explore_hubspoke(),
                #explore_wishbone()
            ))
    print(json.dumps(runs, indent=4, sort_keys = True))

if __name__ == '__main__':
    main()

