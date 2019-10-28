class Namespace():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def update(self, other):
        self.__dict__.update(other)
    def __repr__(self):
        return str(self.__dict__)
    def dict(self):
        return self.__dict__
    def copy(self):
        return Namespace(**self.__dict__)

default_args = {
    'add_labels_to_ising'   : False,
    'add_noise'             : False,
    'add_input_noise'       : 0,
    'batch_regularization'  : 0, #1e-6,
    'batch_size'            : 500,
    'conv_decoder'          : False,
    'conv_autoencoder'      : False, 
    'data_dir'              : '/tmp/mnist_data',
    'data_format'           : 'channels_last', # Faster on GPU
    'dataset'               : 'wishbone',
    'decay'                 : 1,
    'early_stopping_thresh' : 0.,
    'epochs_between_evals'  : 10,
    'export_dir'            : './exported',
    'heat_reg'              : 0.,
    'heat_kernel_path'      : False,
    'heat_kernel_name'      : False,
    'height'                : 5,
    'ising_graph_type'      : 'rings',
    'ising_regularization'  : 0.,
    'l1_reg'                : 0.,
    'l2_reg'                : 0.,
    'l1_hinge_reg'          : 0.,
    'laplacian_type'        : 'unnormalized',
    'layer_num'             : 3,
    'learning_rate'         : 1e-4,
    'middle_width'          : 5,
    'model_dir'             : './model_dir',
    # Boolean to normalize ising penalty as (x^t L x) / (x^t x) if True or (x^t L x) otherwise.
    'normalize_ising_reg'   : False,
    'predict'               : False,
    'predict_dir'           : False,
    'relu_embedding'        : False,
    'softmax_embedding'     : False, 
    'train_and_predict'     : False,
    'train_epochs'          : 40,
    'use_capsule'           : False,
    'use_classifier'        : False,
    'use_regressor'         : False,
    'warm_start_encoder'    : False,
    'width'                 : 10,
}

def get_default_args():
    return Namespace(**default_args)

def build_args_from_stack(arg_stack):
    args = get_default_args()
    for a in reversed(arg_stack): args.update(a)
    return args
