from dataset import Dataset
import util
import classifier
#import capsule
import autoencoder
from args import get_default_args, build_args_from_stack

import argparse
from functools import partial
import numpy as np
import tensorflow as tf
from tensorflow.python.training import session_run_hook
import time
import os
import json
import pickle

_TENSORS_TO_LOG = dict((x, x) for x in ['learning_rate',
                                        'mse_loss',
                                        'ising_weight',
                                        'ising_loss',
                                        'heat_loss',
#                                        'l1_loss',
#                                        'l1_hinge_loss',
#                                        'celoss',
                                        'batch_loss'])
def get_model_fn(flags):
    model_fn = autoencoder.model_fn
    if flags.use_classifier:
        model_fn = classifier.model_fn 
    if flags.use_regressor:
        model_fn = classifier.regressor
    #if flags.use_capsule:
    #    model_fn = capsule.model_fn
    #model_fn = classifier.baseline_model_fn
    return model_fn

def get_model(data, flags, use_gpu = True):
    if use_gpu:
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3))
    else:
        config = tf.ConfigProto(device_count = {'GPU':0})

#    print('WARNING USING BASE MODEL')
#    return tf.estimator.BaselineClassifier(
#        model_dir = flags.model_dir,
#        config = tf.estimator.RunConfig(session_config=config)
#    )
    tf.estimator.LinearClassifier(
        feature_columns = [tf.feature_column.numeric_column(str(i)) for i in range(100)],
        model_dir = flags.model_dir,
        config = tf.estimator.RunConfig(session_config=config),
        optimizer = tf.train.ProximalAdagradOptimizer(
            learning_rate = 0.1,
            #l1_regularization_strength=0.001,
            #l2_regularization_strength=0.001,
        ),
    )
    return tf.estimator.Estimator(
        model_fn = get_model_fn(flags),
        model_dir = flags.model_dir,
        params = {'flags' : flags, 'input_width' : data.input_width(), 'network' : data.get_network()},
        config = tf.estimator.RunConfig(session_config=config)
    )
    
def predict(data, flags):
    def calculate_preds():
        return util.process_predictions(get_model(data, flags).predict(
                input_fn=lambda:data.get_predict_input(), yield_single_examples=False))
    ppath = flags.model_dir + '/preds'
    if os.path.exists(ppath):
        preds = pickle.load(open(ppath, 'rb'))
    else:
        preds = calculate_preds()
        pickle.dump(preds, open(ppath, 'wb'))
    data.predict(preds)

def train(data, flags):
    #print('NO LOGGING FOR BASELINE')
    train_hooks = [
        tf.train.LoggingTensorHook(tensors =_TENSORS_TO_LOG, every_n_iter=1000), 
        #EarlyStoppingHook(thresh=flags.early_stopping_thresh),
        #TestHook(),
    ]
    model = get_model(data, flags)
    # Training Loop
    for _ in range(flags.train_epochs // flags.epochs_between_evals):
        model.train(input_fn=lambda:data.get_train_input(), hooks=train_hooks)
        eval_results = model.evaluate(input_fn=lambda:data.get_eval_input())
        print('Evaluation results:\n\t%s\n' % eval_results)
        if not flags.use_classifier and not flags.use_regressor and eval_results['rmse'] < flags.early_stopping_thresh:
            print('Stopping early, rmse goal reached')
            break


def run(arg_stack):
    # args are overriden [default, dataset_specific, parsed_from_file, command_line]
    args = build_args_from_stack(arg_stack)
    data = Dataset.factory(args.dataset)(args)
    arg_stack.append(data.get_dataset_default_flags())
    args = build_args_from_stack(arg_stack)
    print('======================')
    print(arg_stack)
    data.flags = args
    print('======================')
    print(json.dumps(args.dict(), indent=4, sort_keys=True))

    if args.predict_dir:
        print('Running predict recursively')
        import subprocess
        for root, dirs, files in os.walk(args.predict_dir):
            if 'args' in files:
                arg_path = os.path.join(root, 'args')
                process = subprocess.Popen(['python', 'main.py', '-a', arg_path, '-p'])
                process.wait()
        return

    if args.predict:
        print('Running predict')
        predict(data, args)
    else:
        print('Running train')
        os.makedirs(os.path.dirname(args.model_dir + '/'), exist_ok=True)
        with open(args.model_dir + '/args', 'w+') as f:
            f.write(json.dumps(args.dict(), indent=4, sort_keys=True))
        train(data, args)
    if args.train_and_predict:
        print('Running predict after train')
        data.flags.__dict__['predict'] = 'all'
        print(json.dumps(data.flags.dict(), indent=4, sort_keys=True))
        predict(data, args)

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--args_from_file', help='Load arguments from json')
    parser.add_argument('--model_dir')
    parser.add_argument('-i', '--ising_regularization', type=float, 
        help='ising regularization')
    parser.add_argument('-p', '--predict', nargs='?', const='all')
    parser.add_argument('--height', type=int)
    parser.add_argument('--width', type=int)
    parser.add_argument('--batch_regularization', type=float)
    parser.add_argument('--use_classifier', action='store_true')
    parser.add_argument('--learning_rate', type=float)
    # Only plumbed for autoencoder
    parser.add_argument('--decay', type=float, help='set the learning rate decay')
    parser.add_argument('--layer_num', type=int)
    parser.add_argument('--ising_graph_type')
    parser.add_argument('--middle_width', type=int)
    parser.add_argument('--dataset', help='current options are [wishbone, mnist]')
    parser.add_argument('--train_and_predict', action='store_true')
    parser.add_argument('-pd', '--predict_dir', help='run predict recursively')
    parser.add_argument('--add_labels_to_ising', action='store_true')
    parser.add_argument('--laplacian_type', help='Options [unnormalized, sym, rw]')
    parser.add_argument('--conv_decoder', help='Use convolutional decoder')
    parser.add_argument('--warm_start_encoder', help='give dir for to load from') 
    parser.add_argument('--early_stopping_thresh', type=float)
    parser.add_argument('--l2_reg', help='l2 reg on the ising layer activations;')
    parser.add_argument('--l1_reg', help='l2 reg on the ising layer activations;')

    command_args = {k:v for k,v in vars(parser.parse_args()).items() if v} # strip None values
    parsed_args = {}
    if 'args_from_file' in command_args:
        with open(command_args['args_from_file'], 'r') as handle:
            parsed_args = json.load(handle)
    if isinstance(parsed_args, list):
        for pargs in parsed_args:
            run([command_args, pargs])
        return
    run([command_args, parsed_args])


if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = ''
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
