import tensorflow as tf
import util
import classifier
import autoencoder
from args import get_default_args, build_args_from_stack

import argparse
import numpy as np
import tensorflow as tf
import time
import os
import json
import pickle


def calculate_preds(data, flags):
    model_fn = classifier.model_fn if flags.use_classifier else autoencoder.model_fn
    model = tf.estimator.Estimator(
        model_fn = model_fn,
        model_dir = flags.model_dir,
        params = {'flags' : flags, 'dataset' : data},
        #config = tf.estimator.RunConfig(session_config=tf.ConfigProto(
        #    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1.0)))
        config = tf.estimator.RunConfig(session_config=tf.ConfigProto(
            device_count = {'GPU':0}))
    )
    preds = util.process_predictions(model.predict(input_fn=lambda: data.get_predict_input(), 
                                                   yield_single_examples=False))
def predict(data, flags):
    ppath = flags.model_dir + '/preds'
    if os.path.exists(ppath):
        preds = pickle.load(open(ppath, 'rb'))
    else:
        preds = calculate_preds(data, flags)
        pickle.dump(preds, open(ppath, 'wb'))
    data.predict(preds)


def run(arg_stack):
    # args are overriden [default, dataset_specific, parsed_from_file, command_line]
    args = build_args_from_stack(arg_stack)
    data = Dataset.factory(args.dataset, args)
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
