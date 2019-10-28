import json
import numpy as np
def get_defaults():
    return {
        "dataset": "teapot",
        "height": 1,
        "ising_graph_type": "rings",
        "ising_regularization": 0.1,
        "layer_num": 2,
        "model_dir": "hingeloss/",
        "l1_reg": 0.0001,
        "width": 20,
        "train_and_predict": True,
        "train_epochs": 3000,
    }

def get_hub_defaults():
    return {
        "dataset": "hub_perturb",
        "height": 3,
        "ising_regularization": 0.1,
        "l1_reg": 0,
        "ising_graph_type": "lines",
        "layer_num": 2,
        "train_and_predict": True,
        "relu_embedding": True,
        "width": 2
    }

def explore():
    ising_reg = (0.001, 10)
    l1_reg = (1e-4, 1e-3)
    
    runs = []
    for i in range(300):
        d = get_hub_defaults()
        d['l1_reg'] = np.power(10,np.random.uniform(-4, -3))
        d['ising_regularization'] = np.power(10, np.random.uniform(-2, 1))
        d['model_dir'] = 'hubs/gen/3/%d' % i
        runs.append(d)
    print(json.dumps(runs, indent=4, sort_keys=True))
    return runs

if __name__ == '__main__':
    explore()
    
