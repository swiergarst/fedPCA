### imports
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import math
import time
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from io import BytesIO
from v6_simpleNN_py.model import model
from config_functions import get_datasets, get_config,get_full_dataset
from comp_functions import average, scaffold
from vantage6.client import Client
from scipy.sparse.linalg import eigs
from vantage6.tools.mock_client import ClientMockProtocol

datasets = get_datasets("MNIST_2class", False, False)

client = ClientMockProtocol(
    datasets= datasets,
    module="v6_fedPCA_py"
### connect to server
)
organizations = client.get_organizations_in_my_collaboration()
org_ids = [organization["id"] for organization in organizations]



num_clients = 10                                                                        
PCA_dims = 100


## first step: make the data zero mean and 1 variance
# to do this, every client needs to return its own mean/variance, as well as dataset size. we also need the dimensions of the dataset, so might as well return those within this step

# get some random vals for mean and variance and eigenvectors
global_var_rand = np.random.rand(784)
global_mean_rand = np.random.rand(784)
vecs_rand = np.random.rand(784,100)

task = client.create_new_task(
    input_ = {
        'method' : 'do_PCA',
        'kwargs' : {
            'eigenvecs' : vecs_rand,
            'global_mean' : global_mean_rand,
            'global_var' : global_var_rand
        }
    },
    organization_ids=org_ids
)





























