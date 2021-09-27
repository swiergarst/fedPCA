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

print("requesting metadata")
metadata_task = client.create_new_task(
    input_ = {
        'method' : 'get_metadata'
    },
    organization_ids=org_ids
)

res = np.array(client.get_results(task_id = metadata_task.get("id")))
#print(res)

num_cols = res[0]["num_cols"]

local_means = np.zeros((num_clients, num_cols))
local_vars = np.zeros((num_clients, num_cols))
dataset_sizes = np.zeros(num_clients)

for i in range(num_clients): 
    local_means[i,:] = res[i]["local_mean"]
    local_vars[i,:] = res[i]["local_var"]
    dataset_sizes[i] = res[i]["num_rows"]
# get some random vals for the covariance matrix
cov_rand = np.random.rand(num_cols,num_cols)
w,v  = eigs(cov_rand, k = PCA_dims)

global_mean = average(local_means, dataset_sizes, None, None, None, use_sizes=True, use_imbalances=False)
global_var = average(local_vars, dataset_sizes, None, None, None, use_sizes=True, use_imbalances=False)

cov_partial_task = client.post_task(
        input_= {
            "method" : "calc_cov_mat",
            "kwargs" : {
                "global_mean" : global_mean,
                "global_var" : global_var,
                "rows_to_calc" : 5,
                "iter_num" : 0
            }
        },
        organization_ids=org_ids,
    )

task = client.create_new_task(
    input_ = {
        'method' : 'do_PCA',
        'kwargs' : {
            'eigenvecs' : v.real,
            'global_mean' : global_mean,
            'global_var' : global_var
        }
    },
    organization_ids=org_ids
)





























