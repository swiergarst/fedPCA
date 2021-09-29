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

datasets = get_datasets("A2_raw", False, False)

print("Attempt login to Vantage6 API")
client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "1234")
privkey = "/home/swier/.local/share/vantage6/node/privkey_testOrg0.pem"
client.setup_encryption(privkey)

num_clients = 10                                                                        
PCA_dims = 100

ids = [org['id'] for org in client.collaboration.get(1)['organizations']]

# first step: calculate the mean
print("requesting metadata")
metadata_task = client.post_task(
    input_ = {
        'method' : 'get_metadata'
    },
    name = "PCA, get metadata",
    image = "sgarst/federated-learning:fedPCA15",
    organization_ids=ids,
    collaboration_id=1
)


res = np.array(client.get_results(task_id = metadata_task.get("id")))


while(None in [res[i]["result"] for i in range(num_clients)]):
    res = np.array(client.get_results(task_id = metadata_task.get("id")))
    time.sleep(1)
    #print(res[0]."result")

num_cols = np.load(BytesIO(res[0]["result"]),allow_pickle=True)["num_cols"]

local_means = np.zeros((num_clients, num_cols))
local_std = np.zeros((num_clients, num_cols))
dataset_sizes = np.zeros(num_clients)


for i in range(num_clients):
    local_means[i,:] = np.load(BytesIO(res[i]["result"]),allow_pickle=True)["local_mean"]
    local_std[i,:] = np.load(BytesIO(res[i]["result"]),allow_pickle=True)["local_std"]
    dataset_sizes[i] = np.load(BytesIO(res[i]["result"]),allow_pickle=True)["num_rows"]


# calculate weighted average/std over all clients
global_mean = average(local_means, dataset_sizes, None, None, None, use_sizes=True, use_imbalances=False)
#global_std = average(local_std, dataset_sizes, None, None, None, use_sizes=True, use_imbalances=False)



# second step: calculate the covariance matrix, so we get a 'global variance'/std 
rows_to_calc = 5
global_std_dummy = np.ones(num_cols)

cov_rounds = math.ceil(num_cols/rows_to_calc)
global_cov_mat_unnormed = np.zeros((num_cols,num_cols))
print("starting cov matrix calculations")
for round in range(cov_rounds):
    print("round", round , "of", cov_rounds)
    cov_partial_task = client.post_task(
        input_= {
            "method" : "calc_cov_mat",
            "kwargs" : {
                "global_mean" : global_mean,
                "global_std" : global_std_dummy,
                "rows_to_calc" : rows_to_calc,
                "iter_num" : round
            }
        },
        name = "PCA, covariance calc for std, round" + str(round),
        image= "sgarst/federated-learning:fedPCA15",
        organization_ids=ids,
        collaboration_id=1
    )

    res = np.array(client.get_results(task_id = cov_partial_task.get("id")))

    while(None in [res[i]["result"] for i in range(num_clients)]):
        res = np.array(client.get_results(task_id = cov_partial_task.get("id")))
        time.sleep(1)
        #print(res[0]."result")

    for i in range(num_clients):
        global_cov_mat_unnormed[:,round * rows_to_calc: min((round + 1) * rows_to_calc, num_cols)] += np.load(BytesIO(res[i]["result"]), allow_pickle=True)

global_cov_mat_corrected = global_cov_mat_unnormed / (np.sum(dataset_sizes) - 1)

with open ("cov_mat_global_corrected.npy", "wb") as f:
    np.save(f, global_cov_mat_corrected)




vars = np.diagonal(global_cov_mat_unnormed)



global_std = np.sqrt(vars)



# send weighted average/std back, let nodes calculate local covariance matrix. unfortunately, we have to do this 5 rows at a time b/c of sending file size limits

rows_to_calc = 5

cov_rounds = math.ceil(num_cols/rows_to_calc)
global_cov_mat = np.zeros((num_cols,num_cols))
print("starting cov matrix calculations")
for round in range(cov_rounds):
    print("round", round , "of", cov_rounds)
    cov_partial_task = client.post_task(
        input_= {
            "method" : "calc_cov_mat",
            "kwargs" : {
                "global_mean" : global_mean,
                "global_std" : global_std,
                "rows_to_calc" : rows_to_calc,
                "iter_num" : round
            }
        },
        name = "PCA, covariance calc, round" + str(round),
        image= "sgarst/federated-learning:fedPCA15",
        organization_ids=ids,
        collaboration_id=1
    )

    res = np.array(client.get_results(task_id = cov_partial_task.get("id")))

    while(None in [res[i]["result"] for i in range(num_clients)]):
        res = np.array(client.get_results(task_id = cov_partial_task.get("id")))
        time.sleep(1)
        #print(res[0]."result")

    for i in range(num_clients):
        global_cov_mat[:,round * rows_to_calc: min((round + 1) * rows_to_calc, num_cols)] += np.load(BytesIO(res[i]["result"]), allow_pickle=True)

    
'''
with open ("cov_mat_global.npy", "wb") as f:
    np.save(f, global_cov_mat)
'''

# calculate eigenvalues/vectors of covariance matrix

w,v = eigs(global_cov_mat, k = PCA_dims)



# send the weighted avg/std, as well as the eigenvectors to nodes, so they can finally calculate the PCA.
pca_task = client.post_task(
    input_= {
        "method" : "do_PCA",
        "kwargs" : {
            "eigenvecs" : v.real,
            "global_mean" : global_mean,
            "global_std" : global_std
        }
    },
    name = "final step of PCA",
    image = "sgarst/federated-learning:fedPCA15",
    organization_ids=ids,
    collaboration_id=1
)

while(None in [res[i]["result"] for i in range(num_clients)]):
    res = np.array(client.get_results(task_id = pca_task.get("id")))
    time.sleep(1)


if (np.load(BytesIO(res[0]["result"]), allow_pickle=True).all()):
            print("PCA complete!")

'''
print("blub")
metadata_task = client.post_task(
    input_ = {
        'method' : 'saveFile_test'
    },
    name = "savefile test",
    image = "sgarst/federated-learning:fileTest",
    organization_ids=ids,
    collaboration_id=1
)
'''