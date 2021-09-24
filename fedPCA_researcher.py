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

datasets = get_datasets("A2_raw", False, False)

print("Attempt login to Vantage6 API")
client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "1234")
privkey = "/home/swier/.local/share/vantage6/node/privkey_testOrg0.pem"
client.setup_encryption(privkey)

num_clients = 10                                                                        


ids = [org['id'] for org in client.collaboration.get(1)['organizations']]


## first step: make the data zero mean and 1 variance
# to do this, every client needs to return its own mean/variance, as well as dataset size. we also need the dimensions of the dataset, so might as well return those within this step

print("requesting metadata")
metadata_task = client.post_task(
    input_ = {
        'method' : 'get_metadata'
    },
    name = "PCA, get metadata",
    image = "sgarst/federated-learning:fedPCA7",
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
local_stds = np.zeros((num_clients, num_cols))
dataset_sizes = np.zeros(num_clients)


for i in range(num_clients):
    local_means[i,:] = np.load(BytesIO(res[i]["result"]),allow_pickle=True)["local_mean"]
    local_stds[i,:] = np.load(BytesIO(res[i]["result"]),allow_pickle=True)["local_std"]
    dataset_sizes[i] = np.load(BytesIO(res[i]["result"]),allow_pickle=True)["num_rows"]


# calculate weighted average/std over all clients
## TODO: doublecheck if this math is legit

global_mean = average(local_means, dataset_sizes, None, None, None, use_sizes=True, use_imbalances=False)
global_std = average(local_stds, dataset_sizes, None, None, None, use_sizes=True, use_imbalances=False)


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
        image= "sgarst/federated-learning:fedPCA7",
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

    

print(global_cov_mat)

with open ("cov_mat_global.npy", "wb") as f:
    np.save(f, global_cov_mat)



# calculate global covariance matrix by summing up all the local cov matrices



# send the weighted avg/std, as well as global covariance matrix to nodes, so they can finally calculate the PCA.


sys.exit()







task = client.post_task(
    input_ = {
        'method' : 'get_cov_mat'
    },
    name = "PCA, first step",
    image = "sgarst/federated-learning:fedPCAOut5",
    organization_ids=ids,
    collaboration_id = 1
)



res = np.array(client.get_results(task_id = task.get("id")))

while(None in [res[i]["result"] for i in range(num_clients)]):
    res = np.array(client.get_results(task_id = task.get("id")))
    time.sleep(1)
    #print(res[0]."result")
result = []
for i in range(num_clients):
    result.append(np.load(BytesIO(res[i]["result"]),allow_pickle=True))

print(result[0].shape)
sys.exit()

global_cov_mat = np.sum(local_cov_mats)