### imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
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


task = client.post_task(
    input_ = {
        'method' : 'get_cov_mat'
    },
    name = "PCA, first step",
    image = "sgarst/federated-learning:fedPCA3",
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

print(result)
sys.exit()

global_cov_mat = np.sum(local_cov_mats)