import numpy as np
import pandas as pd

num_clients = 2

prefix = "/home/swier/Documents/afstuderen/datasets/RMA/"
folders = ["A" + str(i) for i in range(1, num_clients + 1)]
data_paths = [prefix  + folders[i - 1] + "/2node_PCA.npy" for i in range(1,num_clients + 1)]
meta_paths = [prefix + folders[i - 1] + "/2node_metadata.npy" for i in range(1,num_clients + 1) ]


for i in range(num_clients):
    data_file = np.load(data_paths[i])
    meta_file = np.load(meta_paths[i], allow_pickle=True)
    
    comps = np.asarray(["comp " + str(i) for i in range(100)])
    metacols = np.asarray(["test/train", "label"])
    full_cols = np.concatenate((comps, metacols))

    full_df = pd.DataFrame(np.concatenate((data_file,meta_file), axis=1), columns=full_cols)
    
    full_df.to_csv(prefix + "PCA_2node_client" + str(i) + '.csv',index=False)

