import numpy as np

from vantage6.tools.util import info











def rpc_master(data):
    pass


def RPC_get_cov_mat(data):
    data_vals = data.drop(columns = ['test/train', 'label']).values

    num_cols = data_vals.shape[1]
    result = np.zeros((num_cols,num_cols))
    for i in range(num_cols):
        if (i%100) == 0:
            info(f"column  {i} of {num_cols}")
        vec = np.copy(data_vals[:,i])
        result[i,:] = np.dot(vec,data_vals)

    return result