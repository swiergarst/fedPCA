import numpy as np
import tables as tb
from vantage6.tools.util import info











def rpc_master(data):
    pass


def RPC_calc_cov_mat(data, global_mean, global_std, rows_to_calc, iter_num):
    #data_vals = data.drop(columns = ['test/train', 'label']).values
    num_rows = data.drop(columns = ['test/train', 'label']).values.shape[0]
    num_cols = data.drop(columns = ['test/train', 'label']).values.shape[1]

    

    #f = tb.open_file('tmp.h5', 'w')
    #filters = tb.Filters(complevel=5, complib='blosc')
    #result = f.create_carray(f.root, 'data', tb.Float32Atom(), shape=(rows_to_calc, num_cols), filters=filters)

    rows = data.drop(columns = ['test/train', 'label']).values[iter_num * rows_to_calc: min(((iter_num + 1) * rows_to_calc), num_rows),:]

    result = np.zeros((num_cols, num_rows))

    
    for i in range(num_cols):
        if (i%100) == 0:
            info(f"column  {i} of {num_cols}")
        col = np.copy(data.drop(columns = ['test/train', 'label']).values[:,i])
        result[i,:] = np.dot(col,rows)
    
    '''  
    f.close()

    file = tb.open_file('tmp.h5', 'r')
    a = file.root.data
    '''  


    return result

def RPC_get_metadata(data):
    #returns all the required metadata to the server for the construction of the PCA later on
    local_mean = np.mean(data.drop(columns = ['test/train', 'label']).values, axis=0)
    local_std = np.std(data.drop(columns = ['test/train', 'label']).values, axis=0)
    num_rows = data.drop(columns = ['test/train', 'label']).values.shape[0]
    num_cols = data.drop(columns = ['test/train', 'label']).values.shape[1]    
    
    return {
        "local_mean" : local_mean,
        "local_std" : local_std,
        "num_rows" : num_rows,
        "num_cols" : num_cols
    }
