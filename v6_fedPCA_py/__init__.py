import numpy as np
import tables as tb
from vantage6.tools.util import info
from sklearn.preprocessing import StandardScaler










def rpc_master(data):
    pass


def RPC_calc_cov_mat(data, global_mean, global_std, rows_to_calc, iter_num):

    
    #data_vals = data.drop(columns = ['test/train', 'label']).values
    num_rows = data.drop(columns = ['test/train', 'label']).values.shape[0]
    num_cols = data.drop(columns = ['test/train', 'label']).values.shape[1]

    

    #f = tb.open_file('tmp.h5', 'w')
    #filters = tb.Filters(complevel=5, complib='blosc')
    #result = f.create_carray(f.root, 'data', tb.Float32Atom(), shape=(rows_to_calc, num_cols), filters=filters)
    row_amt = min( rows_to_calc, num_cols - (iter_num) * rows_to_calc )

    begin_row = iter_num * rows_to_calc

    rows = data.drop(columns = ['test/train', 'label']).values[:, begin_row:begin_row + row_amt]

    result = np.zeros((num_cols, row_amt))
    info(f"shape of result: {result.shape}")
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
    local_var = np.var(data.drop(columns = ['test/train', 'label']).values, axis=0)
    num_rows = data.drop(columns = ['test/train', 'label']).values.shape[0]
    num_cols = data.drop(columns = ['test/train', 'label']).values.shape[1]    
    
    return {
        "local_mean" : local_mean,
        "local_var" : local_var,
        "num_rows" : num_rows,
        "num_cols" : num_cols
    }


def RPC_do_PCA(data,eigenvecs, global_mean, global_var):
    # standardize the data
    scaler = StandardScaler()
    scaler.mean_ = global_mean
    scaler.var_ = global_var
    scaler.scale_ = np.sqrt(global_var)

    stand_data = (scaler.transform(data.drop(columns = ['test/train', 'label']).values))

    data_PCA = np.matmul(stand_data, eigenvecs)
    with open("/mnt/data/PCA_local.npy", "wb") as f:
        np.save(f, data_PCA)

    return True


def RPC_saveFile_test(data):
    rand_arr = np.random.rand(100,100)
    with open("/mnt/data/blub.npy", "wb") as f:
        np.save(f, rand_arr)

    return True