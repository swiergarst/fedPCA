import numpy as np
from vantage6.tools.util import info











def rpc_master(data):
    pass


def RPC_calc_cov_mat(data, global_mean, global_std, rows_to_calc, iter_num):

    
    #data_vals = data.drop(columns = ['test/train', 'label']).values
    num_rows = data.drop(columns = ['test/train', 'label']).values.shape[0]
    num_cols = data.drop(columns = ['test/train', 'label']).values.shape[1]

        # standardize the data
    # workaround for points where the variance = 0
    for i, var in enumerate(global_std):
        if var == 0:
            global_std[i] = 1
    stand_data = ((data.drop(columns = ['test/train', 'label']).values) - global_mean) / global_std


    #f = tb.open_file('tmp.h5', 'w')
    #filters = tb.Filters(complevel=5, complib='blosc')
    #result = f.create_carray(f.root, 'data', tb.Float32Atom(), shape=(rows_to_calc, num_cols), filters=filters)
    row_amt = min( rows_to_calc, num_cols - (iter_num) * rows_to_calc )

    begin_row = iter_num * rows_to_calc

    rows = stand_data[:, begin_row:begin_row + row_amt]

    result = np.zeros((num_cols, row_amt))
    info(f"shape of result: {result.shape}")
    for i in range(num_cols):
        if (i%100) == 0:
            info(f"column  {i} of {num_cols}")
        col = np.copy(stand_data[:,i])
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


def RPC_do_PCA(data,eigenvecs, global_mean, global_std):
        # standardize the data
    # workaround for points where the variance = 0
    for i, var in enumerate(global_std):
        if var == 0:
            global_std[i] = 1
    stand_data = ((data.drop(columns = ['test/train', 'label']).values) - global_mean) / global_std

    data_PCA = np.matmul(stand_data, eigenvecs)



    with open("/mnt/data/PCA_blub.npy", "wb") as f:
        np.save(f, data_PCA)
    
    metadata = data[['test/train', 'label']].values
    with open("mnt/data/metadata.npy", "wb") as f:
        np.save(f, metadata)

    #print(data_PCA)
    return True

def RPC_save_labels(data):
    metadata = data[['test/train', 'label']].values
    with open("mnt/data/metadata.npy", "wb") as f:
        np.save(f, metadata)

    return True


def RPC_saveFile_test(data):
    rand_arr = np.random.rand(100,100)
    with open("/mnt/data/blub.npy", "wb") as f:
        np.save(f, rand_arr)

    return True