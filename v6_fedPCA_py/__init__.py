import numpy as np
import tables as tb
from vantage6.tools.util import info











def rpc_master(data):
    pass


def RPC_get_cov_mat(data):
    #data_vals = data.drop(columns = ['test/train', 'label']).values
    num_rows = data.drop(columns = ['test/train', 'label']).values.shape[0]
    num_cols = data.drop(columns = ['test/train', 'label']).values.shape[1]
    f = tb.open_file('tmp.h5', 'w')
    filters = tb.Filters(complevel=5, complib='blosc')
    result = f.create_carray(f.root, 'data', tb.Float32Atom(), shape=(num_rows, num_cols), filters=filters)
    result[:,:] = data.drop(columns = ['test/train', 'label']).values

    '''
    for i in range(num_cols):
        if (i%100) == 0:
            info(f"column  {i} of {num_cols}")
        vec = np.copy(data.drop(columns = ['test/train', 'label']).values[:,i])
        result[i,:] = np.dot(vec,data.drop(columns = ['test/train', 'label']).values)
    '''
    f.close()

    file = tb.open_file('tmp.h5', 'r')
    a = file.root.data
    out = a[:,:]


    return out