import numpy as np

prefix = "/home/swier/Documents/afstuderen/datasets/RMA/A2/PCA/AML_A2_PCA_client"
paths = [prefix + str(i) + ".npy" for i in range(10)]


for i in range(10):
    npy_file = np.load(paths[i])
    np.savetxt(prefix + str(i) + ".csv", npy_file, delimiter=',')