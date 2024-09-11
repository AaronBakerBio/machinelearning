'''
K Nearest Neighbors

>>> data_NF = np.asarray([
...     [1., 0.],
...     [0., 1.],
...     [-1., 0.],
...     [0., -1.]])
>>> query_QF = np.asarray([
...     [0.9, 0.],
...     [0., -0.9]])

Example Test K=1
----------------
# Find the single nearest neighbor for each query vector
>>> neighb_QKF = calc_k_nearest_neighbors(data_NF, query_QF, K=1)
>>> neighb_QKF.shape
(2, 1, 2)

# Neighbor of [0.9, 0]
>>> neighb_QKF[0]
array([[1., 0.]])

# Neighbor of [0, -0.9]
>>> neighb_QKF[1]
array([[ 0., -1.]])

Example Test K=3
----------------
# Now find 3 nearest neighbors for the same queries
>>> neighb_QKF = calc_k_nearest_neighbors(data_NF, query_QF, K=3)
>>> neighb_QKF.shape
(2, 3, 2)

# Neighbor of [0.9, 0]
>>> neighb_QKF[0]
array([[ 1.,  0.],
       [ 0.,  1.],
       [ 0., -1.]])

# Neighbor of [0, -0.9]
>>> neighb_QKF[1]
array([[ 0., -1.],
       [ 1.,  0.],
       [-1.,  0.]])
'''
import numpy as np


def calc_k_nearest_neighbors(data_NF, query_QF, K=1):
    ''' Compute and return k-nearest neighbors under Euclidean distance

    Args
    ----
    data_NF : 2D np.array, shape = (n_examples, n_feats) == (N, F)
        Each row is a feature vector for one example in dataset
        this is the training set
    query_QF : 2D np.array, shape = (n_queries, n_feats) == (Q, F)
        Each row is a feature vector whose neighbors we want to find
        (this is the test set)
    K : int, must satisfy K >= 1 and K <= n_examples aka N
        Number of neighbors to find per query vector

    Returns
    -------
    neighb_QKF : 3D np.array, (n_queries, n_neighbors, n_feats) == (Q, K, F)
        Entry q,k is feature vector of k-th nearest neighbor of the q-th query
        If two vectors are equally close, then we break ties by taking the one
        appearing first in row order in the original data_NF array
    '''
    Q, F = query_QF.shape
    neighb_QKF = np.empty([Q, K, F])
    for index in range(0, Q):
        neighb_QKF[index] = euclid_x(query_QF[index], data_NF, K)
    return neighb_QKF

    # TODO fixme

def euclid_x(point, Query_Array, k):
    """Compute each Euclidean distance from the Query_Array (Which should be the value inside)"""
    k_best = np.empty([k, Query_Array.shape[1]])
    order_list = np.empty([Query_Array.shape[0], 2])
    index = 0
    for x in Query_Array:
        order_list[index] = [np.linalg.norm(x - point), index]
        index += 1
    order_list = order_list[order_list[:, 0].argsort()]
    for index_2 in range(0, k):
        k_best[index_2] = Query_Array[int(order_list[index_2][1])]
    return k_best
