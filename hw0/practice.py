import heapq

import numpy
import numpy as np


# def euclid_x(point, Query_Array, k):
#     """Compute each Euclidean distance from the Query_Array (Which should be the value inside)"""
#     # k_nearest = np.empty([k])
#     index = 0
#     order_heap = []
#     k_best = numpy.empty([k, Query_Array.shape[1]])
#     for x in Query_Array:
#         # version using python
#         # k_nearest[index] = np.sqrt((x[0] - point[0]) ** 2 + (x[1] - point[1]) ** 2)
#         # version using numpy linalg norm
#         # track k smallest using min heap
#         heapq.heappush(order_heap, (np.linalg.norm(x - point), index))
#         index += 1
#     for heap_ind in range(0, k):
#         my_ind = heapq.heappop(order_heap)
#         if my_ind is not None:
#             k_best[heap_ind] = (Query_Array[my_ind[1]])
#     return k_best

def euclid_x(point, Query_Array, k):
    """Compute each Euclidean distance from the Query_Array (Which should be the value inside)"""
    k_best = numpy.empty([k, Query_Array.shape[1]])
    order_list = numpy.empty([Query_Array.shape[0],2])
    index = 0
    for x in Query_Array:
        # store the distance and the index
        order_list[index] = [np.linalg.norm(x - point), index]
        index += 1
    print(order_list)
    order_list = order_list[order_list[:, 1].argsort()]
    print(order_list)
    for index_2 in range(0, k):
        k_best = Query_Array[int(order_list[index_2][1])]
    return k_best
def main():
    data_NF = np.asarray([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    query_QF = np.asarray([[0.9, 0.], [0., -0.9]])
    k = 2
    x = euclid_x(data_NF[3], query_QF, k)



if __name__ == "__main__":
    main()
