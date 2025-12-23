import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

def hrp_order_from_correlation(corr):
    dist = np.sqrt(0.5 * (1 - corr))
    link = linkage(dist, method="single")
    return leaves_list(link)

def hrp_order_from_distance(D):
    link = linkage(squareform(D), method="single")
    return leaves_list(link)
