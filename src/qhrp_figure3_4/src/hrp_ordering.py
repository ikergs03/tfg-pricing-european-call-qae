import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

def hrp_order_from_correlation(corr):
    dist = np.sqrt(0.5 * (1 - corr))
    dist_condensed = squareform(dist, checks=False)
    link = linkage(dist_condensed, method="single")
    return leaves_list(link)

def hrp_order_from_distance(D):
    dist_condensed = squareform(D, checks=False)
    link = linkage(dist_condensed, method="single")
    return leaves_list(link)
