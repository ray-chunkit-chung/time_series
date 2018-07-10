
# Original codes from the site:
# - http://alexminnaar.com/time-series-classification-and-clustering-with-python.html
# - https://github.com/alexminnaar/time-series-classification-and-clustering

# Fast dtw here:
# https://pypi.org/project/fastdtw/

# Lower-bounded dtw is cool (See, alexminnaar). Try a multi-variant one here 
# http://ciir.cs.umass.edu/pubfiles/mm-40.pdf


import numpy as np

def dtw(s1, s2, w=None):
    """ Dynamic time wrapping: Univariant.
    Performance:  
    1000 comparison no window: Wall time: 9min 8s
    1000 comparison window=1/5 ts length: Wall time: 5min 32s

    http://alexminnaar.com/time-series-classification-and-clustering-with-python.html
    """
    DTW={}
    if w is None:
        for i in range(len(s1)):
            DTW[(i, -1)] = float('inf')
        for i in range(len(s2)):
            DTW[(-1, i)] = float('inf')
    else:
        w = max(w, abs(len(s1)-len(s2)))
        for i in range(-1,len(s1)):
            for j in range(-1,len(s2)):
                DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    if w is None:
        for i in range(len(s1)):
            for j in range(len(s2)):
                dist = (s1[i]-s2[j])**2
                DTW[(i, j)] = dist + min(DTW[(i-1, j)], DTW[(i, j-1)], DTW[(i-1, j-1)])
    else:
        for i in range(len(s1)):
            for j in range(max(0, i-w), min(len(s2), i+w)):
                dist= (s1[i]-s2[j])**2
                DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
            
    return np.sqrt(DTW[len(s1)-1, len(s2)-1])

def LB_Keogh(s1, s2, r):
    """http://alexminnaar.com/time-series-classification-and-clustering-with-python.html
    LB_Keogh: Univariant.
    Performance:
    1000 comparison r = 1/5 ts length: Wall time: 16.5 s
    """
    LB_sum = 0
    for ind, i in enumerate(s1):
        lower_bound = min(s2[(ind-r if ind-r>=0 else 0): (ind+r)])
        upper_bound = max(s2[(ind-r if ind-r>=0 else 0): (ind+r)])
        if i > upper_bound:
            LB_sum = LB_sum + (i-upper_bound)**2
        elif i < lower_bound:
            LB_sum = LB_sum + (i-lower_bound)**2
    return np.sqrt(LB_sum)


from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
def _fastdtw(s1,s2):
    """ Fastdtw by Kazuaki Tanida. Shiroyagi corp.
    https://pypi.org/project/fastdtw/
    Multivariant.
    Performance: 1000 comparison Wall time: 5min 28s.
                 Comparable to alexminnaar's window-ed dtw.
                 For kmeans use, recommend to use LB_Keogh for first run and fastdtw for an over night run.
    """
    dist, path = fastdtw(s1.reshape(-1,1), s2.reshape(-1,1),dist=euclidean)
    return dist, path



# Pierre's dtw performance is about 3s per comparison.
# Need 100x improvement to be pratical.

# - https://github.com/pierre-rouanet/dtw/blob/master/dtw.py
# def dtw_v3(x, y, dist, warp=1):
#     """
#     Computes Dynamic Time Warping (DTW) of two sequences.
#     :param array x: N1*M array
#     :param array y: N2*M array
#     :param func dist: distance used as cost measure
#     :param int warp: how many shifts are computed.
#     Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
#     """
#     assert len(x)
#     assert len(y)
#     r, c = len(x), len(y)
#     D0 = zeros((r + 1, c + 1))
#     D0[0, 1:] = inf
#     D0[1:, 0] = inf
#     D1 = D0[1:, 1:]  # view
#     for i in range(r):
#         for j in range(c):
#             D1[i, j] = dist(x[i], y[j])
#     C = D1.copy()
#     for i in range(r):
#         for j in range(c):
#             min_list = [D0[i, j]]
#             for k in range(1, warp + 1):
#                 i_k = min(i + k, r - 1)
#                 j_k = min(j + k, c - 1)
#                 min_list += [D0[i_k, j], D0[i, j_k]]
#             D1[i, j] += min(min_list)
#     if len(x)==1:
#         path = zeros(len(y)), range(len(y))
#     elif len(y) == 1:
#         path = range(len(x)), zeros(len(x))
#     else:
#         path = _traceback(D0)
#     return D1[-1, -1] / sum(D1.shape), C, D1, path

# A Vectorized form is here
# - https://stackoverflow.com/questions/28025739/vectorized-loop-for-dynamic-time-wrapping
# r, c = np.array(D.shape)-1
# for a in range(1, r+c):
#     # We have I>=0, I<r, J>0, J<c and J-I+1=a
#     I = np.arange(max(0, a-c), min(r, a))
#     J = I[::-1] + a - min(r, a) - max(0, a-c)
#     # We have to use two np.minimum because np.minimum takes only two args.
#     D[I+1, J+1] += np.minimum(np.minimum(D[I, J], D[I, J+1]), D[I+1, J])
    