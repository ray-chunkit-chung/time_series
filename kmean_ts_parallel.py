# Codes are based on Jason Snell • Oct. 23rd, 2017 • New Relic News
# Optimizing k-means Clustering for Time Series Data
# https://blog.newrelic.com/2017/10/23/optimizing-k-means-clustering/

# Start work node on shell
#$> ipcluster start -n 4

import ipyparallel as ipp
c = ipp.Client()
v = c[:]
v.use_cloudpickle()

with v.sync_imports():
    import numpy as np


def euclid_dist(t1, t2):
    """Input dim: n X t_len, t_len
       Ouput dim: n
    """
    return np.sqrt(((t1-t2)**2).sum(axis=1))

def init_centroids(data, num_clust):
    """kmeans++: better init centroids"""
    v.push(dict(data=data))
    centroids = np.zeros([num_clust, data.shape[1]]) 
    centroids[0,:] = data[np.random.randint(0, data.shape[0], 1)]
    for i in range(1, num_clust):
        # For each ts, find min dist to any available centroids. Dim: n. Parallelized
        D2 = np.array(v.map_sync(lambda x: np.sqrt(((data-x)**2).sum(axis=1)), centroids[0:i, :])) \
               .min(axis=0)
#         D2 = np.array([euclid_dist(data, c)**2 for c in centroids[0:i, :]]).min(axis=0)
        # Make a cum probs from the dist. Far ts tends to get higher prob
        cumprobs = (D2/D2.sum()).cumsum()
        # Use the ts as seed if it pass a random hurdle 
        ind = np.where(cumprobs >= np.random.random())[0][0]
        centroids[i, :] = data[ind]
    return centroids

def k_means(data, num_clust, num_iter, kmeanpp=False):
    print('Init...')
    t1 = time.time()
    if kmeanpp==True:
        # kmean++ init
        centroids = init_centroids(data, num_clust)
    else:
        # Randomly pick a few time series as seed
        centroids = data[np.random.randint(0, data.shape[0], num_clust)]
    t2 = time.time()
    print("Took {} seconds".format(t2 - t1))

    for itr in range(num_iter):
        if itr % int(num_iter/10)==0: print('Iter: {}'.format(itr))
        ##### E step
        # - Distance of data to centroids. Dim: c x n. Parallelized
        # - Assign centroids. Dim: n  (get min arg along c axis)
        assigned_centroids = np.array(v.map_sync(lambda x: np.sqrt(((data-x)**2).sum(axis=1)), centroids)) \
                               .argmin(axis=0)
#         assigned_centroids = np.array([euclid_dist(data, centroid) for centroid in centroids]) \
#                                .argmin(axis=0)
        
        ##### M step
        # new_centroids. Dim: c X t_en
        new_centroids = np.array(
            [data[assigned_centroids==c_idx].mean(axis=0) for c_idx in np.unique(assigned_centroids)])
        
        # If clusters collapse, add randomly chosen centroid
        k = centroids.shape[0]
        if k - new_centroids.shape[0] > 0:
            print("Adding {} centroid(s)".format(k - new_centroids.shape[0]))
            additional_centroids = data[np.random.randint(0, data.shape[0], k - new_centroids.shape[0])] 
            new_centroids = np.append(new_centroids, additional_centroids, axis=0)
        
        # Early stopping if no centroids change
        if not np.any(new_centroids != centroids):
            print("Early stopping!")
            break

        centroids = new_centroids

    return centroids

def main():
    pass

if __name__ == '__main__':
    main()
