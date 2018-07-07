import time
import numpy as np


def euclid_dist(t1, t2):
    """Input dim: n X t_len, t_len
       Ouput dim: n
    """
    return np.sqrt(((t1-t2)**2).sum(axis=1))
#     return np.linalg.norm(t1-t2, axis=1)


def init_centroids(data, num_clust):
    """kmeans++: better init centroids"""
    centroids = np.zeros([num_clust, data.shape[1]]) 
    #### Create init one by one
    centroids[0,:] = data[np.random.randint(0, data.shape[0], 1)]
    for i in range(1, num_clust):
        # For each ts, find min dist to any available centroids. Dim: n
        D2 = np.array([euclid_dist(data, c)**2 for c in centroids[0:i, :]]) \
               .min(axis=0)
        # Make a cum probs from the dist. Far ts tends to get higher prob
        cumprobs = (D2/D2.sum()).cumsum()
        # Use the ts as seed if it pass a random hurdle 
        ind = np.where(cumprobs >= np.random.random())[0][0]
        centroids[i, :] = data[ind]
    return centroids


def kmeans_ts(data, num_clust, num_iter, kmeanpp=False):
	"""
	kmeans for time series
	"""
    print('kmean++ init...')
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
        # - Distance of data to centroids. Dim: c x n
        # - Assign centroids. Dim: n  (get min arg along c axis)
        assigned_centroids = np.array([euclid_dist(data, centroid) for centroid in centroids]) \
                               .argmin(axis=0)
        
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


from sklearn.cluster import KMeans
def kmeans_sklearn(data, num_clust, num_iter):
    kmeans = KMeans(n_clusters=num_clust, max_iter=num_iter).fit(data)
    return kmeans 


def load_sin_data()
    """ Make 10000 sin amplitude signal with noise. Length of signal 500
    """
	n = 10000   # num of sample 
	ts_len = 500  # length of time series

	phases = np.array(np.random.randint(0, 50, [n, 2]))
	pure = np.sin([np.linspace(-np.pi * x[0], -np.pi * x[1], ts_len) for x in phases])
	noise = np.array([np.random.normal(0, 1, ts_len) for x in range(n)])

	signals = pure * noise
	# Normalize everything between 0 and 1
	signals += np.abs(np.min(signals))
	signals /= np.max(signals)

	return signals


def load_linear_data()
    """ Make 10000 linear signal with 5 slopes and with noise. Length of signal 500
    """
    n = 10000
    t_len = 500

    xs = np.linspace(0, 1, t_len)
    noise = np.array([np.random.normal(0,0.1,t_len) for _ in range(n)])
    ts = np.array([xs * -2 for _ in range(int(n/5))])
    for i in range(-1,3):
        new_ts = np.array([xs * i for _ in range(int(n/5))])
        ts = np.append(ts, new_ts, axis=0)

    ts += noise
    np.random.shuffle(ts)
    return ts


from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
def load_tslearn_data()
    X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
    X_train = X_train[y_train < 4]  # Keep first 3 classes
    np.random.shuffle(X_train)
    X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train[:50])  # Keep only 50 time series
    X_train = TimeSeriesResampler(sz=40).fit_transform(X_train)  # Make time series shorter
    X_train = X_train.reshape(50,-1)
    return X_train


def main():
    """
    TODO: rewrite to more sklearn style. At least give me assignment and centroids
    """
	c = 100
	i = 100
    # data = load_sin_data()
    data = load_linear_data()
    t1 = time.time()
	centroids = kmeans_ts(data, num_clust=c, num_iter=i, kmeanpp=True)
	print('Result: ', centroids)
    t2 = time.time()
    print("Took {} seconds".format(t2 - t1))


if __name__ == '__main__':
	main()
