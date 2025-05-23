import numpy as np

#################################
# DO NOT IMPORT OHTER LIBRARIES
#################################

def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data - numpy array of points
    :param generator: random number generator. Use it in the same way as np.random.
            In grading, to obtain deterministic results, we will be using our own random number generator.


    :return: a list of length n_clusters with each entry being the *index* of a sample
             chosen as centroid.
    '''
    p = generator.randint(0, n) #this is the index of the first center
    #############################################################################
    # TODO: implement the rest of Kmeans++ initialization. To sample an example
	# according to some distribution, first generate a random number between 0 and
	# 1 using generator.rand(), then find the the smallest index n so that the 
	# cumulative probability from example 1 to example n is larger than r.
    #############################################################################
    centers=[p]
    for _ in range(1,n_cluster):
        distances = np.array([min(np.sum((x[i] - x[c]) ** 2) for c in centers) for i in range(n)
                              ])
        total = np.sum(distances)
        probs = distances/total
        cumulative = np.cumsum(probs)

        r = generator.rand()
        next_center = np.searchsorted(cumulative,r)
        centers.append(next_center)


    # DO NOT CHANGE CODE BELOW THIS LINE
    return centers


# Vanilla initialization method for KMeans
def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)



class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple in the following order:
                  - final centroids, a n_cluster X D numpy array, 
                  - a length (N,) numpy array where cell i is the ith sample's assigned cluster's index (start from 0), 
                  - number of times you update the assignment, an Int (at most self.max_iter)
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        self.generator.seed(42)
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        ###################################################################
        # TODO: Update means and membership until convergence 
        #   (i.e., average K-mean objective changes less than self.e)
        #   or until you have made self.max_iter updates.
        ###################################################################
        # center_index = get_lloyd_k_means(N,self.n_cluster,x,self.generator)
        prev_j = float('inf')
        count = 0
        cur_centers= x[self.centers]
        for _ in range(self.max_iter):
            # dists = np.zeros((N, self.n_cluster))
            # for i in range(N):
            #     for j in range(self.n_cluster):
            #         dists[i,j] = np.linalg.norm(x[i]-cur_centers[j])
            dists = np.linalg.norm(x[:, None, :] - cur_centers[None, :, :], axis=2)
            labels = np.argmin(dists,axis=1)
            next_centers = np.zeros_like(cur_centers)
            for c in range(self.n_cluster):
                points_to_c = x[labels==c]
                next_centers[c] = points_to_c.mean(axis=0)
            j = np.sum((x-next_centers[labels])**2)
            avg_change = abs(prev_j-j)/N
            count+=1
            self.centers = cur_centers
            cur_centers = next_centers
            if avg_change<self.e:
                break
            prev_j = j

        return self.centers,labels,count

class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Store following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (numpy array of length n_cluster)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        ################################################################
        # TODO:
        # - assign means to centroids (use KMeans class you implemented, 
        #      and "fit" with the given "centroid_func" function)
        # - assign labels to centroid_labels
        ################################################################
        kmeans = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e, generator=self.generator)
        centroids,labels,_ = kmeans.fit(x,centroid_func)
        centroid_labels = np.zeros(self.n_cluster)
        for c in range(self.n_cluster):
            assigned_c = y[labels==c]
            counting_y = np.bincount(assigned_c)
            centroid_labels[c] = np.argmax(counting_y)
        # DO NOT CHANGE CODE BELOW THIS LINE
        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        ##########################################################################
        # TODO:
        # - for each example in x, predict its label using 1-NN on the stored 
        #    dataset (self.centroids, self.centroid_labels)
        ##########################################################################
        distance = np.zeros((N,self.n_cluster))
        for i in range(N):
            for j in range(self.n_cluster):
                distance[i,j] = np.linalg.norm(x[i]-self.centroids[j])

        assign_c = np.argmin(distance,axis=1)
        predicted_labels = self.centroid_labels[assign_c]
        return predicted_labels







def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors (aka centroids)

        Return a new image by replacing each RGB value in image with the nearest code vector
          (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'
    ##############################################################################
    # TODO
    # - replace each pixel (a 3-dimensional point) by its nearest code vector
    ##############################################################################
    H,W,_ = image.shape
    pixs = image.reshape(H*W,3)
    # N = pixs.shape[0]
    # C = code_vectors.shape[0]
    # dists = np.zeros((N, C))

    # for i in range(N):
    #     for j in range(C):
    #         diff = pixs[i] - code_vectors[j]
    #         dists[i, j] = np.sqrt(np.sum(diff ** 2))
    dists = np.linalg.norm(pixs[:, None, :] - code_vectors[None, :, :], axis=2)

    nearest_idx = np.argmin(dists, axis=1)
    compressed_pixels = code_vectors[nearest_idx]  # shape: (H*W, 3)

    compressed_image = compressed_pixels.reshape(H, W, 3)

    return compressed_image


