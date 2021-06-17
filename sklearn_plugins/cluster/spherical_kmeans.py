"""Implementation of Spherical K-Means Clusting that is compatible with sklearn.
"""
from typing import Union

import numpy as np
from numpy import random
from numpy.random import RandomState
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

__author__ = "Hung-Tien Huang"
__copyright__ = "Copyright 2021, Hung-Tien Huang"


class SphericalKMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    """Spherical K-Means Clustering.

    An implementation of spherical K-Means clustering based on Coats and Ng, "Learning Feature Representations with K-Means", 2012.

    Input flow:
    ```pseudo
        if normalize:
            __normalizer.fit_transform(x)
        if standarize:
            __std_scalar.fit_transform(x)
        __pca.fit_transform(x)
        perform_kmeans(x)
    ```

    Attributes:
        n_clusters (int, optional): The number of clusters to form as well as the number of centroids to generate.. Defaults to 500.
        n_components (Union[int, float, str, None], optional): Number of components to keep after PCA. If n_components is not set all components are kept. Defaults to 0.8.
        normalize (bool, optional): Normalize features within individual sample to zero mean and unit variance prior to training. Defaults to True.
        standarize (bool, optional): Standarize individual features across dataset to zero mean and unit variance after normalization prior to whitening. Defaults to True.
        whiten (bool, optional): When True, the components_ vectors are multiplied by the square root of n_samples and then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances. Defaults to True.
        max_iter (int, optional): Maximum number of iterations of the k-means algorithm for a single run.. Defaults to 100.
        tol (float, optional): Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence. Defaults to 0.01.
        copy (bool, optional): When True, the data are modified in-place. Defaults to True.
    """
    # public:
    n_clusters: int
    n_components: Union[int, float, str, None]
    whiten: bool
    normalize: bool
    standarize: bool
    max_iter: int
    tol: float
    random_state: Union[int, RandomState, None]
    copy: bool

    # protected
    _pca_: PCA
    _std_scalar_: StandardScaler

    # private
    __centroids: np.ndarray
    __n_components: int
    __n_samples: int

    def __init__(self,
                 n_clusters: int = 500,
                 n_components: Union[int, float, str, None] = 0.8,
                 normalize: bool = True,
                 standarize: bool = True,
                 whiten: bool = True,
                 max_iter: int = 100,
                 tol: float = 0.01,
                 random_state: Union[int, random.RandomState, None] = None,
                 copy: bool = True):
        """Constructor for SphericalKMeans

        Args:
            n_clusters (int, optional): The number of clusters to form as well as the number of centroids to generate.. Defaults to 500.
            n_components (Union[int, float, str, None], optional): Number of components to keep after PCA. If n_components is not set all components are kept. Defaults to 0.8.
            normalize (bool, optional): Normalize features within individual sample to zero mean and unit variance prior to training. Defaults to True.
            standarize (bool, optional): Standarize individual features across dataset to zero mean and unit variance after normalization prior to whitening. Defaults to True.
            whiten (bool, optional): When True, the components_ vectors are multiplied by the square root of n_samples and then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances. Defaults to True.
            max_iter (int, optional): Maximum number of iterations of the k-means algorithm for a single run.. Defaults to 100.
            tol (float, optional): Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence. Defaults to 0.01.
            copy (bool, optional): When True, the data are modified in-place. Defaults to True.
        """
        # copy arguments
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.normalize = normalize
        self.standarize = standarize
        self.whiten = whiten
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.copy = copy
        # create instances
        self._pca_ = PCA(n_components=n_components, copy=copy, whiten=whiten)
        if standarize:
            self._std_scalar_ = StandardScaler(copy=copy)
        else:
            self._std_scalar_ = None
        # private attributes
        self.__centroids = None
        self.__n_components = -1
        self.__n_samples = -1

    def fit(self, X: np.ndarray) -> "SphericalKMeans":
        """Compute k-means clustering.

        Args:
            X (np.ndarray): (n_samples, n_features) Training instances to cluster. It must be noted that the data will be converted to C ordering, which will cause a memory copy if the given data is not C-contiguous. If a sparse matrix is passed, a copy will be made if it’s not in CSR format.
        
        Returns:
            self (SphericalKMeans): Fitted estimator
        """
        # # features of each sample has zero mean and unit variance; data-point level
        # if self.normalize:
        #     X, _ = normalize(X, norm="l2", axis=1, copy=self.copy)
        # # each features in the dataset has zero mean and unit variance; dataset level
        # if self.standarize:
        #     X = self._std_scalar_.fit_transform(X)
        # # PCA whiten
        # X = self._pca_.fit_transform(X)
        X = self.__preprocess_input(X, is_train=True)
        # configure dimension
        self.__n_samples, self.__n_components = X.shape
        # start k-means
        self.__init_centroids()
        avg_centoids_shift: float = np.inf
        iter: int = 0
        while iter < self.max_iter and avg_centoids_shift < self.tol:
            # centroid.shape = (n_components, n_clusters)
            prev_centroids: np.ndarray = np.copy(self.__centroids)
            self.__update_centroids(X)
            centroids_shift: np.ndarray = np.linalg.norm(prev_centroids -
                                                         self.__centroids,
                                                         axis=0)
            avg_centoids_shift: float = np.mean(centroids_shift)
            iter = iter + 1
        return self

    def predict(self, X: np.ndarray, copy: bool = True) -> np.ndarray:
        X = self.__preprocess_input(X, is_train=False, copy=copy)
        S_proj: np.ndarray = np.matmul(X, self.__centroids)
        labels: np.ndarray = np.argmax(S_proj, axis=1)
        return labels

    # private
    def __init_centroids(self):
        """Initialize the centroids

        Randomly initialize the centoids from a standard normal distribution and then normalize to unit length.
        """
        random_state: RandomState = self.random_state
        # initialize from standard normal
        self.__centroids = random_state.standard_normal(
            size=[self.__n_components, self.n_clusters])
        # normalize to unit length
        self.__centroids, _ = normalize(self.__centroids,
                                        axis=0,
                                        norm="l2",
                                        copy=False)

    def __preprocess_input(self,
                           X: np.ndarray,
                           is_train: bool = False,
                           copy: bool = True) -> np.ndarray:
        """Preprocess Input

        Args:
            X (np.ndarray): (n_samples, n_features) raw input data
            is_train (bool, optional): If true, call fit_transform(X) on std_scalar and pca. Otherwise, call std_scalar_.fit(X, copy=copy). Defaults to False.
            copy (bool, optional): Whether or not to modify in-place during inference call. Defaults to True.

        Returns:
            X (np.ndarray): (n_samples, n_components) preprocessed input
        """
        # features of each sample has zero mean and unit variance; data-point level
        if self.normalize:
            X, _ = normalize(X, norm="l2", axis=1, copy=self.copy)
        # each features in the dataset has zero mean and unit variance; dataset level
        if self.standarize:
            if is_train == False:
                X = self._std_scalar_.transform(X, copy=copy)
            else:
                X = self._std_scalar_.fit_transform(X)
        # PCA whiten
        if is_train == False:
            X = self._pca_.transform(X)
        else:
            X = self._pca_.fit_transform(X)
        return X

    def __update_centroids(self, X: np.ndarray):
        """Update Centroids

        Args:
            X (np.ndarray): (n_samples, n_components)
        """
        # centroid.shape = (n_components, n_clusters)
        # X.shape = (n_samples, n_components)
        # S_proj.shpae = (n_samples, cluster) each sample's projection on each cluster
        S_proj: np.ndarray = np.matmul(X, self.__centroids)
        labels: np.ndarray = np.argmax(S_proj, axis=1)
        # S_code.shpae = (n_samples, cluster)
        S_code: np.ndarray = np.zeros_like(S_proj)
        S_code[np.arange(self.__n_samples),
               labels] = S_proj[np.arange(self.__n_samples), labels]
        # update centroids
        self.__centroids = np.matmul(X.transpose(), S_code) + self.__centroids
        # normalize centroids
        self.__centroids, _ = normalize(self.__centroids,
                                        axis=0,
                                        norm="l2",
                                        copy=False)
