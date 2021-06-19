"""Implementation of Spherical K-Means Clusting that is compatible with sklearn.
"""
import copy
import multiprocessing as mp
import os
from typing import Dict, List, Tuple, Union

import numpy as np
from numpy import random
from numpy.random import RandomState
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.utils import check_random_state

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
        n_clusters (int): The number of clusters to form as well as the number of centroids to generate.. Defaults to 500.
        n_components (Union[int, float, str, None]): Number of components to keep after PCA. If n_components is not set all components are kept. Defaults to 0.8.
        normalize (bool): Normalize features within individual sample to zero mean and unit variance prior to training. Defaults to True.
        standarize (bool): Standarize individual features across dataset to zero mean and unit variance after normalization prior to whitening. Defaults to True.
        whiten (bool): When True, the components_ vectors are multiplied by the square root of n_samples and then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances. Defaults to True.
        max_iter (int): Maximum number of iterations of the k-means algorithm for a single run.. Defaults to 100.
        tol (float): Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence. Defaults to 0.01.
        copy (bool): When True, the data are modified in-place. Defaults to True.
    """
    # public:
    n_clusters: int
    n_components: Union[int, float, str, None]
    whiten: bool
    normalize: bool
    standarize: bool
    n_init: int
    n_processes: Union[int, None]
    max_iter: int
    tol: float
    random_state: Union[int, RandomState, None]
    copy: bool

    # private
    __centroids_: np.ndarray
    __n_components_: int
    __n_samples_: int
    __inertia_: float
    __pca_: PCA
    __std_scalar_: StandardScaler

    def __init__(self,
                 n_clusters: int = 500,
                 n_components: Union[int, float, str, None] = 0.8,
                 normalize: bool = True,
                 standarize: bool = True,
                 whiten: bool = True,
                 n_init: int = 10,
                 n_processes: int = None,
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
            n_init (int, optional): Number of time the algorithm will be run with different centroid initialization. The final results will be the berst output of n_init consecutive runs in terms of inertia.
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
        self.n_init = n_init
        self.n_processes = n_processes
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.copy = copy
        # create instances
        self.__pca_ = PCA(n_components=n_components, copy=copy, whiten=whiten)
        if standarize:
            self.__std_scalar_ = StandardScaler(copy=copy)
        else:
            self.__std_scalar_ = None
        # private attributes
        self.__centroids_ = None
        self.__n_components_ = -1
        self.__n_samples_ = -1
        self.__inertia_ = 0.0

    def fit(self, X: np.ndarray, y=None) -> "SphericalKMeans":
        """Compute k-means clustering.

        Args:
            X (np.ndarray): (n_samples, n_features) Training instances to cluster. It must be noted that the data will be converted to C ordering, which will cause a memory copy if the given data is not C-contiguous. If a sparse matrix is passed, a copy will be made if it’s not in CSR format.
            y (Ignored): Not used, present here for API consistency by convention.
        
        Returns:
            self (SphericalKMeans): Fitted estimator
        """
        # preprocess input
        X = self.__preprocess_input(X, is_train=True, copy=self.copy)
        # configure dimension
        self.__n_samples_, self.__n_components_ = X.shape
        # start k-means
        if self.n_init == 1:
            self.__centroids_, self.__inertia_ = self.__fit_kmeans(
                X, self.random_state)
        else:
            random_state: RandomState = check_random_state(self.random_state)
            random_seeds: np.ndarray = random_state.randint(low=0,
                                                            high=np.iinfo(
                                                                np.uint32).max,
                                                            size=self.n_init)
            n_processes: int = self.__get_n_usable_processes()
            if n_processes == 1:
                self.__centroids_, self.__inertia_ = self.__fit_kmeans(
                    X, self.random_state)
            else:
                random_states_list: np.ndarray = np.array_split(
                    random_seeds, n_processes)
                ret_queue: mp.Queue = mp.Queue()
                args_list: List[Tuple[np.ndarray, int, mp.Queue]] = [
                    (X, seeds, ret_queue) for seeds in random_states_list
                ]
                centroids_list, inertia_list = self.__fit_kmeans_processes(
                    args_list, ret_queue)
                min_index: int = np.argmin(inertia_list)
                self.__inertia_ = inertia_list[min_index]
                self.__centroids_ = centroids_list[min_index]
        return self

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling `fit(X)` followed by `predict(X)`.

        Args:
            X (np.ndarray): New data to transform.
            y (Ignored): Not used, present here for API consistency by convention.

        Returns:
            labels (np.ndarray): Index of the cluster each sample belongs to.
        """
        self.fit(X)
        if self.copy == False:
            _, labels = self.__calculate_projections_labels(
                X, self.__centroids_)
            return labels
        return self.predict(X, copy=self.copy)

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Compute clustering and transform X to cluster-distance space.

        Convenience method; equivalent to calling `fit(X)` followed by `transform(X)`.

        Args:
            X (np.ndarray): New data to transform.
            y (Ignored): Not used, present here for API consistency by convention.

        Returns:
            S_proj (np.ndarray): X transformed in the new space.
        """
        self.fit(X)
        if self.copy == False:
            S_proj: np.ndarray = self.__calculate_projections(
                X, self.__centroids_)
            return S_proj
        return self.transform(X, copy=self.copy)

    def predict(self, X: np.ndarray, copy: bool = True) -> np.ndarray:
        """Predict the closest cluster each sample in X belongs to.

        Args:
            X (np.ndarray): (n_samples, n_features) New data to predict.
            copy (bool, optional): Whether or not to modify in-place during inference call. Defaults to True.

        Returns:
            labels (np.ndarray): (n_samples) Index of the cluster each sample belongs to.
        """
        X = self.__preprocess_input(X, is_train=False, copy=copy)
        _, labels = self.__calculate_projections_labels(X, self.__centroids_)
        return labels

    def transform(self, X: np.ndarray, copy: bool = True) -> np.ndarray:
        """Transform X to a cluster-distance space.

        Args:
            X (np.ndarray): (n_samples, n_features) New data to transform.
            copy (bool, optional): Whether or not to modify in-place during inference call. Defaults to True.

        Returns:
            S_proj (np.ndarray): (n_samples, n_features) X transformed in the new space.
        """
        X = self.__preprocess_input(X, is_train=False, copy=copy)
        S_proj = self.__calculate_projections(X, self.__centroids_)
        return S_proj

    def score(self, X: np.ndarray, copy: bool = True):
        """Opposite of the sum of squared distances of samples to their closest cluster center.

        Args:
            X (np.ndarray): (n_samples, n_features) raw input data

        Returns:
            score float: -1 * inertia
        """
        X = self.__preprocess_input(X, is_train=False, copy=copy)
        _, labels = self.__calculate_projections_labels(X, self.__centroids_)
        inertia: float = self.__calculate_inertia(X, labels, self.__centroids_)
        return -1.0 * inertia

    def preprocess_input(self, X: np.ndarray, copy=True) -> np.ndarray:
        """Preprocess Input

        Args:
            X (np.ndarray): (n_samples, n_features) raw input data
            copy (bool, optional): Whether or not to modify in-place during inference call. Defaults to True.

        Returns:
            X (np.ndarray): (n_samples, n_components) preprocessed input
        """
        return self.__preprocess_input(X, is_train=False, copy=copy)

    def get_params(self, deep: bool = False):
        """
        Get parameters for this estimator.

        Args:
            deep (bool): If True, will return the parameters for this estimator and contained subobjects that are estimators. Default=True.

        Returns:
            params (dict): Parameter names mapped to their values.
        """

        params: Dict = super().get_params()
        return params

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects (such as :class:`~sklearn.pipeline.Pipeline`). The latter have parameters of the form ``<component>__<parameter>`` so that it's possible to update each component of a nested object.

        Parameters:
            **params (dict): Estimator parameters.

        Returns:
            self (SphericalKMeans): cloned estimator.
        """
        super().set_params(**params)
        self.__pca_ = PCA(n_components=self.n_components,
                          copy=self.copy,
                          whiten=self.whiten)
        if self.standarize:
            self.__std_scalar_ = StandardScaler(copy=self.copy)
        else:
            self.__std_scalar_ = None
        self.__centroids_ = None
        self.__n_components_ = -1
        self.__n_samples_ = -1
        self.__inertia_ = 0.0
        return self

    @property
    def pca_(self) -> PCA:
        """Deep copy of the PCA instance used for whitening."""
        return copy.deepcopy(self.__pca_)

    @property
    def std_scalar_(self) -> Union[StandardScaler, None]:
        """Deep copy of the StandardScalar instance used for standarization."""
        return copy.deepcopy(self.std_scalar_)

    @property
    def centroids_(self) -> np.ndarray:
        """Deep copy of the centroids of clusters."""
        return copy.deepcopy(self.__centroids_)

    @property
    def n_components_(self) -> int:
        """The estimated number of components."""
        return self.__n_components_

    @property
    def inertia_(self) -> float:
        """Sum of squared distances of samples to their closest cluster center."""
        return self.__inertia_

    # private
    def __init_centroids(self, random_state: RandomState) -> np.ndarray:
        """Initialize the centroids

        Randomly initialize the centoids from a standard normal distribution and then normalize to unit length.
        """
        # initialize from standard normal
        centroids: np.ndarray = random_state.standard_normal(
            size=[self.__n_components_, self.n_clusters])
        # normalize to unit length
        centroids = normalize(centroids, axis=0, norm="l2", copy=False)
        return centroids

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
            X = normalize(X, norm="l2", axis=1, copy=self.copy)
        # each features in the dataset has zero mean and unit variance; dataset level
        if self.standarize:
            if is_train == False:
                X = self.__std_scalar_.transform(X, copy=copy)
            else:
                X = self.__std_scalar_.fit_transform(X)
        # PCA whiten
        if is_train == False:
            X = self.__pca_.transform(X)
        else:
            X = self.__pca_.fit_transform(X)
        return X

    def __update_centroids(self, X: np.ndarray,
                           centroids: np.ndarray) -> np.ndarray:
        """Update Centroids

        Args:
            X (np.ndarray): (n_samples, n_components)
            centroids (np.ndarray): (n_components, n_clusters) Centroids used to calculate inertia.

        Returns:
            centroids (np.ndarray): 
        """
        # # centroid.shape = (n_components, n_clusters)
        # # X.shape = (n_samples, n_components)
        # # S_proj.shpae = (n_samples, n_clusters) each sample's projection on each cluster
        S_proj, labels = self.__calculate_projections_labels(X, centroids)
        # S_code.shpae = (n_samples, cluster)
        S_code: np.ndarray = np.zeros_like(S_proj)
        S_code[np.arange(self.__n_samples_),
               labels] = S_proj[np.arange(self.__n_samples_), labels]
        # update centroids
        centroids = np.matmul(X.transpose(), S_code) + centroids
        # normalize centroids
        centroids = normalize(centroids, axis=0, norm="l2", copy=False)
        return centroids

    def __calculate_projections(self, X: np.ndarray,
                                centroids: np.ndarray) -> np.ndarray:
        """Project X onto self._centroids_.

        Args:
            X (np.ndarray): (n_samples, n_components) New Data to transform
            centroids (np.ndarray): (n_components, n_clusters) Centroids used to project input data.

        Returns:
            S_proj (np.ndarray): (n_samples, n_clusters) Input X's projection onto dictionary
        """
        S_proj: np.ndarray = np.matmul(X, centroids)
        return S_proj

    def __calculate_projections_labels(
            self, X: np.ndarray,
            centroids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Given X, calculate projection onto self._centoids and infer cluster label

        Args:
            X (np.ndarray): (n_samples, n_components) New Data to transform
            centroids (np.ndarray): (n_components, n_clusters) Centroids used to project input data.

        Returns:
            S_proj (np.ndarray): (n_samples, n_clusters) Input X's projection onto dictionary
            labels (np.ndarray): (n_samples) The cluster labels that X is assigned to
        """
        # centroid.shape = (n_components, n_clusters)
        # X.shape = (n_samples, n_components)
        # S_proj.shpae = (n_samples, n_clusters) each sample's projection on each cluster
        S_proj: np.ndarray = self.__calculate_projections(X, centroids)
        labels: np.ndarray = np.argmax(S_proj, axis=1)
        return S_proj, labels

    def __calculate_inertia(self, X: np.ndarray, labels: np.ndarray,
                            centroids: np.ndarray) -> float:
        """Calculate inertia

        Args:
            X (np.ndarray): (n_samples, n_components) New data to transform.
            labels (np.ndarray): (n_samples) The cluster labels that X is assigned to.
            centroids (np.ndarray): (n_components, n_clusters) Centroids used to calculate inertia.

        Returns:
            inertia (float): The sum of squared distances of samples to their closest cluster center.
        """
        X_distance: np.ndarray = np.linalg.norm(np.transpose(X) -
                                                centroids[:, labels],
                                                axis=0)
        inertia: float = np.sum(np.square(X_distance))
        return inertia

    def __fit_kmeans(self, X: np.ndarray, random_state: Union[int, RandomState,
                                                              None]):
        """Fit Kmeans once given the random_state.

        Args:
            X (np.ndarray): New data to transform.
            random_state (Union[int, RandomState, None]): Random state used to initialize centroids.

        Returns:
            centroids (np.ndarray): Fitted cluster centroids.
            inertia (float): Sum of squared distances of samples to their closest cluster center.
        """
        centroids = self.__init_centroids(check_random_state(random_state))
        avg_centoids_shift: float = np.inf
        iter: int = 0
        while iter < self.max_iter and avg_centoids_shift > self.tol:
            # centroid.shape = (n_components, n_clusters)
            prev_centroids: np.ndarray = np.copy(centroids)
            centroids = self.__update_centroids(X, centroids)
            centroids_shift: np.ndarray = np.linalg.norm(prev_centroids -
                                                         centroids,
                                                         axis=0)
            avg_centoids_shift: float = np.mean(centroids_shift)
            iter = iter + 1
        _, labels = self.__calculate_projections_labels(X, centroids)
        inertia: float = self.__calculate_inertia(X, labels, centroids)
        return centroids, inertia

    def __fit_kmeans_single_process(self, X: np.ndarray,
                                    random_states: List[Union[int, RandomState,
                                                              None]],
                                    ret_queue: mp.Queue):
        """Use singel process to restart KMeans len(random_states) times.

        Given a list of random_states, exaust all the random_state with self.__fit_kmeans and put the result back to ret_queue.

        Args:
            X (np.ndarray): New data to transform.
            random_states (List[Union[int, RandomState, None]]): A List of random state used to initialize centroids for each fit_kmeans.
            ret_queue (mp.Queue): mp.Queue used to store result.
        """
        centroids_list: List[np.ndarray] = list()
        inertia_list: List[float] = list()
        for random_state in random_states:
            centroids, inertia = self.__fit_kmeans(X, random_state)
            centroids_list.append(centroids)
            inertia_list.append(inertia)
        ret_queue.put((centroids_list, inertia_list))

    def __fit_kmeans_processes(
            self, args_list: List[Tuple[np.ndarray, int, mp.Queue]],
            ret_queue: mp.Queue) -> Tuple[List[np.ndarray], List[float]]:
        """Use multiple process to fit KMeans len(args_list) times.

        Given a list of arguments to self.__fit_kmeans_single_process, exaust all the args with self.__fit_kmeans_single_process and return (List[centroids], List[inertia]).

        Args:
            args_list (List[Tuple[np.ndarray, int, mp.Queue]]): A list of arguments used to execute self.__fit_kmeans_single_process
            ret_queue (mp.Queue): mp.Queue used to store result.

        Returns:
            centroids (List[np.ndarray]): List of fitted cluster centroids.
            inertia (List[float]): List of inertia.
        """
        processes: List[mp.Process] = list()
        for args in args_list:
            p = mp.Process(target=self.__fit_kmeans_single_process, args=args)
            p.start()
            processes.append(p)
        centroids_list: List[np.ndarray] = list()
        inertia_list: List[float] = list()
        for p in processes:
            p.join()
            centroids, inertia = ret_queue.get()
            centroids_list.extend(centroids)
            inertia_list.extend(inertia)
        return centroids_list, inertia_list

    def __get_n_usable_processes(self) -> int:
        """Get usable processes.

        Returns:
            int: Usable processes.
        """
        max_n_processes: int = len(os.sched_getaffinity(0))
        if self.n_processes == None:
            return max_n_processes
        elif self.n_processes < max_n_processes:
            return self.n_processes
        return max_n_processes
