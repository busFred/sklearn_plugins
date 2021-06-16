"""[summary]
"""
from typing import Union
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA


class SphericalKMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    """[summary]

    """
    # private
    __pca: PCA

    # public:
    n_clusters: int
    n_components: Union[int, float, str, None]
    whiten: bool
    normalize: bool
    standarize: bool
    max_iter: int
    tol: float
    copy: bool

    def __init__(self,
                 n_clusters: int = 500,
                 n_components: Union[int, float, str, None] = 0.8,
                 normalize: bool = True,
                 standarize: bool = True,
                 whiten: bool = True,
                 max_iter: int = 100,
                 tol: float = 0.01,
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
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.normalize = normalize
        self.standarize = standarize
        self.whiten = whiten
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy
        self.__pca = PCA(n_components=n_components, copy=copy, whiten=whiten)

    pass
