
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit

def kernel_pca_scorer(estimator,X,y=None):
    """Computes r2 score of how good estimator can describe data in low-dimensions"""
    X_reduced = estimator.transform(X)
    X_preimage = estimator.inverse_transform(X_reduced)
    score = r2_score(X, X_preimage)

    if np.isnan(score): score = 0
    return score

class KernelPCASearchCV:
    def __init__(self,n_components,n_iter=-1,scaler = None) -> None:
        """
        n_components: components to use for pca
        n_iter: iterations to do on parameters search. Put -1 to use all parameters space
        scaler: data scaler to use. `None` to not scale
        """
        self.kpca = KernelPCA(fit_inverse_transform=True,n_components=n_components, n_jobs=-1) 
        """Kernel pca"""
        self.score : float = float('nan')
        """r2 score of kernel pca performance"""
        self.scaler = scaler
        """Scaler used for data"""
        self.param_grid = {
            "gamma": np.linspace(0.01, 1.5, 10),
            "kernel": ["rbf", "sigmoid", "poly"],
            "alpha":[0.00001,0.001,0.1,1]
        }
        max_iter = \
            len(self.param_grid['gamma'])*\
            len(self.param_grid['kernel'])*\
            len(self.param_grid['alpha'])
        self.n_iter = np.min([n_iter,max_iter])
        if self.n_iter<=0: self.n_iter=max_iter

    def fit(self,X):
        features = X.shape[1]

        # make gamma parameter to be in a relative size to features dimensions
        # so we scales around mean of gamma = 1/features
        g_param = self.param_grid.copy()
        g_param['gamma']=g_param['gamma']/features


        # create one single split for whole data
        test_fold = np.zeros(X.shape[0])
        test_fold[:] = -1
        test_fold[-1] = 0
        cv=PredefinedSplit(test_fold)
        # use whole dataset for scoring
        X_input = X
        def local_kernel_pca_scorer(estimator,X,y=None):
            return kernel_pca_scorer(estimator,X_input,None)
        scoring=local_kernel_pca_scorer

        grid_search = RandomizedSearchCV(
            self.kpca, 
            g_param, 
            cv=cv,
            n_iter=self.n_iter,
            n_jobs=-1, 
            scoring=scoring)
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
        grid_search.fit(X)
        self.kpca = grid_search.best_estimator_
        self.score = grid_search.best_score_
    
    def fit_transform(self,X):
        self.fit(X)
        return self.kpca.transform(X)

    def transform(self,X):
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.kpca.transform(X)

    def inverse_transform(self,X):
        X_inv = self.kpca.inverse_transform(X)
        if self.scaler is not None:
            X_inv = self.scaler.inverse_transform(X_inv)
        return X_inv
