
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import RobustScaler

def kernel_pca_scorer(estimator,X,y=None):
    """Computes r2 score of how good estimator can describe data in low-dimensions"""
    X_reduced = estimator.transform(X)
    X_preimage = estimator.inverse_transform(X_reduced)
    score = r2_score(X, X_preimage)

    if np.isnan(score): score = 0
    return score

class KernelPCASearchCV:
    def __init__(self,n_components,cv=3,n_iter=200,scaler = RobustScaler()) -> None:
        self.kpca = KernelPCA(fit_inverse_transform=True,n_components=n_components, n_jobs=-1) 
        """Kernel pca"""
        self.score : float = float('nan')
        """r2 score of kernel pca performance"""
        self.scaler = scaler
        """Scaler used for data"""
        self.cv=cv
        self.param_grid = {
            "gamma": np.linspace(0.2, 4, 10),
            "kernel": ["rbf", "sigmoid", "poly"],
            "alpha":[0.001,0.1,1,5]
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

        grid_search = RandomizedSearchCV(
            self.kpca, 
            g_param, 
            cv=self.cv,
            n_iter=self.n_iter,
            n_jobs=-1, 
            scoring=kernel_pca_scorer)
        X_scaled = self.scaler.fit_transform(X)
        self.__X_scaled = X_scaled
        grid_search.fit(X_scaled)
        self.kpca = grid_search.best_estimator_
        self.score = grid_search.best_score_
    
    def fit_transform(self,X):
        self.fit(X)
        return self.kpca.transform(self.__X_scaled)

    def transform(self,X):
        X_scaled = self.scaler.transform(X)
        return self.kpca.transform(X_scaled)

    def inverse_transform(self,X):
        X_inv = self.kpca.inverse_transform(X)
        return self.scaler.inverse_transform(X_inv)
