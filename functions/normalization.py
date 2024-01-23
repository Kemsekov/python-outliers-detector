import numpy as np

class Normalization:
    def __init__(self,translation = None,scale = None) -> None:
        self.translation=translation
        self.scale=scale

    def fit(self,X : np.ndarray):
        self.translation = np.mean(X,axis=0)
        self.scale = np.std(X,axis=0)
        return self.transform(X)
    
    def transform(self,X: np.ndarray):
        return (X-self.translation)/self.scale
    
    def restore(self,X_transformed : np.ndarray):
        return (X_transformed*self.scale)+self.translation
