
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict

def XGB_search_params():
    params = {
        # 'max_bin': [256,512,1024],
        'max_depth':        np.random.randint(2,12,size=5),
        # 'max_leaves':[15,20,25,30,40],
        'n_estimators':     np.random.randint(2,120,size=10),
        # 'learning_rate':[0.01,0.05,0.1,0.3],
        'colsample_bytree': np.random.uniform(0,1,size=5),
        'min_child_weight': np.random.randint(0,15,size=5),
        'reg_alpha' :       np.random.randint(0,120,size=10),
        'reg_lambda':       np.random.uniform(0,1,size=5),
        'gamma':            np.random.randint(1,12,size=5),
    }
    return params

def cross_val_score_mean_std(scores,name):
    print(f"-----------{name}-----------")
    print("Mean ",np.mean(scores))
    print("Std ",np.std(scores))

def cross_val_scores_regression(
        X,y,
        model : RegressorMixin,
        evaluate_scoring,
        cv=5,
        repeats=3,
        seed = 42, 
        fit_params : dict = None):
    """
    Computes samples error from model prediction using repeated cross-validation.
    For a model that have well fitted hyperparameters and is capable of learning
    underlying data relation, this method provides a robust way to measure how much
    each sample is off from general data distribution.
    Resulting scores then could be used to find outliers in a data and remove them.

    Returns:
    mean errors on each sample, mean total model error from all repeats
    """
    total_errors = []
    pred_scores = []
    shuffle = np.arange(0,len(y))

    pred_method = "predict"

    pred_indices = np.arange(len(y))
    inv_shuffle=np.zeros_like(shuffle)

    for i in range(repeats):
        np.random.seed(i+seed)
        np.random.shuffle(shuffle)
        y_shuffled = y[shuffle]
        X_shuffled = X[shuffle]
        pred=cross_val_predict(
            model,
            X_shuffled,
            y_shuffled,
            cv=cv,
            method=pred_method,
            n_jobs=-1,
            fit_params=fit_params)
        pred_score= (y_shuffled-pred)**2
        total_error=evaluate_scoring(y_shuffled,pred)
        inv_shuffle[shuffle]=pred_indices

        total_errors.append(total_error)
        pred_scores.append(pred_score[inv_shuffle])
    return np.mean(pred_scores,axis=0),np.mean(total_errors)

def cross_val_scores_classification(
        X,y,
        model : ClassifierMixin,
        evaluate_scoring,
        cv=5,
        repeats=3,
        seed = 42, 
        fit_params : dict = None
    ):
    """
    Computes samples error from model class prediction using repeated cross-validation.
    For a model that have well fitted hyperparameters and is capable of learning
    underlying data relation, this method provides a robust way to measure how much
    each sample is off from general data distribution.
    Resulting scores then could be used to find outliers in a data and remove them.

    Returns:
    mean errors on each sample, mean total model error from all repeats
    """
    total_errors = []
    pred_scores = []
    shuffle = np.arange(0,len(y))

    pred_method = "predict_proba"

    pred_indices = np.arange(len(y))
    y_ones = np.ones_like(y)
    inv_shuffle=np.zeros_like(shuffle)

    for i in range(repeats):
        np.random.seed(i+seed)
        np.random.shuffle(shuffle)
        y_shuffled = y[shuffle]
        X_shuffled = X[shuffle]
        pred=cross_val_predict(
            model,
            X_shuffled,
            y_shuffled,
            cv=cv,
            method=pred_method,
            n_jobs=-1,
            fit_params=fit_params)
        
        true_class_pred=pred[pred_indices,y_shuffled]
        pred_score = (1-true_class_pred)**2
        total_error=evaluate_scoring(y_ones,true_class_pred)
        inv_shuffle[shuffle]=pred_indices

        total_errors.append(total_error)
        pred_scores.append(pred_score[inv_shuffle])
    return np.mean(pred_scores,axis=0),np.mean(total_errors)

def cross_val_scores(
        X,y,
        model : ClassifierMixin|RegressorMixin,
        evaluate_scoring,
        cv=5,
        repeats=3,
        seed = 42, 
        fit_params : dict = None):
    """
    Computes samples error from model prediction using repeated cross-validation.
    For a model that have well fitted hyperparameters and is capable of learning
    underlying data relation, this method provides a robust way to measure how much
    each sample is off from general data distribution.
    Resulting scores then could be used to find outliers in a data and remove them.

    X: independent features

    y: dependent features for regression or for classification

    model: classification or regression model that implements `ClassifierMixin` or `RegressorMixin`

    evaluate_scoring: function that returns score from true and predicted values. see `sklearn.metrics`

    cv: cross-validation used for cross_val_predict. Use integers

    repeats: how many times repeat cross validation predictions on shuffled data

    seed: random seed

    Returns:
    mean errors on each sample, mean total model error from all repeats
    """

    is_classification = isinstance(model,ClassifierMixin)
    if is_classification:
        return cross_val_scores_classification(
            X,y,
            model,
            evaluate_scoring,
            cv,
            repeats,
            seed,
            fit_params)
    return cross_val_scores_regression(
        X,y,
        model,
        evaluate_scoring,
        cv,
        repeats,
        seed,
        fit_params)

def get_full_data(X,y):
    y_full_mask = ~np.isnan(y)
    X=X[y_full_mask]
    y=y[y_full_mask]
    return X,y

def negate(func):
    def negated(t,p,**args): return func(t,p,**args)
    return negated

def find_outliers(
        X,y,special_model,
        outliers_to_remove = 0.05,
        iterations = 5,
        gamma = 0.5,
        evaluate_loss=metrics.mean_squared_error,
        cv=5,
        repeats=3,
        seed = 42,
        plot=False,
        elements_to_plot=40):
    """
    Finds outliers in a data by repeatedly fitting a special model and selecting samples with worst prediction performance as outliers.
    
    X: input 2-dim data

    y: output 1-dim data

    special_model: model which is used to determine samples with highest error

    outliers_to_remove: how much outliers to remove from data
    
    repeats: integer, how many cross-validations to do. Each repeat shuffles data runs cross-validation on it again and
    then algorithm averages predictions from all such repeats.
    
    iterations: how many iterations to do of algorithm
    
    gamma: how much decrease amount of removed items on each next iteration

    seed: algorithm random seed

    plot: render results or not
    
    Returns: array mask this is true where outlier is found, total score of model prediction with given outliers removed
    """

    outlier_remove_partition=\
        outliers_to_remove*(gamma-1)/(gamma**iterations-1)

    prev_eval_score = float('inf')
    prev_outliers=[]

    outliers_mask = np.zeros_like(y,dtype=bool)
    for i in range(iterations):

        X_clean = X[~outliers_mask]
        y_clean = y[~outliers_mask]

        pred_loss_values, eval_score = cross_val_scores(
            X=X_clean,
            y=y_clean,
            model=special_model,
            evaluate_scoring=evaluate_loss,
            cv=cv,
            repeats=repeats,
            seed=seed)
        if plot: print("Evaluate score ",eval_score)
        
        if eval_score>prev_eval_score: 
            if plot: print("Increase in total error. Reverting previous and stopping...")
            outliers_mask[prev_outliers]=False
            break

        prev_eval_score=eval_score

        # list of true indices of clean data relative to original data
        clean_data_indices = np.where(~outliers_mask)[0].astype(int)
        # indices of samples sorted by prediction error in ascending order
        sorted_ind = np.argsort(-pred_loss_values)
        indices = clean_data_indices[sorted_ind]

        # elements to move to outliers
        to_remove = int(outlier_remove_partition*len(y))
        if to_remove==0: to_remove=1

        prev_outliers=indices[:to_remove]
        outliers_mask[prev_outliers]=True
        outlier_remove_partition*=gamma

        if plot:
            to_render=sorted_ind[:elements_to_plot]
            x=np.arange(0,len(to_render))
            plt.figure(figsize=(8,5))
            plt.plot(x,pred_loss_values[to_render])
            plt.xlabel("Sample")
            plt.ylabel("Prediction score")
            plt.show()
    return outliers_mask, eval_score

def cross_val_classification_report(model,X,y,cv, target_names = None):
    y_true = []
    y_pred = []
    for train,test in cv.split(X,y):
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]
        y_test = y[test]
        y_test_pred = model.fit(X_train,y_train).predict(X_test)
        y_true.append(y_test)
        y_pred.append(y_test_pred)

    y_true=np.concatenate(y_true)
    y_pred=np.concatenate(y_pred)

    return classification_report(y_true,y_pred,target_names=target_names)

def generate_colors_for_classification(y : np.ndarray,seed=42):
    """Returns a color-representation of array y, where each unique class replaced with color"""
    classes = np.sort(np.unique(y))
    np.random.seed(seed)
    colors = np.random.uniform(0,1,size=(len(classes),3))
    results = np.zeros((len(y),3))

    for cls,color in zip(classes,colors):
        results[y==cls]=color
    return results

from sklearn.metrics import r2_score, mean_absolute_error
def kernel_pca_scorer(estimator,X,y=None):
    """Computes r2 score of how good estimator can describe data in low-dimensions"""
    X_reduced = estimator.transform(X)
    X_preimage = estimator.inverse_transform(X_reduced)
    score = r2_score(X, X_preimage)

    if np.isnan(score): score = 0
    return score

class KernelPCAFitResult:
    def __init__(self,kpca : KernelPCA, r2: float, X_transform : np.ndarray, data_scaler : StandardScaler) -> None:
        self.kpca = kpca
        """Kernel pca"""
        self.r2 = r2
        """r2 score of kernel pca performance"""
        self.X_transform = X_transform
        """Transformed of scaled X to low dimensions data"""
        self.data_scaler = data_scaler
        """Scaler used for data"""

def optimalKernelPCA(X,n_components,cv=3,n_iter=200):
    """Finds optimal kernel PCA using hyperparameters search"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    param_grid = {
        "gamma": np.linspace(0.001, 1, 20),
        "kernel": ["rbf", "sigmoid", "poly"],
        "alpha":[0.1,1,3]
    }
    
    max_iter = len(param_grid['gamma'])*len(param_grid['kernel'])*len(param_grid['alpha'])
    n_iter = np.min([n_iter,max_iter])

    kpca=KernelPCA(fit_inverse_transform=True,n_components=n_components, n_jobs=-1) 
    grid_search = RandomizedSearchCV(
        kpca, 
        param_grid, 
        cv=cv,
        n_iter=n_iter,
        n_jobs=-1, 
        scoring=kernel_pca_scorer)
    grid_search.fit(X_scaled)
    kpca : KernelPCA= grid_search.best_estimator_
    X_transform = kpca.transform(X_scaled)
    
    kpca_r2_score = grid_search.best_score_
    return KernelPCAFitResult(kpca,kpca_r2_score, X_transform, scaler)
