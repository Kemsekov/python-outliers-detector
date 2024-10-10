
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict, cross_val_score
import numpy as np
from sklearn.model_selection import BaseCrossValidator, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

def test_ridge_fit(X,y,degree=3):
    """
    Prints r2 score mean and std of ridge fit on model with data transformed by given degree polynomial features.
    
    The better(cleaner) given data is, the higher will be output metric of given method
    """
    scaler = StandardScaler()
    X_n=scaler.fit_transform(X)
    model = make_pipeline(PolynomialFeatures(degree=degree), Ridge())

    r2_scoring = metrics.make_scorer(metrics.r2_score)
    print("r2 score of poly-features ridge regression")
    cross_val_score_mean_std(
        cross_val_score(
            model,
            X_n,
            y,
            cv=5,
            scoring=r2_scoring
        ),
        y.name)

def XGB_search_params():
    params = {
        # 'max_bin': [256,512,1024],
        'max_depth':        [2,4,6,8,10],
        'colsample_bytree': [1, 0.8, 0.6, 0.4],
        'min_child_weight': [1,2,5,7],
        'gamma':            [0,2,5,15],
        'eta':              [0.1,0.3,0.5],
        'lambda':           [0.2,1,2]
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

def _classes_count(y):
    classes = np.unique(y)
    classes_count = np.zeros(np.max(classes)+1)
    for cls in classes:
        classes_count[cls]=np.count_nonzero(y==cls)
    return classes_count

def cross_val_scores_classification(
        X,y,
        model : ClassifierMixin,
        evaluate_scoring,
        cv=5,
        repeats=3,
        seed = 42, 
        fit_params : dict = None,
        class_weight_scale_power = 0.8
    ):
    """
    Computes samples error from model class prediction using repeated cross-validation.
    For a model that have well fitted hyperparameters and is capable of learning
    underlying data relation, this method provides a robust way to measure how much
    each sample is off from general data distribution.
    Resulting scores then could be used to find outliers in a data and remove them.

    class_weight_scale_power: how much to consider class size in computing scores for samples.
    0 means to not consider class sizes in scores, 1 means linearly consider class sizes in scores

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

    classes_count = _classes_count(y)
    scale = np.power(classes_count[y],class_weight_scale_power)

    for i in range(repeats):
        np.random.seed(i+seed)
        np.random.shuffle(shuffle)
        y_shuffled = y[shuffle]
        X_shuffled = X[shuffle]
        scale_shuffled = scale[shuffle]
        pred=cross_val_predict(
            model,
            X_shuffled,
            y_shuffled,
            cv=cv,
            method=pred_method,
            n_jobs=-1,
            fit_params=fit_params)
        
        true_class_pred=pred[pred_indices,y_shuffled]

        pred_score = (2-true_class_pred)**2*scale_shuffled

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
        fit_params : dict = None,
        class_weight_scale_power = 0.5
        ):
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

    class_weight_scale_power: how much to consider class size in computing scores for samples.

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
            fit_params,
            class_weight_scale_power)
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
        gamma = 0.9,
        evaluate_loss=metrics.mean_squared_error,
        cv=5,
        repeats=3,
        class_weight_scale_power = 0.5,
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

    class_weight_scale_power: how much to consider class size in computing scores for samples. Use this parameter to balance removed samples count for each class. 0 means to not consider class sizes at all, 1 means linearly consider class sizes in scores.

    seed: algorithm random seed

    plot: render results or not
    
    Returns: array mask this is true where outlier is found, total score of model prediction with given outliers removed
    """
    if gamma<=0 or gamma>=1:
        raise ValueError("gamma must be in range (0;1)")
    
    if outliers_to_remove<=0 or outliers_to_remove>1:
        raise ValueError("outliers_to_remove must be in range (0;1]")

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
            seed=seed,
            class_weight_scale_power=class_weight_scale_power)
        if plot: print("Evaluate score ",eval_score)
        
        if eval_score>prev_eval_score: 
            if plot: print("Increase in total error. Reverting previous and stopping...")
            outliers_mask[prev_outliers]=False
            break
            pass

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
    def color_scale(x): 
        minx = np.min(x,axis=0)
        maxx = np.max(x,axis=0)
        scale = 1/(maxx-minx)
        return ((x - minx)*scale * 255).astype(int)

    for cls,color in zip(classes,colors):
        results[y==cls]=color
    return color_scale(results)
