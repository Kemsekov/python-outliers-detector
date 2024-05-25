
from matplotlib import pyplot as plt
import numpy as np




from sklearn import metrics
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict, cross_validate

def XGB_search_params():
    params = {
        # 'max_bin': [256,512,1024],
        'max_depth':np.arange(2,12,2),
        # 'max_leaves':[15,20,25,30,40],
        'n_estimators':np.arange(2,80,20),
        # 'learning_rate':[0.01,0.05,0.1,0.3],
        'colsample_bytree':np.random.uniform(0,1,size=5),
        'min_child_weight': np.arange(0,10,2),
        'reg_alpha' : np.arange(0,100,20),
        'reg_lambda':np.random.uniform(0,1,size=5),
        'gamma': np.arange(1,9),
    }
    return params

def cross_val_score_mean_std(scores,name):
    print(f"-----------{name}-----------")
    print("Mean ",np.mean(scores))
    print("Std ",np.std(scores))

def cross_val_scores(X,y,model : ClassifierMixin|RegressorMixin,evaluate_scoring,cv=5,repeats=3,seed = 42, fit_params : dict = None):
    """
    Computes cross-validated scores for each sample and total model error.

    X: independent features

    y: dependent features for regression or for classification

    model: classification or regression model that implements `ClassifierMixin` or `RegressorMixin`

    evaluate_scoring: function that returns score from true and predicted values. see `sklearn.metrics`

    cv: cross-validation used for cross_val_predict. Use integers

    repeats: how many times repeat cross validation predictions on shuffled data

    seed: random seed

    Returns:
    mean of prediction scores, also normalized relative to class occurrence (for classification)
    total mean evaluation score for model from all folds/repeats
    """
    total_errors = []
    pred_scores = []
    shuffle = np.arange(0,len(y))

    is_classification = isinstance(model,ClassifierMixin)
    if is_classification:
        pred_method = "predict_proba"
    else:
        pred_method = "predict"

    # count a size of each class as a fraction relative to total data length
    if is_classification:
        classes, counts = np.unique(y,return_counts=True)
        classes_counts=np.zeros(shape=(np.max(classes)+1))
        for class_,count in zip(classes,counts):
            classes_counts[class_]=count/len(y)
    
    pred_indices = np.arange(len(y))
    y_ones = np.ones_like(y)
    inv_shuffle=np.zeros_like(shuffle)

    for i in range(repeats):
        np.random.seed(i+seed)
        np.random.shuffle(shuffle)
        y_shuffled = y[shuffle]
        X_shuffled = X[shuffle]
        pred=cross_val_predict(model,X_shuffled,y_shuffled,cv=cv,method=pred_method,n_jobs=-1,fit_params=fit_params)
        if is_classification:
            # compute errors relative to each class size, so smaller classes will have greater impact on total
            # prediction error, so algorithm will 
            # pred_score = (1-pred[pred_indices,y_shuffled])**2 * classes_counts[y_shuffled]
            true_class_pred=pred[pred_indices,y_shuffled]

            pred_score = (1-true_class_pred)**2
        else:
            pred_score= (y_shuffled-pred)**2
            # pred_score = [pred_loss([a],[b]) for a,b in zip(y_shuffled,pred)]
        
        pred_score=np.array(pred_score)

        if is_classification:
            total_error=evaluate_scoring(y_ones,true_class_pred)
        else:
            total_error=evaluate_scoring(y_shuffled,pred)
        
        inv_shuffle[:]=0
        inv_shuffle[shuffle]=pred_indices

        total_errors.append(total_error)
        pred_scores.append(pred_score[inv_shuffle])
    
    pred_scores=np.array(pred_scores)
    total_error=np.array(total_error)
    return np.mean(pred_scores,axis=0),np.mean(total_errors)

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
        iterations = 3,
        gamma = 0.2,
        evaluate_loss=metrics.mean_squared_error,
        cv=6,
        repeats=3,
        seed = 42,
        plot=False,
        elements_to_plot=40):
    """
    Finds outliers in a data by repeatedly fitting a special model and selecting samples with worst prediction performance as outliers.
    
    X: input 2-dim data

    y: output 1-dim data

    special_model: model which is used to determine samples with highest error

    outliers_to_remove: how much outliers to remove from dataset
    
    repeats: integer, how many cross-validations to do. Each repeat shuffles data runs cross-validation on it again and
    then algorithm averages predictions from all such repeats.
    
    iterations: how many iterations to do of algorithm
    
    gamma: how much decrease `outlier_remove_partition` on each next iteration

    seed: algorithm random seed

    plot: render results or not
    
    Returns: array of outlier indices, total score of model prediction with given outliers removed
    """

    outlier_remove_partition=outliers_to_remove*(gamma-1)/(gamma**iterations-1)

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
            if plot: print("Removed all outliers. Stopping execution.")
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
    outliers_indices = np.nonzero(outliers_mask)[0]
    return outliers_indices, eval_score

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

