
from matplotlib import pyplot as plt
import numpy as np


def XGB_search_params():
    params = {
        'max_depth':np.arange(1,7),
        'max_leaves':[15,20,25,30,40],
        'n_estimators':[2,5,10,15,20,40],
        'learning_rate':[0.01,0.05,0.1,0.3],
        'subsample':[0.1,0.2,0.5,0.7],
        'colsample_bytree':[0.1,0.3,0.5,0.8,0.95],
        'min_child_weight': [1,3, 5, 7, 10],
        'reg_lambda':[0.4, 0.6, 0.8, 1, 1.2, 1.4],
        'gamma': [0, 0.5, 1, 1.5, 2, 2.5, 5],
    }
    return params

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict, cross_validate

# scoring = metrics.mean_absolute_error
def run_iteration(X,y,model,pred_scoring,evaluate_scoring,cv=5,repeats=3,seed = 42):
    total_errors = []
    pred_scores = []
    shuffle = np.arange(0,len(y))
    for i in range(repeats):
        np.random.seed(i+seed)
        np.random.shuffle(shuffle)
        y_shuffled = y[shuffle]
        X_shuffled = X[shuffle]

        pred=cross_val_predict(model,X_shuffled,y_shuffled,cv=cv)
        pred_score = [pred_scoring([a],[b]) for a,b in zip(y_shuffled,pred)]
        pred_score=np.array(pred_score)
        total_error=evaluate_scoring(y_shuffled,pred)
        
        inv_shuffle=np.zeros_like(shuffle)
        inv_shuffle[shuffle]=np.arange(len(shuffle))

        total_errors.append(total_error)
        pred_scores.append(pred_score[inv_shuffle])
    
    pred_scores=np.array(pred_scores)
    total_error=np.array(total_error)
    
    return np.mean(pred_scores,axis=0),np.mean(total_errors)

def find_outliers(
        X,y,special_model,
        outlier_remove_partition = 0.05,
        pred_scoring=metrics.mean_absolute_error,
        evaluate_loss=metrics.mean_squared_error,
        cv=6,
        repeats=3,
        iterations = 5,
        seed = 42,
        max_stack_count=2,
        plot=False):
    """
    Finds outliers in a data by repeatedly fitting a special model and selecting samples with worst prediction performance as outliers.
    
    X: input 2-dim data

    y: output 1-dim data

    special_model: model which is used to determine samples with highest error
    outlier_remove_partition: which fraction of left non-outlier samples to remove in each iteration
    
    pred_scoring: scoring used for samples. Higher values means sample is more likely to be an outlier
    evaluate_loss: loss used for model performance evaluation
    cv: integer, how many folds to do on cross-validations to do on model fitting
    
    repeats: integer, how many cross-validations to do. Each repeat shuffles data runs cross-validation on it again and
    then algorithm averages predictions from all such repeats.
    
    iterations: how many iterations to do
    
    seed: algorithm random seed
    
    max_stack_count: max count of non-improving iterations allowed before algorithm stops

    plot: render results or not
    
    Returns: array of outlier indices, total score of model prediction with given outliers removed
    """
    outliers=[]
    prev_outliers = []
    prev_score = float('inf')

    stack_count = 0

    for iteration in range(iterations):
        X_cleaned = [row for i,row in enumerate(X) if i not in outliers]
        y_cleaned = [row for i,row in enumerate(y) if i not in outliers]

        X_cleaned=np.array(X_cleaned)
        y_cleaned=np.array(y_cleaned)

        pred, eval_score = run_iteration(
            X=X_cleaned,
            y=y_cleaned,
            model=special_model,
            pred_scoring=pred_scoring,
            evaluate_scoring=evaluate_loss,
            cv=cv,
            repeats=repeats,
            seed=seed+iteration)
        if plot: print("evaluate score ",eval_score)

        if eval_score>prev_score: 
            outliers=[o for o in outliers if o not in prev_outliers]
            prev_outliers=[]
            outlier_remove_partition/=2
            stack_count+=1
            seed-=1
            if stack_count>=max_stack_count: break
            continue
        stack_count=0
        prev_score=eval_score

        indices = np.argsort(-pred)
        to_remove_count = int(outlier_remove_partition*len(indices))
        if to_remove_count==0: break

        prev_outliers=indices[:to_remove_count]
        outliers=np.concatenate([outliers,prev_outliers])

        if not plot: continue
        indices=indices[:25]
        x=np.arange(0,len(indices))
        plt.figure(figsize=(8,5))
        plt.plot(x,pred[indices])
        plt.xticks(x,labels=indices)
        plt.xlabel("Sample")
        plt.ylabel("Prediction score")
        plt.show()
    if plot: print("total removed ",len(outliers))
    return np.array(outliers,dtype=np.int32), eval_score

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

