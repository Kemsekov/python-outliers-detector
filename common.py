
import numpy as np


def XGB_search_params():
    params = {
        'max_depth':np.arange(1,6),
        'max_leaves':[15,20,25,30,40],
        'n_estimators':[2,5,10,15,20,40],
        'learning_rate':[0.01,0.05,0.1,0.3],
        'subsample':[0.1,0.2,0.5,0.7], 
        'colsample_bytree':[0.1,0.3,0.5,0.8,0.95],
        'min_child_weight': [1,3, 5,7, 10],
        'reg_lambda':[0.4, 0.6, 0.8, 1, 1.2, 1.4],
        'gamma': [0, 0.5, 1, 1.5, 2, 2.5, 5],
    }
    return params

from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict, cross_validate

# scoring = metrics.mean_absolute_error
def run_iteration(X,y,model,pred_scoring,evaluate_scoring,cv=5,repeats=3,seed = 42):
    total_errors = []
    pred_scores = []
    for i in range(repeats):
        shuffle = np.arange(0,len(y))
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

