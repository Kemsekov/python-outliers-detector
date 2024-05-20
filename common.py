
from matplotlib import pyplot as plt
import numpy as np




from sklearn import metrics
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict, cross_validate

def XGB_search_params():
    params = {
        # 'max_bin': [256,512,1024],
        'max_depth':np.arange(2,12),
        # 'max_leaves':[15,20,25,30,40],
        'n_estimators':np.arange(2,80,20),
        # 'learning_rate':[0.01,0.05,0.1,0.3],
        'colsample_bytree':np.random.uniform(0,1,size=5),
        'min_child_weight': np.arange(0,10,2),
        'reg_alpha' : np.arange(0,100,10),
        'reg_lambda':np.random.uniform(0,1,size=5),
        'gamma': np.arange(1,9),
    }
    return params

def cross_val_score_mean_std(scores,name):
    print(f"-----------{name}-----------")
    print("Mean ",np.mean(scores))
    print("Std ",np.std(scores))

def cross_val_scores(X,y,model : ClassifierMixin|RegressorMixin,pred_loss,evaluate_scoring,cv=5,repeats=3,seed = 42):
    """
    Computes cross-validated scores for each sample and total model error.

    X: independent features

    y: dependent features for regression or for classification

    model: classification or regression model that implements `ClassifierMixin` or `RegressorMixin`
    
    pred_loss: function that computes loss from true and predicted values. see `sklearn.metrics`

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
    
    for i in range(repeats):
        np.random.seed(i+seed)
        np.random.shuffle(shuffle)
        y_shuffled = y[shuffle]
        X_shuffled = X[shuffle]
        print("cross val predict")
        pred=cross_val_predict(model,X_shuffled,y_shuffled,cv=cv,method=pred_method,n_jobs=-1)
        print("filling pred_score...")
        if is_classification:
            def class_vector(expected,actual): 
                v = np.zeros_like(actual)
                v[expected]=1
                return v
            # compute errors relative to each class size, so smaller classes will have greater impact on total
            # prediction error
            
            # pred_class is vector
            pred_score = [pred_loss(class_vector(true_class,pred_class),pred_class)/classes_counts[true_class] for true_class,pred_class in zip(y_shuffled,pred)]
        else:
            pred_score= (y_shuffled-pred)**2
            # pred_score = [pred_loss([a],[b]) for a,b in zip(y_shuffled,pred)]
        
        pred_score=np.array(pred_score)
        print("evaluate scoring...")

        if is_classification:
            diff_from_true_class=np.array([p[c] for p,c in zip(pred,y_shuffled)])
            total_error=evaluate_scoring(np.ones_like(y_shuffled),diff_from_true_class)
        else:
            total_error=evaluate_scoring(y_shuffled,pred)
        print("inv shuffle")
        inv_shuffle=np.zeros_like(shuffle)
        inv_shuffle[shuffle]=np.arange(len(shuffle))

        total_errors.append(total_error)
        pred_scores.append(pred_score[inv_shuffle])
    
    pred_scores=np.array(pred_scores)
    total_error=np.array(total_error)
    print("return")    
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
        outlier_remove_partition = 0.05,
        pred_loss=metrics.mean_absolute_error,
        evaluate_loss=metrics.mean_squared_error,
        cv=6,
        repeats=3,
        iterations = 5,
        seed = 42,
        max_stack_count=2,
        plot=False,
        elements_to_plot=25):
    """
    Finds outliers in a data by repeatedly fitting a special model and selecting samples with worst prediction performance as outliers.
    
    X: input 2-dim data

    y: output 1-dim data

    special_model: model which is used to determine samples with highest error
    outlier_remove_partition: which fraction of left non-outlier samples to remove in each iteration
    
    pred_loss: loss used for samples. Higher values means sample is more likely to be an outlier
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
        if plot:
            print("computing cross_val_scores...")

        pred_loss_values, eval_score = cross_val_scores(
            X=X_cleaned,
            y=y_cleaned,
            model=special_model,
            pred_loss=pred_loss,
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

        indices = np.argsort(-pred_loss_values)
        to_remove_count = int(outlier_remove_partition*len(indices))
        if to_remove_count==0: break

        prev_outliers=indices[:to_remove_count]
        outliers=np.concatenate([outliers,prev_outliers])

        if not plot: continue
        indices=indices[:elements_to_plot]
        x=np.arange(0,len(indices))
        plt.figure(figsize=(8,5))
        plt.plot(x,pred_loss_values[indices])
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

