# About
This is small application that implements improved version of neurofilter outliers detection from [this paper](https://www.warse.org/IJATCSE/static/pdf/file/ijatcse139922020.pdf)

To see example go to ipynb notebooks.

To launch it, install venv, install packages from `requirements.txt`

Launch `outlier_regression.ipynb` or `outlier_classification.ipynb` and see results

# How to clean data
## 1. Define model for your data 
It could be any model that implements `sklearn.base.ClassifierMixin` or `sklearn.base.RegressorMixin` interface.
```py
# assume X,y is your regression/classification data
from common import fit_XGB_model
model = fit_XGB_model(X,y,task="regression")
```
## 2. Ensure that your model is already performing reasonable-well on your dirty data

for regression
```py
from sklearn.model_selection import RepeatedKFold, cross_val_score
import sklearn.metrics as metrics
from common import cross_val_score_mean_std

cv = RepeatedKFold(n_splits=5, n_repeats=2)
scoring = metrics.make_scorer(metrics.r2_score)
cleaned_data_score=cross_val_score(model,X,y,cv=cv,scoring=scoring)
# it will print cross-validated mean and std of r2 metric score on your data
cross_val_score_mean_std(cleaned_data_score,'label name')
```

for classification
```py
from common import cross_val_classification_report
from sklearn.model_selection import RepeatedStratifiedKFold

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)
report = cross_val_classification_report(
    model=model,
    X=X,
    y=y,
    cv=cv,
    target_names=['class 1','class 2','...']
)
# it will print classification report from model cross-validation
print(report)
```

You need to have at least some reasonable metrics on your dataset with some initial model.

## 3. Find outliers
```py
from common import find_outliers

X_numpy = np.array(X)
y_numpy = np.array(y)

# how many samples you want to remove
outliers_to_remove=0.2

outliers_mask, pred_loss, score = find_outliers(
    X_numpy,
    y_numpy,
    model,
    outliers_to_remove=outliers_to_remove,
    iterations=5,
    gamma=0.9,
    evaluate_loss=metrics.mean_absolute_error,
    cv=5,
    repeats=3
)

# true fraction of removed outliers
removed_outliers=sum(outliers_mask)/len(X)
```

This code will find outliers in a dataset, using provided model to iteratively filter out bad samples from dataset. 

`outliers_mask` - `True` if sample at given index is considered outlier

`pred_loss` - array of losses for each sample from it's predicted value. 
Higher losses indicates that given sample is more likely to be an outlier.

`score` - resulting model score(you can ignore it)

Amount of removed outliers can differ from requested amount of outliers your provided when algorithm do not see any improvements in model performance after removal of elements.

For parameters description see `find_outliers` method.

## 4. Check that your dataset is cleaner than before
You can do this by recomputing your metrics on clean data
```py
X_clean = np.array(X)[~outliers_mask]
y_clean = np.array(y)[~outliers_mask]

# repeat step 2 with clean data
```