# About
This is small application that implements outliers detection from [this paper](https://www.warse.org/IJATCSE/static/pdf/file/ijatcse139922020.pdf)

# Setup
1. `python 3.11`

2. `python -m venv ./venv`

3. source binaries like 

    `source venv/bin/activate.bash`

4. `pip install -r requirements.txt`
5. Run 

    `python gui.py`

# How it works
Open dataset in csv format
`choose dataset`

set parameters

`count of models` - count of models to build at the same time.
It is a good idea to use many at the same time, because it averages error over samples and will give you
more robust representation of sample errors.

`xi` - value between 0 and 1 that can be used to set min and max amount of neurons of models.
Higher `xi` makes model to prone to overfit but increases it's capacity

`test split` - splits data into tasting and training for building models

`elements to show` - how many elements to show on the graph.

`input dimensions` - size of input dimensions for models.

`rows to remove` - a list of whole numbers of rows to remove (possible outliers)

Buttons

`run iteration` - builds `count of models` models, find error on all samples and averages this error over all models
and plots first `elements to show` samples with highest error, so you can see and deduce which are outliers.
Also it plots average total error of models dependent on iteration index, so you can see how your error decreases as you remove more and more outliers.

`save results` - saves filtered from outliers data and outliers into different csv files in the same directory where dataset csv file is located.


