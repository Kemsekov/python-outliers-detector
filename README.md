# About
This is small application that implements neurofilter outliers detection from [this paper](https://www.warse.org/IJATCSE/static/pdf/file/ijatcse139922020.pdf)

See video how it works [here](https://youtu.be/H-FMqUopDFw)
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
All first `k` values in `csv` file is treated as input values for models,
meanwhile all remaining values are model output that they must predict.
You can set this value to size of output dimensions, so default `-1` means that
all values except the last one in csv file is input values, meanwhile the least one
is regression value that we need to predict.

`rows to remove` - a list of whole numbers of rows to remove (possible outliers)

`N rows to remove` - a number of samples with highest error to remove



Buttons

`run iteration` - builds `count of models` models, find error on all samples and averages this error over all models
and plots first `elements to show` samples with highest error, so you can see and deduce which are outliers.
Also it plots average total error of models dependent on iteration index, so you can see how your error decreases as you remove more and more outliers.

`plot` - replot the last iteration results with updated values `show indices` checkbox and `elements to show`

`save results` - saves filtered from outliers data and outliers into different csv files in the same directory where dataset csv file is located.

![image](https://github.com/Kemsekov/python-outliers-detector/assets/57869319/08b677c7-4b8e-40fc-9dcd-51ccc48a46a5)
![image](https://github.com/Kemsekov/python-outliers-detector/assets/57869319/6f277f19-4e1c-43b5-ae22-8f9ec07d4400)
![image](https://github.com/Kemsekov/python-outliers-detector/assets/57869319/6cea5a85-a048-4870-8a8e-cd59a1e139d1)
![image](https://github.com/Kemsekov/python-outliers-detector/assets/57869319/a51f5b57-69c4-45fc-8f2f-5bebc4dff411)


# Change model
in file `functions/build_model.py` we have sole method `build_model` that is used to build model which is gonna be used to create models which computes error vector of samples.

By default it is using 2 layers with same setup as [described in this paper](https://www.warse.org/IJATCSE/static/pdf/file/ijatcse139922020.pdf), but you can change it.

# Error computation.

By default cross validation on test data split on all models is used.


