# Import the Tkinter module
import os
import tkinter as tk
from tkinter import filedialog as fd
import numpy as np
import pandas as pd
from functions.utils import *
from functions.build_model import build_model
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

high_error_sorted_indices=[]

root = tk.Tk()
models_count = tk.IntVar(value=5)
xi = tk.DoubleVar(value=0)
test_split = tk.DoubleVar(value=0.1)
input_dimensions = tk.IntVar()
elements_to_show = tk.IntVar(value=10) # max elements to show on plot
rows_to_remove = tk.StringVar()
n_rows_to_remove = tk.IntVar()
show_indices = tk.BooleanVar(value=True)
avg_error : np.ndarray[np.float64] = None

working_dir=""

columns: pd.Index = None #dataset csv columns
iteration_errors = []
removed_rows = []
shuffled_data: np.ndarray[np.float32] # dataset copy that is used for shuffle

data: np.ndarray[np.float32] = None
normalization : Normalization

iteration_errors = []
removed_rows = []
def score_function(xi):
    models=[]
    for i in range(0,models_count.get()):
        np.random.shuffle(shuffled_data)
        # test size dependent on test_split
        test_size = int(len(data)*test_split.get())
        train = shuffled_data[test_size:]
        data_input_split=input_dimensions.get()
        models.append(build_model(train[:,:data_input_split],train[:,data_input_split:],xi))
    avg_error = average_prediction_error(data,models,data_input_split)
    avg_model_error = np.average(avg_error)
    return avg_model_error
def find_xi_grid_search():
    if data is None:
        print("Open dataset")
        return
    global xi
    scores = [score_function(xi_) for xi_ in np.arange(0, 0.5, 0.1)]
    best_score = min(scores)
    best_xi = np.arange(0.0, 0.5, 0.1)[scores.index(best_score)]
    xi.set(best_xi)
    return
def compute_errors():
    global avg_error
    models = []
    for i in range(0,models_count.get()):
        np.random.shuffle(shuffled_data)
        # test size dependent on test_split
        test_size = int(len(data)*test_split.get())
        train = shuffled_data[test_size:]
        # ironically we don't use test samples to get model precision because it is way too much random
        test = shuffled_data[:test_size]
        data_input_split=input_dimensions.get()
        
        models.append(build_model(train[:,:data_input_split],train[:,data_input_split:],xi.get()))
    
    # error over all samples
    avg_error = average_prediction_error(data,models,data_input_split)

    # average total model error
    avg_model_error = np.average(avg_error)
    iteration_errors.append(avg_model_error)
    print(f"Avg model error: {avg_model_error}")

def plot():
    global high_error_sorted_indices,avg_error
    if avg_error is None: return
    max_elements = elements_to_show.get()
    
    high_error_sorted_indices = np.argsort(-avg_error)[:max_elements]
    X = np.arange(0,max_elements)

    plt.clf()
    plt.gcf().set_size_inches(6,7,forward=True)
    plt.subplot(2,1,1)
    plt.plot(X,avg_error[high_error_sorted_indices],marker='.')
    if(show_indices.get()):
        plt.xticks(X, labels=high_error_sorted_indices)
    plt.xlabel("outlier id")
    plt.ylabel("prediction error")

    plt.subplot(2,1,2)
    plt.plot(np.arange(0,len(iteration_errors)),iteration_errors,marker='.')
    plt.xlabel("iteration N")
    plt.ylabel("model error")

    plt.ion()
    plt.show()

def remove_rows():
    input_str = rows_to_remove.get()
    if str.lower(input_str)!="":
        row_indices = [int(r) for r in input_str.split(" ")]
        global removed_rows,data
        for r in row_indices:
            removed_rows.append(data[r,:])
        data = np.delete(data,row_indices,axis=0)
        print(f'removed {input_str}')

    count_to_remove = n_rows_to_remove.get()
    if(count_to_remove != 0):
        row_indices = high_error_sorted_indices[:count_to_remove]
        for r in row_indices:
            removed_rows.append(data[r,:])
        data = np.delete(data,row_indices,axis=0)
        print(f'removed N {count_to_remove}')

def run_iteration():
    if data is None: 
        print("No database selected")
        return
    print(f"run")
    remove_rows()
    compute_errors()
    plot()
    rows_to_remove.set("")
    n_rows_to_remove.set(0)

def open_file():
    global data,normalization,columns,shuffled_data,working_dir, iteration_errors, removed_rows
    dataset = fd.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if dataset=="": return

    dataset_dir, dataset_name = os.path.split(dataset)
    working_dir=dataset_dir
    print(dataset_dir)
    print(dataset_name)
    data,normalization = load_dataset(dataset_dir,dataset_name)
    columns = pd.read_csv(f"{dataset_dir}/{dataset_name}").columns
    shuffled_data = data.copy()
    iteration_errors = []
    removed_rows = []
    input_dimensions.set(-1)

def save_results():
    if data is None: 
        print("No dataset is opened")
        return
    data_without_outliers = normalization.restore(data)
    removed_rows_restored = normalization.restore(np.array(removed_rows))

    df1 = pd.DataFrame(data_without_outliers, columns=columns)
    df2 = pd.DataFrame(removed_rows_restored, columns=columns)

    df1.to_csv(f"{working_dir}/without_outliers_xi={xi.get()}.csv",header=True,index=False)
    df2.to_csv(f"{working_dir}/outliers_xi={xi.get()}.csv",header=True,index=False)

def gui():
    hack_font = ("Hack",15)

    label = tk.Label(root,text="count of models: ",font=hack_font)
    label.grid(row=1,column=0)

    entry = tk.Entry(root,textvariable=models_count,font=hack_font)
    entry.grid(row=1,column=1)

    label = tk.Label(root,text="xi: ",font=hack_font)
    label.grid(row=2,column=0)

    entry = tk.Entry(root,textvariable=xi,font=hack_font)
    entry.grid(row=2,column=1)

    label = tk.Label(root,text="test split: ",font=hack_font)
    label.grid(row=3,column=0)

    entry = tk.Entry(root,textvariable=test_split,font=hack_font)
    entry.grid(row=3,column=1)

    label = tk.Label(root,text="elements to show: ",font=hack_font)
    label.grid(row=4,column=0)

    entry = tk.Entry(root,textvariable=elements_to_show,font=hack_font)
    entry.grid(row=4,column=1)

    label = tk.Label(root,text="input dimensions: ",font=hack_font)
    label.grid(row=5,column=0)

    entry = tk.Entry(root,textvariable=input_dimensions,font=hack_font)
    entry.grid(row=5,column=1)

    def numbers_list_validation(input):
        # Check if the input is empty
        if not input:
            return True
        # Check if the input contains only digits and spaces
        if all(char.isdigit() or char == " " for char in input):
            return True
        # Otherwise, return False
        return False
    vcmd = root.register(numbers_list_validation)

    label = tk.Label(root,text="N rows to remove: ",font=hack_font)
    label.grid(row=7,column=0)

    entry = tk.Entry(root,textvariable=n_rows_to_remove,font=hack_font)
    entry.grid(row=7,column=1)

    label = tk.Label(root,text="rows to remove: ",font=hack_font)
    label.grid(row=6,column=0)

    entry = tk.Entry(root,textvariable=rows_to_remove,font=hack_font, validate="key", validatecommand=(vcmd, "%P"))
    entry.grid(row=6,column=1)

    button = tk.Button(root,text="run iteration",font=hack_font,command=run_iteration)
    button.grid(row=1,column=2)

    button = tk.Button(root, text="choose dataset", command=open_file,font=hack_font)
    button.grid(row=2,column=2)

    button = tk.Button(root, text="save results", command=save_results,font=hack_font)
    button.grid(row=3,column=2)

    button = tk.Button(root, text="find xi", command=find_xi_grid_search,font=hack_font)
    button.grid(row=4,column=2)

    checkbox = tk.Checkbutton(root,text="show indices",variable=show_indices,font=hack_font)
    checkbox.grid(row=5,column=2)

    checkbox = tk.Button(root,text="plot",font=hack_font,command=plot)
    checkbox.grid(row=6,column=2)

    root.geometry("800x400")
    root.mainloop()

gui()

