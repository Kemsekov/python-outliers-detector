import os
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog as fd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import font

NOT_USED = 'white'
INDEPENDENT = 'green'
DEPENDENT = 'blue'

common_button_style={
    "padx":10, "pady":10, "sticky":'nsew'
}

class FeatureUsage:
    def __init__(self,columns : list[str]) -> None:
        self._usage = {k:NOT_USED for k in columns}
    
    def used(self,column : str): return self._usage[column]!=NOT_USED
    def independent(self,column : str): return self._usage[column]==INDEPENDENT
    def dependent(self,column : str): return self._usage[column]==DEPENDENT

    def keys(self): return self._usage.keys()
    def __len__(self): return self._usage.__len__()
    def __getitem__(self, column : str): return self._usage[column]
    def __setitem__(self, column : str,new_state): self._usage[column]=new_state

class DataWindow:
    def __init__(self, root: tk.Tk,font : font.Font) -> None:
        self.root = root
        self.font = font
        self.feature_usage : FeatureUsage = None
        data_window = tk.Toplevel(root)
        data_window.title("Data")
        data_window.geometry("250x200")
        
        open_button = tk.Button(data_window, text="Open dataset", command=self.open_dataset, font=font)
        open_button.grid(row=0,column=0,**common_button_style)

        save_button = tk.Button(data_window, text="Save dataset", command=self.save_dataset, font=font)
        save_button.grid(row=1,column=0,**common_button_style)

        set_data_layout = tk.Button(data_window, text="Set data layout", command=self.set_data_layout, font=font)
        set_data_layout.grid(row=2,column=0,**common_button_style)

        self.data_window=data_window

        self.dataset : str = None
        self.data : pd.DataFrame = None
        self._dataset_dir : str = None
        self._dataset_name : str = None

    def open_dataset(self):
        dataset = fd.askopenfilename(filetypes=[("CSV files", "*.csv"),("Excel files","*.xls"),("X Excel files","*.xlsx")])
        if not os.path.isfile(dataset):
            messagebox.showerror("Error", "Cannot open file")
            return
        dataset_dir, dataset_name = os.path.split(dataset)
        self.dataset = dataset
        self._dataset_dir = dataset_dir
        self._dataset_name = dataset_name
        if dataset.endswith(".csv"):
            self.data = pd.read_csv(dataset)
        if dataset.endswith(".xls") or dataset.endswith(".xlsx"):
            self.data = pd.read_excel(dataset)
        self.feature_usage = FeatureUsage(self.data.columns)

    def save_dataset(self):
        pass
    def set_data_layout(self):
        if self.data is None:
            messagebox.showerror("Error","Dataset is not open")
            return
        DataLayoutWindow(self.root,self.feature_usage,self.font)

class DataLayoutWindow:
    def __init__(self, root: tk.Tk, feature_usage: FeatureUsage, font) -> None:
        self.root = root
        layout_window = tk.Toplevel(root)
        layout_window.title("Data layout")
        layout_window.geometry("400x600")

        # Configure the grid layout to allow the listbox to expand
        layout_window.grid_columnconfigure(1, weight=1)
        layout_window.grid_rowconfigure(0, weight=1)
        layout_window.grid_rowconfigure(1, weight=1)
        layout_window.grid_rowconfigure(2, weight=1)
        layout_window.grid_rowconfigure(3, weight=1)

        clear_button = tk.Button(layout_window, text="Clear", command=self.clear, bg='white', font=font)
        clear_button.grid(row=0, column=0, **common_button_style)

        make_all_independent = tk.Button(layout_window, text="All independent", command=self.all_independent, font=font)
        make_all_independent.grid(row=1, column=0, **common_button_style)

        independent_button = tk.Button(layout_window, text="Independent", command=self.set_as_independent, bg='green', font=font)
        independent_button.grid(row=2, column=0, **common_button_style)
        
        dependent_button = tk.Button(layout_window, text="Dependent", command=self.set_as_dependent, bg='blue', font=font)
        dependent_button.grid(row=3, column=0, **common_button_style)

        listbox = tk.Listbox(layout_window, selectmode='multiple', font=font)
        # Make the listbox fill the remaining space
        listbox.grid(row=0, column=1, rowspan=4, **common_button_style)

        self.columns = list(feature_usage.keys())
        for column in self.columns:
            listbox.insert(tk.END, column)
        
        for i,k in enumerate(self.columns):
            v = feature_usage[k]
            listbox.itemconfig(i, {'bg':v})

        self.listbox = listbox
        self.feature_usage=feature_usage

    def clear(self):
        listbox=self.listbox
        # Get the indices of the selected options
        selected_indices = listbox.curselection()
        # Clear previous selection highlights
        listbox.selection_clear(0, tk.END)
        # Set the background color of selected items to blue
        for i in selected_indices:
            self.feature_usage[self.columns[i]]=NOT_USED
            listbox.itemconfig(i, {'bg':NOT_USED})
        
    def all_independent(self):
        listbox=self.listbox
        # Clear previous selection highlights
        listbox.selection_clear(0, tk.END)
        # Set the background color of selected items to blue
        for i in range(listbox.size()):
            self.feature_usage[self.columns[i]]=INDEPENDENT
            listbox.itemconfig(i, {'bg':INDEPENDENT})
        
    def set_as_dependent(self):
        listbox=self.listbox
        # Get the indices of the selected options
        selected_indices = listbox.curselection()
        # Clear previous selection highlights
        listbox.selection_clear(0, tk.END)
        # Set the background color of selected items to blue
        for i in selected_indices:
            self.feature_usage[self.columns[i]]=DEPENDENT
            listbox.itemconfig(i, {'bg':DEPENDENT})
        
    def set_as_independent(self):
        listbox=self.listbox
        # Get the indices of the selected options
        selected_indices = listbox.curselection()
        # Clear previous selection highlights
        listbox.selection_clear(0, tk.END)
        # Set the background color of selected items to blue
        for i in selected_indices:
            self.feature_usage[self.columns[i]]=INDEPENDENT
            listbox.itemconfig(i, {'bg':INDEPENDENT})

class LabelInputBox(tk.Frame):
    def __init__(self, master, label_text, entry_var,font) -> None:
        super().__init__(master)
        self.label = tk.Label(self,text=label_text,font=font)
        self.label.grid(row=0,column=0)
        self.entry = tk.Entry(self,textvariable=entry_var,font=font)
        self.entry.grid(row=0,column=1)


class ModelWindow:
    def __init__(self,root : tk.Tk,model, data : pd.DataFrame, feature_usage : FeatureUsage, font) -> None:
        self.root = root
        self.model = model
        self.data = data
        self.feature_usage = feature_usage
        model_window = tk.Toplevel(root)
        model_window.title("Model selection")
        model_window.geometry("400x600")

        model_type = tk.StringVar()
        classification_rb = tk.Radiobutton(model_window, text="Classification", variable=model_type, value="Classification", font=font)
        classification_rb.grid(row=0,column=0,**common_button_style)
        
        regression_rb = tk.Radiobutton(model_window, text="Regression", variable=model_type, value="regression", font=font)
        regression_rb.grid(row=0,column=1,**common_button_style)
        self.model_type=model_type

        data_subset = tk.IntVar(model_window,value=len(data))
        data_subset_frame = LabelInputBox(model_window,"Data subset size",data_subset,font)
        data_subset_frame.grid(row=1,column=0,**common_button_style)
        self.data_subset=data_subset

        repeats = tk.IntVar(model_window,value=3)
        repeats_frame = LabelInputBox(model_window,"Repeats",repeats,font)
        repeats_frame.grid(row=1,column=1,**common_button_style)
        self.repeats=repeats

        gridsearch_button = tk.Button(model_window, text="GridSearchCV", command=self.grid_search_cv)
        gridsearch_button.grid(row=2, column=0, **common_button_style)

        cv = tk.IntVar(model_window,value=5)
        cv_frame = LabelInputBox(model_window,"CV",cv,font)
        cv_frame.grid(row=2,column=1)
        self.cv=cv

        randomsearch_button = tk.Button(model_window, text="RandomSearchCV", command=self.random_search_cv)
        randomsearch_button.grid(row=3, column=1, **common_button_style)

        n_iter = tk.IntVar(model_window,value=300)
        n_iter_frame = LabelInputBox(model_window,"Iterations",n_iter,font)
        n_iter_frame.grid(row=3,column=1, **common_button_style)
        self.n_iter=n_iter
    def grid_search_cv(self):
        pass
    def random_search_cv(self):
        pass
