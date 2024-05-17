import os
import tkinter as tk
import tkinter.ttk as ttk
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

class MainWindow:
    def __init__(self, root : tk.Tk,model,font = None) -> None:
        self.root = root
        self.font = font
        self.model=model
        root.title("Outlier detection")
        root.geometry("300x300")
        open_button = tk.Button(root, text="Data", command=self.open_data, font=font)
        open_button.grid(row=0,column=0,**common_button_style)

        save_button = tk.Button(root, text="Model", command=self.open_model, font=font)
        save_button.grid(row=1,column=0,**common_button_style)

        set_data_layout = tk.Button(root, text="Outliers", command=self.open_outliers, font=font)
        set_data_layout.grid(row=2,column=0,**common_button_style)
        self.data_window : DataWindow = None
        self.model_window : ModelWindow = None

    def open_data(self):
        # TODO: fix that if we reopen data window we have to redo all steps of data loading
        if self.data_window is not None: 
            self.data_window.data_window.destroy()
        self.data_window = DataWindow(self.root,self.font)

    def open_model(self):
        if self.data_window is None or self.data_window.data is None:
            messagebox.showerror("Error","Dataset is not loaded")
            return

        if self.model_window is not None: self.model_window.model_window.destroy()
        self.model_window=ModelWindow(self.root,self.model,self.data_window.data,self.data_window.feature_usage,self.font)

    def open_outliers(self):
        pass

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
        self.columns=feature_usage.keys()

        features_table = Table(layout_window,rows=[[k] for k in feature_usage.keys()])
        
        # Make the listbox fill the remaining space
        features_table.grid(row=0, column=1, rowspan=4, **common_button_style)
        
        for i,k in enumerate(self.columns):
            v = feature_usage[k]
            id = features_table.row_id_at(i)
            features_table.tree.item(id,tags=[v])
        
        features_table.tree.tag_configure(NOT_USED,background=NOT_USED)
        features_table.tree.tag_configure(INDEPENDENT,background=INDEPENDENT)
        features_table.tree.tag_configure(DEPENDENT,background=DEPENDENT)

        self.features_table = features_table
        self.feature_usage=feature_usage

    def set_selection_value(self,value):
        features_table=self.features_table
        # Get the indices of the selected options
        selected_rows_ids = features_table.tree.selection()
        # Clear previous selection highlights
        features_table.selection_clear()
        # clear colors
        for row_id in selected_rows_ids:
            row = features_table.tree.item(row_id)

            self.feature_usage[row['values'][0]]=value
            features_table.tree.item(row_id,tags=[value])

    def all_independent(self):
        features_table=self.features_table
        # Clear previous selection highlights
        features_table.selection_clear()
        # clear colors
        for row_index in range(len(self.columns)):
            row_id = features_table.row_id_at(row_index)
            row = features_table.tree.item(row_id)
            self.feature_usage[row['values'][0]]=INDEPENDENT
            features_table.tree.item(row_id,tags=[INDEPENDENT])
        
    def clear(self):
        self.set_selection_value(NOT_USED)
    def set_as_dependent(self):
        self.set_selection_value(DEPENDENT)
        
    def set_as_independent(self):
        self.set_selection_value(INDEPENDENT)


class LabelInputBox(tk.Frame):
    def __init__(self, master, label_text, entry_var,font) -> None:
        super().__init__(master)
        self.label = tk.Label(self,text=label_text,font=font)
        self.label.grid(row=0,column=0)
        self.entry = tk.Entry(self,textvariable=entry_var,font=font)
        self.entry.grid(row=0,column=1)

class Table(tk.Frame):
    def __init__(self,master,rows,headings=None, show_scrollbar = True,row_selection_event = None, **args) -> None:
        """
        A table structure for tkinter
        master: master widget
        rows: 2 dimensional matrix of row values
        headings: optional. If set will add heading on top of the table
        show_scrollbar: adds scrollbar on the right side of the table
        row_selection_event: Fired when row is selected. Function that accepts (table,event) as parameters.
        """
        super().__init__(master,**args)

        tree_show = "headings" if headings is not None else ""
        columns = [i for i in range(len(rows[0]))]
        
        style = ttk.Style()
        style.configure("Treeview.Heading", font=font.Font(family='Helvetica', size=30, weight='bold'))

        tree = ttk.Treeview(self, show=tree_show, columns=columns)

        if headings is not None:
            for c,h in zip(columns,headings):
                tree.heading(c,text=h)
        tree_scroll = ttk.Scrollbar(self,orient=tk.VERTICAL,command=tree.yview)
        tree.configure(yscroll=tree_scroll.set)
        for row in rows:
            tree.insert("",tk.END,values=row)
        

        if row_selection_event is not None:
            tree.bind("<<TreeviewSelect>>", lambda event: row_selection_event(self,event))
        
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        tree.grid(row=0,column=0)
        if show_scrollbar:
            tree_scroll.grid(row=0, column=1, sticky=tk.N + tk.S)
        self.tree = tree

        # tree.item(self.row_at(0),tags=["1"])
    
    def row_selection(self):
        """returns selected rows"""
        return [self.tree.item(selection) for selection in self.tree.selection()]
    
    def rows_count(self): return len(self.tree.get_children())

    def row_at(self,index): 
        """get row element under given index"""
        return self.tree.item(self.row_id_at(index))

    def row_id_at(self,index): 
        """
        returns row_id under given index. 
        
        it then can be used to change row elements like.

        `table.tree.item(table.row_id_at(0),tags=["1"],values=["1","2"])`
        
        Example above will change row under index 0
        """
        return self.tree.get_children()[index]

class ModelWindow:
    def __init__(self,root : tk.Tk,model, data : pd.DataFrame, feature_usage : FeatureUsage, font) -> None:
        self.root = root
        self.model = model
        self.data = data
        self.feature_usage = feature_usage
        model_window = tk.Toplevel(root)
        model_window.title("Model selection")
        model_window.geometry("800x300")
        self.model_window=model_window

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

        gridsearch_button = tk.Button(model_window, text="GridSearchCV", command=self.grid_search_cv,font=font)
        gridsearch_button.grid(row=2, column=0, **common_button_style)

        cv = tk.IntVar(model_window,value=5)
        cv_frame = LabelInputBox(model_window,"CV",cv,font)
        cv_frame.grid(row=2,column=1)
        self.cv=cv

        randomsearch_button = tk.Button(model_window, text="RandomSearchCV", command=self.random_search_cv,font=font)
        randomsearch_button.grid(row=3, column=0, **common_button_style)

        n_iter = tk.IntVar(model_window,value=300)
        n_iter_frame = LabelInputBox(model_window,"Iterations",n_iter,font)
        n_iter_frame.grid(row=3,column=1, **common_button_style)
        self.n_iter=n_iter
    def grid_search_cv(self):
        pass
    def random_search_cv(self):
        
        pass
