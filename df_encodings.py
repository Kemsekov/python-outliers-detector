import pandas as pd
from sklearn.preprocessing import LabelEncoder  # Fix import

def one_hot_encode(data):
    """Replaces string data with their one-hot encoded versions"""
    # Check if input is a DataFrame or Series
    if isinstance(data, pd.Series):
        # One-hot encode the Series directly
        return pd.get_dummies(data, prefix=data.name, dtype=float)
    elif isinstance(data, pd.DataFrame):
        # Make a copy of the original dataframe
        df_copy = data.copy()
        # Loop through each column
        for col in df_copy.columns:
            # Check if the column is of type object (text)
            if df_copy[col].dtype == 'object':
                # Create one-hot encoding for the column
                dummies = pd.get_dummies(df_copy[col], prefix=col, dtype=float)
                # Insert the dummies columns right after the original column
                df_copy = pd.concat([df_copy.iloc[:, :df_copy.columns.get_loc(col) + 1], dummies, df_copy.iloc[:, df_copy.columns.get_loc(col) + 1:]], axis=1)
                # Drop the original column
                df_copy = df_copy.drop(col, axis=1)
        return df_copy
    else:
        raise ValueError("Input must be a pandas DataFrame or Series")

def label_encode(data):
    """Does label-encoding of series or dataframe"""
    # Check if input is a DataFrame or Series
    if isinstance(data, pd.Series):
        # Perform label encoding directly on the Series
        le = LabelEncoder()
        return (pd.Series(le.fit_transform(data), name=data.name),le)
    elif isinstance(data, pd.DataFrame):
        # Make a copy of the original dataframe
        df_copy = data.copy()
        encoders = []
        # Loop through each column
        for col in df_copy.columns:
            # Check if the column is of type object (text)
            if df_copy[col].dtype == 'object':
                # Perform label encoding
                le = LabelEncoder()
                df_copy[col] = le.fit_transform(df_copy[col])
                encoders.append(le)
        return (df_copy,encoders)
    else:
        raise ValueError("Input must be a pandas DataFrame or Series")
