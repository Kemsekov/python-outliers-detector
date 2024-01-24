import pandas as pd

# Define a function that iterates over columns and creates one-hot encoding for text columns
def one_hot_encode(df):
    # Make a copy of the original dataframe
    df_copy = df.copy()
    # Loop through each column
    for col in df_copy.columns:
        # Check if the column is of type object (text)
        if df_copy[col].dtype == 'object':
            # Create one-hot encoding for the column
            dummies = pd.get_dummies(df_copy[col], prefix=col,dtype=float)
            # Insert the dummies columns right after the original column
            df_copy = pd.concat([df_copy.iloc[:, :df_copy.columns.get_loc(col) + 1], dummies, df_copy.iloc[:, df_copy.columns.get_loc(col) + 1:]], axis=1)
            # Drop the original column
            df_copy = df_copy.drop(col, axis=1)
    # Return the transformed dataframe
    return df_copy
print("Transform all string values of dataset into dummies encoding")
path = input("Path to csv: ")
to_save = input("Path to save: ")

data = pd.read_csv(path)
one_hot_encode(data).to_csv(to_save)



