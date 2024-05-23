# Load the data
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

df_raw = pd.read_excel("/Users/sepuliini/Desktop/DDO_kurssi/Metall. Application.xls", header=1)
df = df_raw.copy()

# Drop empty and unclear rows
df = df.drop(labels=[107, 108, 109, 110, 111, 738, 739, 741], axis='index')

# Drop duplicate columns, data number and paper number columns
df = df.drop(list(df.columns[0:5]), axis='columns')
df = df.rename(columns={'Nb.1': 'Nb', 'Ti.1': 'Ti', 'V.1': 'V'})

df_elem = df.iloc[:, 0:df.columns.get_loc('Sn')+1]

# Replace NaNs with zeros
df_elem.iloc[:, 0:df.columns.get_loc('Sn')+1].fillna(0, inplace=True)

# More ambiguous rows to delete
df_elem = df_elem.drop([134, 674, 675, 676], axis='index')

# Drop column Sn because it is all zeros
df_elem.drop(['Sn'], axis='columns', inplace=True)

# Merge columns Nb and Cb as they are the same element
df_elem['Nb'] = df_elem['Nb'].mask(df_elem['Nb'] == 0, df_elem['Cb'])
df_elem.drop(['Cb'], axis='columns', inplace=True)

# Convert all valid numeric values to float
def convert_to_float(value):
    try:
        return float(value)
    except ValueError:
        return np.nan

df_elem = df_elem.applymap(convert_to_float)

# Drop rows where conversion to float failed (NaNs introduced)
df_elem.dropna(inplace=True)

"""
We still have columns that are mostly empty. We can decide a limit for 
the ratio of non-null to null values we are willing to accept, for example 
20 %. If a column is below this value, the column and all rows associated with it
are deleted.
"""

# An example: print all non-zero indices of variable 'Ca' 
print(df_elem['Ca'][df_elem['Ca'].gt(0)].index)

def remove_sparse_columns(df):
    dfc = df.copy()
    n = dfc.shape[0]
    lim = 0.2
    
    # Run while the column with fewest non-nulls has under 20% values
    while np.min(dfc.gt(0).sum(axis=0).values) / n < lim: 
        
        print(dfc.gt(0).sum(axis=0))
        
        # Find emptiest column
        emptiest_column_index = np.argmin(dfc.gt(0).sum(axis=0).values)
        # Get column name
        col_to_remove = dfc.columns[emptiest_column_index]
        mask = dfc[col_to_remove].gt(0)
        # Get rows associated with the column to delete
        idx_to_remove = dfc[col_to_remove][mask].index
        print(col_to_remove)
        print(idx_to_remove)
        
        dfc = dfc.drop(idx_to_remove, axis='index')
        dfc = dfc.drop(col_to_remove, axis='columns')
        print(dfc.shape)
        n = dfc.shape[0]
        
    return dfc

df_elem_cleaned = remove_sparse_columns(df_elem)    

"""
Create DataFrames for yield strength, ultimate tensile strength 
and elongation strength
"""

df_ys = df_elem_cleaned.copy()
df_uts = df_elem_cleaned.copy()
df_el = df_elem_cleaned.copy()

dfs = [df_ys, df_uts, df_el]
targets = ['YS(Mpa)', 'UTS(Mpa)', '%EL']

for data, target in zip(dfs, targets):
    
    # Get target column from original data
    target_column = df[target]
    
    # Drop rows that are not present in the DataFrame
    target_column = target_column[target_column.index.isin(data.index)]

    # Add target column to DataFrame
    data[target] = target_column
    
    # Drop rows with NaNs
    data.dropna(axis='index', inplace=True)
    

import re

"""
Some of the target variables contain bad values. These are either given as ranges (e.g. 0.2-0.3)
or as approximations (e.g. ~2.5). We could also drop these rows.
"""
targets = ['YS', 'UTS', 'EL']
for i, target in enumerate(targets):
    dfs[i] = dfs[i].rename(columns={dfs[i].columns[-1]: target})
    
    # Try converting to float
    try:
        dfs[i] = dfs[i].astype('float')
    except ValueError:
        # Mask for bad rows
        bad_rows = [type(val) != int and type(val) != float for val in dfs[i][target]]
        # Indices for bad rows
        bad_rows = dfs[i].index[bad_rows]
        for row_idx in bad_rows:
            
            # If range, convert to mean
            if '-' in dfs[i][target][row_idx]:
                values = [float(x) for x in dfs[i][target][row_idx].split('-')]
                dfs[i].at[row_idx, target] = np.mean(values)
                
            # If approximation, convert to exact value
            else:
                number = float(re.sub("[^0-9.]", "", dfs[i][target][row_idx]))
                dfs[i].at[row_idx, target] = number
        
        dfs[i] = dfs[i].astype('float')

# Scale the datasets
scaler = StandardScaler()
df_elem_cleaned_scaled = pd.DataFrame(scaler.fit_transform(df_elem_cleaned), columns=df_elem_cleaned.columns)
dfs_scaled = [pd.DataFrame(scaler.fit_transform(df), columns=df.columns) for df in dfs]

# Save the cleaned datasets to Excel files
df_elem_cleaned.to_excel("/Users/sepuliini/Desktop/DDO_kurssi/cleaned_data_elem.xlsx", index=False)
dfs[0].to_excel("/Users/sepuliini/Desktop/DDO_kurssi/cleaned_data_ys.xlsx", index=False)
dfs[1].to_excel("/Users/sepuliini/Desktop/DDO_kurssi/cleaned_data_uts.xlsx", index=False)
dfs[2].to_excel("/Users/sepuliini/Desktop/DDO_kurssi/cleaned_data_el.xlsx", index=False)