import os
import pandas as pd
import numpy as np
import yaml

def _strip_spaces(df):
    """Strip spaces from column names."""
    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    return df

def _find_header(path: str, key: str):
    """Find the header of a csv file."""
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if key in line:
                return i


def extract_defect_labels(
    csv_file_folder: str,
    save_path: str = None, 
    header_key: str = 'DEFECT_NAME', 
    column: list = ['GLASS_ID', 'CELL_ID', 'DEFECT_CODE', 'DEFECT_NAME', 'IMAGE_FILE1'],
    nrows: int = -4,
    file_extension: str = '.dat'):
    """Extract defect labels from csv files in a folder."""
    
    csv_file_list = os.listdir(csv_file_folder)
    csv_file_list = [os.path.join(csv_file_folder, file) for file in csv_file_list if file.endswith(file_extension)]
    csv_file_list.sort()

    df_list = []
    for file in csv_file_list:
        # Get the glass id
        glass_id = os.path.basename(file).split('_')[0]

        # Add in the defect labels
        header = _find_header(file, header_key)
        df = pd.read_csv(file, header=header)
        df = _strip_spaces(df)
        df['GLASS_ID'] = glass_id
        df = df[column]

        # Number of lines of dataframe to keep / drop
        df = df[:nrows]

        df_list.append(df)

    df = pd.concat(df_list, axis=0)

    # filter out blank rows in any column
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.dropna(axis=0, how='any')
    df = df.reset_index(drop=True)

    # If save path is specified save as a xml file
    if save_path is not None:
        df.to_csv(save_path, index=False)
    
    return df