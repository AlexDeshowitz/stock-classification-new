from typing import Dict, List

import numpy as np
import pandas as pd


def fix_columns(columns: list) -> list:
    
    '''function that takes a list of columns and modifies them to be easier to read -- assign to df.columns
    
    Args:
        columns: list of the columns in the dataframe
    
    Returns: list of columns to be set as the dataframe columns
    
    '''
    
    column_string_replace = ['\n','@',' ','__', '/', '-']


    columns = columns.map(lambda x: x.strip())
    columns = columns.map(lambda x : x.lower())

    for string in column_string_replace:
        columns = columns.map(lambda x : x.replace(string, '_') if isinstance (x, (str, bytes)) else x)

    return columns