"""
This is a boilerplate pipeline 'Ml_pre_processing'
generated using Kedro 0.19.5
"""

"""
This is a boilerplate pipeline 'data_pre_processing'
generated using Kedro 0.18.2
"""

from typing import Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



def clean_stock_data(dataframe: pd.DataFrame) -> pd.DataFrame :

    '''removes nulls and in the future will be built out to do any additonal cleaning on the dataframe that is necessary
    Args:
        dataframe: pandas dataframe containing all of the potential features
        parameters: 
            calculation_field: field on which all of the features are built

    Returns:
        dataframe: dataset that is ready to load into a machine learning framework
    '''


    # remove records the preceed the target period to have complete information:
    dataframe.dropna(inplace = True)

    ### Not necessary - Unless we made this one step to write out, want to be able to go ack to original indexes
    #dataframe = dataframe.reset_index(drop = True) # we won't reset the index for now for traceability back to the date, ticker combination later after training

    # set the date as an index to us post-forecasting: This is a bad idea, come back to the concept
    #dataframe.set_index(keys = 'date', verify_integrity = False, inplace = True) # verify integrity Fale to allow duplicates**
    
    # remove fields that will not be used as predictive features (can be hardcoded since dataframe structure will be the same):
    dataframe = dataframe.drop(columns = [ 'date', 'high', 'low', 'open', 'volume', 'adj_close'])
    

    return dataframe

def identify_fields_to_standardize(dataframe: pd.DataFrame, parameters: Dict) -> np.array :

    '''creates a list of the continuous fields to standardize by dimension within the predictive model; NOTE: this is used within the standardizer
    
    Args:
        dataframe: dataframe that contains all of the fields of interest to be used in the calculations
        parameters:
            continuous_feature_cutoff: ratio of unique values to record count to be used to codify continuous features -> removes records from the standardization process which don't have enough data to standardize (e.g., boolean)

    Returns: list of continuous fields to use in the standardization process based on user's specifications of "uniqueness" threshold    

    '''

    numeric_fields = dataframe.select_dtypes(include = 'number').columns
    records = len(dataframe)

    record_summary = pd.DataFrame(dataframe[numeric_fields].nunique(), columns = ['unique_values'])
    record_summary['rows_in_df'] = records
    record_summary['value_to_record_ratio'] = record_summary['unique_values'] / record_summary['rows_in_df']

    # filter for a threshold specified by the user:
    record_summary = record_summary[record_summary['value_to_record_ratio'] > parameters['continuous_feature_cutoff']]
    # remove percentage features # TODO: later add in functionality to remove percentage based features

    return record_summary.index


def standardize_continuous_features(dataframe: pd.DataFrame, global_parameters: Dict, parameters: Dict) -> pd.DataFrame:

    '''function that identifies the continuous features in the dataframe and standardizes each feature by equity to enable scaling relative to each equity
    
    Args:
        Dataframe: Pandas dataframe to be used in machine learning
        Parameters:
            stock_field: field indicating the stock for the window function to scan
            calculation_field: field for which the target is being calculated (used for drop in main row merge)
    
    Returns:
        Dataframe: containing the standardized data fields
    
    '''

    continuous_fields = list(identify_fields_to_standardize(dataframe = dataframe, parameters = parameters))

    # add in the ticker for grouping next:
    continuous_fields.append(global_parameters['stock_field'])

    # downselect to the fields that will be used to standardize:
    continuous_dataframe = dataframe[continuous_fields]

    # calculate z-scores: --> Standardizes within each feature to scale accordingly
    z_scores = (continuous_dataframe - continuous_dataframe.groupby(by = global_parameters['stock_field']).transform('mean')) / continuous_dataframe.groupby(by = global_parameters['stock_field']).transform('std')

    # drop the null ticker (not needed post groupby): 
    z_scores.drop(columns = [ global_parameters['stock_field'], global_parameters['calculation_field'] ], inplace = True)

    # rename the fields to indicate standardization:
    z_scores.columns = z_scores.columns + '_std'

    # drop original continuous fields # TODO: coming back after calculation checks:
    if parameters['drop_original_fields'] == True:
        continuous_fields.remove(global_parameters['stock_field'])
        dataframe.drop(columns = continuous_fields, inplace = True)

    # append the fields back into the core dataframe:
    z_scores = pd.concat([dataframe, z_scores], axis = 1)

    # remove the standardized target field:
    z_scores.drop(columns = z_scores.columns[z_scores.columns.str.contains('target')][1], inplace = True)

    # remove unnecessary items:
    del continuous_fields, continuous_dataframe

    return z_scores


def one_hot_encode_tickers(dataframe: pd.DataFrame, parameters: Dict) -> pd.DataFrame:

    '''Returns one-hot encoded features to the predictive dataset NOTE: May not work, but this retains some of the information in the original dataframe while also potentially giving the global model a nudge
       Note: we choose not to drop first for now, even though it's a trap; Can be used post processing or as model features
    Args:
        dataframe: core dataset that has been augmented with additional features
        parameters:
            stock_field: text field containing the 
    Returns:   
        dataframe with augmented columns
    
    '''

    dataframe = pd.get_dummies(data = dataframe, prefix = "ind", columns = [parameters['stock_field']], drop_first = False)

    return dataframe


def create_training_test_splits(dataframe: pd.DataFrame, parameters: Dict) -> Tuple:

    '''Function that splits out training and test sets for machine learning; for the purposes of this model the way we piose the problem allows for random train test split
    Args:
        dataframe: pandas dataframe containing only the target field and the features to be used by the classifier
        parameters:
            test_ratio: proportion of samples in the dataframe to be used as a test set once the models are tuned and evaluated

    '''

    # define Y and x:
    target_feature = list(dataframe.columns[dataframe.columns.str.contains('target')])

    y = dataframe[target_feature]
    X = dataframe.drop(columns = target_feature)

    # create the training and test splits:
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=parameters['test_size'], random_state=parameters['seed'], stratify = y)

    return X_train, X_test, y_train, y_test





