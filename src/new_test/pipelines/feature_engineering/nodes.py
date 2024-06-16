"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.5
"""

"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.2
"""

#TODO: add in target-date relative features in FE pipeline

import pandas as pd
import numpy as np



def calculate_rolling_means(dataframe: pd.DataFrame, parameters: dict) -> pd.DataFrame: 

    
    '''return a dataframe that includes all of the rolling means either straight or exponential appended to the overall dataset
        
        Args:
            dataframe: main dataset (from raw to be fed in or specified in the modeling catalog
            date field: field containing the trading date of the equity
            field: field to use on which to calculate the rolling standard deviations
            day_ranges: list of the days over which the rolling mean is to be calculates (e.g., 6, 7, 15)
            exponential: If True will calculate exponential moving averages instead of simple moving averages
        
        Returns: Dataframe containing the user-specified moving averages
        
    '''
    dataframe = dataframe.sort_values(by =[parameters['stock_field'], parameters['date_field']]  )

    for days in parameters['day_ranges']: # loop through each day range and append the new column after running for each security
        
        moving_averages = []

        if parameters['exponential'] == True:

            for equity in dataframe[parameters['stock_field']].unique():
                temp = dataframe[dataframe[parameters['stock_field']] == equity]
                stock_moving_average = temp[parameters['calculation_field']].ewm(span = days, min_periods = days).mean()
                moving_averages.extend(stock_moving_average)
                del temp

            dataframe[str(days) +'_' + parameters['calculation_field'] + '_' + 'ema'] = moving_averages
        
        else:
            for equity in dataframe[parameters['stock_field']].unique():
                temp = dataframe[dataframe[parameters['stock_field']] == equity].reset_index(drop = True)
                stock_moving_average = temp[parameters['calculation_field']].rolling(days).mean()
                moving_averages.extend(stock_moving_average)
                del temp
            
            dataframe[str(days) +'_' + parameters['calculation_field'] + '_' + 'sma'] = moving_averages
    
    return dataframe


def calculate_rolling_standard_deviations(dataframe: pd.DataFrame, parameters: dict) -> pd.DataFrame:

    '''Function that calculates the standard deviations of the specified fields
    
    
    Args:
        Dataframe: pandas dataframe that contains a series of values over which the standard deviations are to be calculated
    
    Returns: Dataframe with standard deviation values added as columns in the dataset
    '''

    #TODO: come back and remove this nested for loop with more efficient process: pyspark likely a better option

    for days in parameters['day_ranges']:

        standard_deviations = []

        for equity in dataframe[parameters['stock_field']].unique():

            temp = dataframe[dataframe[parameters['stock_field']] == equity].reset_index(drop = True)
            standard_deviations.extend(temp[parameters['calculation_field']].rolling(days).std())
            print(len(standard_deviations))
        
        dataframe[str(days) +'_' + parameters['calculation_field'] + '_' + 'std'] = standard_deviations

        del standard_deviations

    return dataframe


def create_above_below_indicator_fields(dataframe: pd.DataFrame, 
                                        parameters: dict) -> pd.DataFrame:

    #TODO: Come back and format functions to be in proper format with black/linting with proper documentation

    '''Function that adds indicator fields or calculates percentage differences or both depending on arguments
    
    Args:
        dataframe: input as a pandas dataframe with fields already included for calculation

    Returns: dataframe that includes representative fields for the functionality specified by the user
    '''

    target_columns = dataframe.columns[dataframe.columns.str.contains("|".join(['sma', 'ema']))]

    for column in target_columns:

        if parameters['indicator_return_type'] == 'boolean':

            dataframe['above_'+ column + '_ind'] = np.where(dataframe[column].isna(), 
            np.nan, np.where(dataframe[column] < dataframe[parameters['calculation_field']], 1, 0))
        
        elif parameters['indicator_return_type'] == 'percentage':

            dataframe[column + '_pct_diff'] = np.where(dataframe[column].isna(), 
            np.nan,  dataframe[column] / dataframe[parameters['calculation_field']] -1)
        
        else:

            dataframe['above_'+ column + '_ind'] = np.where(dataframe[column].isna(), 
            np.nan, np.where(dataframe[column] < dataframe[parameters['calculation_field']], 1, 0))

            dataframe[column + '_pct_diff'] = np.where(dataframe[column].isna(), 
            np.nan,  dataframe[column] / dataframe[parameters['calculation_field']] -1)

    return dataframe


def calculate_cumulative_days_above(dataframe: pd.DataFrame, parameters: dict) -> pd.DataFrame:

    '''Calculates the cumulative days spent above a given moving average(s)
    CAUTION: Must contain indicator field for each respective moving average -- requires running in the pipeline to remove from pipeline, recalculate features separately
    
    Args:
        dataframe: dataframe containing a series of moving average fields across different equity tickers
        mnoving averages

    Returns: pandas dataframe containing the newly created cumulative features
    '''

    fields_to_calc = dataframe.columns[dataframe.columns.str.contains("|".join(['close_ema_ind', 'close_sma_ind']))]
    # sort values to ensure consistency (in case something changes in the dataframe):
    dataframe = dataframe.sort_values(by =[parameters['stock_field'], parameters['date_field'] ])

    #TODO: Figure out a better way to ensure field + ticker consistency without nested for loop

    for field in fields_to_calc: 
        
        #TODO: Figure out a better way to do this running total on a series
        temp = dataframe[[parameters['stock_field'], parameters['date_field'], field]].reset_index(drop = True)
        temp.fillna(0, inplace = True) # fill nulls for consistency

        groups = ((temp[parameters['stock_field']]!=temp[parameters['stock_field']].shift()) | (temp[field]!=temp[field].shift())).cumsum()

        dataframe['cum_days_above_' + field ] = temp.groupby(by = groups)[field].cumsum()

        del temp, groups

    return dataframe.reset_index(drop = True)


def create_bollinger_bands(dataframe: pd.DataFrame, global_parameters: dict, function_parameters: dict) -> pd.DataFrame:

    #TODO: evolve model to include functionality to include multiple sets of bollinger bands

    '''function that returns bollinger bands for each equity in the datasets sent to the model
    
    Args:
        Dataframe: pandas dataframe containing the equities for which the bollinger bands are calculated
        calculation_field: field used to calculate all features (set in the globals parameters)
        moving_average_used: moving average field (calculated in prior step) to be used in the model
        number_of_std: number of standard deviations from the mean to calculate the upper and lower bands
        use_sma: Boolean to indicate whether to use the EMA or SMA in order to calculate the bollinger bands
        return_top_distance: Boolean for whether to return field indicating distance to the upper band
        return_bottom_distance: Boolean for whether to return field indicating distance to the bottom band
        return_gap: Boolean for whether to return the distance between bands and the proportion relative to the price

    Returns: dataframe with bollinger bands and moving trends added to the core dataset
    '''

    assert np.isin(str(function_parameters['moving_average_used']) +'_' + global_parameters['calculation_field'] + '_' + 'std', dataframe.columns), \
    'please ensure the moving average number of days is in the days parameter'

    if function_parameters['use_sma'] == True:
        
        assert np.isin(str(function_parameters['moving_average_used']) + '_' + global_parameters['calculation_field'] + '_' + 'sma', dataframe.columns), \
            'Please ensure the moving average calculated is SMA'
        

        dataframe['upper_bollinger_band'] = dataframe[str(function_parameters['moving_average_used']) + '_' + global_parameters['calculation_field'] + '_' + 'sma'] + \
                                            (function_parameters['number_of_std'] * dataframe[str(function_parameters['moving_average_used']) +'_' + global_parameters['calculation_field'] + '_' + 'std'])

        dataframe['lower_bollinger_band'] = dataframe[str(function_parameters['moving_average_used']) + '_' + global_parameters['calculation_field'] + '_' + 'sma'] - \
                                            (function_parameters['number_of_std'] * dataframe[str(function_parameters['moving_average_used']) +'_' + global_parameters['calculation_field'] + '_' + 'std'])

        if function_parameters['return_top_distance'] == True:
            dataframe['bol_pct_from_top'] = dataframe[global_parameters['calculation_field']] / dataframe['upper_bollinger_band'] -1

        if function_parameters['return_bottom_distance'] == True:
            dataframe['bol_pct_from_bottom'] = dataframe[global_parameters['calculation_field']] / dataframe['lower_bollinger_band'] -1

        if function_parameters['return_gap'] == True:
            dataframe['bol_range'] = dataframe['upper_bollinger_band'] - dataframe['lower_bollinger_band']
            dataframe['bol_range_pct'] = (dataframe['upper_bollinger_band'] - dataframe['lower_bollinger_band']) / dataframe[global_parameters['calculation_field']]

    elif function_parameters['use_sma'] == False:

        assert np.isin(str(function_parameters['moving_average_used']) + '_' + global_parameters['calculation_field'] + '_' + 'ema', dataframe.columns), \
            'Please ensure the moving average calculated is EMA'

        dataframe['upper_bollinger_band'] = dataframe[str(function_parameters['moving_average_used']) + '_' + global_parameters['calculation_field'] + '_' + 'ema'] + \
                                            (function_parameters['number_of_std'] * dataframe[str(function_parameters['moving_average_used']) +'_' + global_parameters['calculation_field'] + '_' + 'std'])

        dataframe['lower_bollinger_band'] = dataframe[str(function_parameters['moving_average_used']) + '_' + global_parameters['calculation_field'] + '_' + 'ema'] - \
                                            (function_parameters['number_of_std'] * dataframe[str(function_parameters['moving_average_used']) +'_' + global_parameters['calculation_field'] + '_' + 'std'])


        if function_parameters['return_top_distance'] == True:
            dataframe['bol_pct_from_top'] = dataframe[global_parameters['calculation_field']] / dataframe['upper_bollinger_band'] -1

        if function_parameters['return_bottom_distance'] == True:
            dataframe['bol_pct_from_bottom'] = dataframe[global_parameters['calculation_field']] / dataframe['lower_bollinger_band'] -1
    
        if function_parameters['return_gap'] == True:
            dataframe['bol_range'] = dataframe['upper_bollinger_band'] - dataframe['lower_bollinger_band']
            dataframe['bol_range_pct'] = (dataframe['upper_bollinger_band'] - dataframe['lower_bollinger_band']) / dataframe[global_parameters['calculation_field']]


    return dataframe


def create_classification_target_variable(dataframe: pd.DataFrame, global_parameters: dict, prediction_parameters: dict) -> pd.DataFrame:

    '''Function that creates the target feature for the predictive model(s)
    
    Args:
        dataframe: main dataset containing the outputs of the feature engineering pipeline
        target_field: field from which the target feature is the be generated
        stock_field: field containing the stock/ticker symbol(s)
        prediction_horizon: timeframe from which to calculate the prediction (e.g., 20 days out)

    Returns: Dataframe containing the predictive model target
        
    '''

    # always start by sorting and resetting the index:
    dataframe = dataframe.sort_values(by =[global_parameters['stock_field'], global_parameters['date_field'] ]).reset_index(drop = True)

    dataframe['target_'+ str(prediction_parameters['prediction_horizon'])+"_days_ahead"] = dataframe.groupby(by = global_parameters['stock_field'])[global_parameters['calculation_field']].shift(-prediction_parameters['prediction_horizon'])

    # create boolean for classification
    dataframe['target_'+ str(prediction_parameters['prediction_horizon'])+"_days_ahead_ind"] = np.where(dataframe[global_parameters['calculation_field']] < dataframe['target_'+ str(prediction_parameters['prediction_horizon'])+"_days_ahead"],
                                                                                    1 , 0 )
  
    return dataframe

            


   









