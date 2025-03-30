import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




def lake_files(data_path):
    '''
    Inputs: 
    Path to folder, where files 'data_ip.csv' and 'ltbl_ice.csv' are located.
    These are the original files for lake data. This function creates needed columns
    for ice-cover duration analysis and combines a clean data frame where the most interesting
    columns of original data are included.

    Outputs:
    data_ip: modified data frame based on file 'data_ip.csv'.
    ltbl_ice: modified data frame based on file 'ltbl_ice.csv'.
    ltbl_ice_depth: like ltbl_ice, but NaN-values of 'depth_max_m'-column are dropped.
    df_ice_clean: clean combination of data frames data_ip and ltbl_ice_depth

    '''

    # Opening freezing data:
    data_ip = pd.read_csv(data_path+r'\data_ip.csv',index_col=0)
    data_ip[['ice_on','ice_off']] = data_ip[['ice_on','ice_off']].apply(pd.to_datetime)
    data_ip['ice_on_doy'] = np.where(data_ip.ice_on.dt.year < data_ip.year,
                                        data_ip.ice_on.dt.day_of_year,
                                        data_ip.ice_on.dt.day_of_year + 365)
    data_ip['ice_off_doy'] = np.where(data_ip.ice_off.dt.year == data_ip.year,
                                        data_ip.ice_off.dt.day_of_year,
                                        data_ip.ice_off.dt.day_of_year - 365)

    data_ip['ice_cover_duration'] = (data_ip.ice_off-data_ip.ice_on).dt.days
    data_ip = data_ip[(data_ip.ice_cover_duration <= 365) & (data_ip.ice_cover_duration >= 0)]

    # Opening and cleaning lake information:
    ltbl_ice = pd.read_csv(data_path+r'\ltbl_ice.csv',index_col=0)
    ltbl_ice = ltbl_ice.drop(columns=['subset','depth_mean_m','cent_lat_wgs84','cent_lon_wgs84'])
    # Choosing lakes, for which we know max depth:
    ltbl_ice_depth = ltbl_ice[~ltbl_ice.depth_max_m.isna()]
    ltbl_ice_depth = ltbl_ice_depth.reset_index(drop=True)
    
    # Inner merge with no missing values:
    df_ice_clean = pd.merge(data_ip,ltbl_ice_depth,how='inner',on='station_id').dropna(axis=0)
    df_ice_clean = df_ice_clean.reset_index(drop=True)

    return data_ip,ltbl_ice, ltbl_ice_depth, df_ice_clean


def open_era5_monthly_summary(path_to_file):
    '''
    Input to this function should be path to file 'era5_monthly_summary.csv'.
    Function opens this file and modifies the data frame and its columns.
    After this modification, the data frame will be formatted in a way,
    so that it can be merged with the data frame 'df_ice_clean, which
    is a third output of function 'lake_files()'.

    NOTE: The output data frame of this function contains some NaN-values.
    '''
    era5_summary_file = pd.read_csv(path_to_file,
                                header=[0,1],
                                index_col=[0,1,2],
                                low_memory=False).unstack() # need to unstack to get month out of row indexes

    era5_summary_file.columns = era5_summary_file.columns.to_flat_index() # A way to flatten columns and change their name

    era5_summary_file.columns = [x[0]+'_'+x[1]+'_'+str(x[2]) for x in era5_summary_file.columns] # Tuple-columns to string-columns
    return era5_summary_file


def open_statistics_data_full_clean(path_to_file):
    '''
    Input: path to file 'statistics_data_full_clean.csv'

    Function opens the file and changes datatype for datetime columns.
    
    Output: Cleaned file as data frame.
    '''
    df = pd.read_csv(path_to_file,
                      index_col=0,
                      low_memory=False)
    df[['ice_on','ice_off']] = df[['ice_on','ice_off']].apply(pd.to_datetime)
    return df


def open_era5_met_csv(path_to_file):
    '''
    Input: path to 'era5_met.csv'

    Function opens the 'era5_met.csv', which includes meteorological information connected lake stations.
    Column 'date' is turned into datetime-format.
    New datetime-observation-based columns 'day', 'month' and 'year' are created.

    Ouput: function returns modified era5_met data frame.
    '''
    era5_met = pd.read_csv(path_to_file,index_col=0)
    era5_met['date'] = pd.to_datetime(era5_met['date'])
    era5_met['day'] = era5_met.date.dt.day
    era5_met['month'] = era5_met.date.dt.month
    era5_met['year'] = era5_met.date.dt.year
    return era5_met










