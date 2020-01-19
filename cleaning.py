"""Cleaning
Utilities for cleaning the data
"""

import numpy as np
import pandas as pd
from pandas import DataFrame

def encode_days_employed(days: float) -> str:
    """Encode an Employed column based on DAYS_EMPLOYED
    The values of 365E3 will be encoded as 'N', all others will be encoded as 'Y'.
    """
    if days > 200000:
        return 'N'
    else:
        return 'Y'

def pos_to_zero(value: float) -> float:
    """Return 0 for positive numbers, else the number
    """
    if value >= 0.0:
        return 0
    else:
        return value
    
def read_clean_data(path: str = './application_train.csv') -> DataFrame:
    """Reads data and cleans the data set
    
    Cleaning Steps
      * Read csv with Pandas (setting correct data types)
      * Drop columns that will not be used
      * Recode NA values that are not listed as `np.nan`
      * Formattings
      * Encode categorical variables
      * Create new features
    
    Inputs
        path: str
        Path to the data files
    Returns 
        DataFrame with cleaned data
    """
    
    # read the data
    data = pd.read_csv(path,
                       dtype = {
                           'SK_ID_CURR':np.uint32,
                           'DAYS_EMPLOYED':np.int32,
                           'FLAG_MOBIL':str,
                           'FLAG_EMP_PHONE':str,
                           'FLAG_WORK_PHONE':str,
                           'FLAG_CONT_MOBILE':str,
                           'FLAG_PHONE':str,
                           'FLAG_EMAIL':str,
                           'HOUR_APPR_PROCESS_START':str,
                           'REG_REGION_NOT_LIVE_REGION':str,
                           'REG_REGION_NOT_WORK_REGION':str,
                           'LIVE_REGION_NOT_WORK_REGION':str,
                           'REG_CITY_NOT_LIVE_CITY':str,
                           'REG_CITY_NOT_WORK_CITY':str,
                           'LIVE_CITY_NOT_WORK_CITY':str,
                           'FLAG_DOCUMENT_2':str,
                           'FLAG_DOCUMENT_3':str,
                           'FLAG_DOCUMENT_4':str,
                           'FLAG_DOCUMENT_5':str,
                           'FLAG_DOCUMENT_6':str,
                           'FLAG_DOCUMENT_7':str,
                           'FLAG_DOCUMENT_8':str,
                           'FLAG_DOCUMENT_9':str,
                           'FLAG_DOCUMENT_10':str,
                           'FLAG_DOCUMENT_11':str,
                           'FLAG_DOCUMENT_12':str,
                           'FLAG_DOCUMENT_13':str,
                           'FLAG_DOCUMENT_14':str,
                           'FLAG_DOCUMENT_15':str,
                           'FLAG_DOCUMENT_16':str,
                           'FLAG_DOCUMENT_17':str,
                           'FLAG_DOCUMENT_18':str,
                           'FLAG_DOCUMENT_19':str,
                           'FLAG_DOCUMENT_20':str,
                           'FLAG_DOCUMENT_21':str,
                           'AMT_REQ_CREDIT_BUREAU_HOUR':str,
                           'AMT_REQ_CREDIT_BUREAU_DAY':str,
                           'AMT_REQ_CREDIT_BUREAU_WEEK':str,
                           'AMT_REQ_CREDIT_BUREAU_MON':str,
                           'AMT_REQ_CREDIT_BUREAU_QRT':str,
                           'AMT_REQ_CREDIT_BUREAU_YEAR':str
                       },
                       na_values = ['XNA'])

    # define columns to drop
    drop_columns = ['APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG',
                    'COMMONAREA_AVG', 'ELEVATORS_AVG','ENTRANCES_AVG',
                    'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG',
                    'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG',
                    'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE',
                    'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE',
                    'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE',
                    'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 
                    'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI',
                    'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 
                    'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI',
                    'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI', 
                    'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'TOTALAREA_MODE', 'WALLSMATERIAL_MODE',
                    'EMERGENCYSTATE_MODE']
    # drop them
    data.drop(labels = drop_columns, axis = 1, inplace = True)

    # Invert negations and correct data types
    data.DAYS_BIRTH = (-data.DAYS_BIRTH).astype(np.uint16)
    data.DAYS_REGISTRATION = (-data.DAYS_REGISTRATION).astype(np.uint16)
    data.DAYS_ID_PUBLISH = (-data.DAYS_ID_PUBLISH).astype(np.uint16)
    data.DAYS_LAST_PHONE_CHANGE = (-data.DAYS_LAST_PHONE_CHANGE)
    
    # Create an encoding for DAYS_EMPLOYED
    data['EMPLOYED'] = df.DAYS_EMPLOYED.apply(encode_days_employed)
    # Set the large values in daye emplyed to 0, negate all, and retype
    data['DAYS_EMPLOYED'] = (-df.DAYS_EMPLOYED.apply(pos_to_zero)).astype(np.uint16)
    
    # Recode 0 / 1 to N / Y
    data.FLAG_MOBIL.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.FLAG_EMP_PHONE.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.FLAG_WORK_PHONE.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.FLAG_CONT_MOBILE.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.FLAG_PHONE.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.FLAG_EMAIL.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.REG_REGION_NOT_LIVE_REGION.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.REG_REGION_NOT_WORK_REGION.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.LIVE_REGION_NOT_WORK_REGION.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.REG_CITY_NOT_LIVE_CITY.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.REG_CITY_NOT_WORK_CITY.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.LIVE_CITY_NOT_WORK_CITY.replace(to_replace = {'0':'N','1':'Y'},inplace = True)

    data.FLAG_DOCUMENT_2.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.FLAG_DOCUMENT_3.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.FLAG_DOCUMENT_4.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.FLAG_DOCUMENT_5.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.FLAG_DOCUMENT_6.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.FLAG_DOCUMENT_7.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.FLAG_DOCUMENT_8.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.FLAG_DOCUMENT_9.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.FLAG_DOCUMENT_10.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.FLAG_DOCUMENT_11.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.FLAG_DOCUMENT_12.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.FLAG_DOCUMENT_13.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.FLAG_DOCUMENT_14.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.FLAG_DOCUMENT_15.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.FLAG_DOCUMENT_16.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.FLAG_DOCUMENT_17.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.FLAG_DOCUMENT_18.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.FLAG_DOCUMENT_19.replace(to_replace = {'0':'N','1':'Y'},inplace = True)
    data.FLAG_DOCUMENT_20.replace(to_replace = {'0':'N','1':'Y'}, inplace = True)
    data.FLAG_DOCUMENT_21.replace(to_replace = {'0':'N','1':'Y'}, inplace = True)

    # convert nominal / ordinal variables to categories
    data.CODE_GENDER = data.CODE_GENDER.astype('category')
    data.NAME_CONTRACT_TYPE = data.NAME_CONTRACT_TYPE.astype('category')
    data.FLAG_OWN_CAR = data.FLAG_OWN_CAR.astype('category')
    data.FLAG_OWN_REALTY = data.FLAG_OWN_REALTY.astype('category')
    data.CNT_CHILDREN = data.CNT_CHILDREN.astype('category').cat.reorder_categories([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                                                                    12, 14, 19])
    data.NAME_TYPE_SUITE = data.NAME_TYPE_SUITE.astype('category')
    data.NAME_INCOME_TYPE = data.NAME_INCOME_TYPE.astype('category')
    data.NAME_EDUCATION_TYPE = data.NAME_EDUCATION_TYPE.astype('category')
    data.NAME_FAMILY_STATUS = data.NAME_FAMILY_STATUS.astype('category')
    data.NAME_HOUSING_TYPE = data.NAME_HOUSING_TYPE.astype('category')
    data.FLAG_MOBIL = data.FLAG_MOBIL.astype('category')
    data.FLAG_EMP_PHONE = data.FLAG_EMP_PHONE.astype('category')
    data.FLAG_WORK_PHONE = data.FLAG_WORK_PHONE.astype('category')
    data.FLAG_CONT_MOBILE = data.FLAG_CONT_MOBILE.astype('category')
    data.FLAG_PHONE = data.FLAG_PHONE.astype('category')
    data.FLAG_EMAIL = data.FLAG_EMAIL.astype('category')
    data.OCCUPATION_TYPE = data.OCCUPATION_TYPE.astype('category')
    data.WEEKDAY_APPR_PROCESS_START = data.WEEKDAY_APPR_PROCESS_START.astype('category')

    data.HOUR_APPR_PROCESS_START = data.HOUR_APPR_PROCESS_START.astype('category')
    data.REG_REGION_NOT_LIVE_REGION = data.REG_REGION_NOT_LIVE_REGION.astype('category')
    data.REG_REGION_NOT_LIVE_REGION = data.REG_REGION_NOT_LIVE_REGION.astype('category')
    data.REG_REGION_NOT_WORK_REGION = data.REG_REGION_NOT_WORK_REGION.astype('category')
    data.LIVE_REGION_NOT_WORK_REGION = data.LIVE_REGION_NOT_WORK_REGION.astype('category')
    data.REG_CITY_NOT_LIVE_CITY = data.REG_CITY_NOT_LIVE_CITY.astype('category')
    data.REG_CITY_NOT_WORK_CITY = data.REG_CITY_NOT_WORK_CITY.astype('category')
    data.LIVE_CITY_NOT_WORK_CITY = data.LIVE_CITY_NOT_WORK_CITY.astype('category')

    data.FLAG_DOCUMENT_2 = data.FLAG_DOCUMENT_2.astype('category')
    data.FLAG_DOCUMENT_3 = data.FLAG_DOCUMENT_3.astype('category')
    data.FLAG_DOCUMENT_4 = data.FLAG_DOCUMENT_4.astype('category')
    data.FLAG_DOCUMENT_5 = data.FLAG_DOCUMENT_5.astype('category')
    data.FLAG_DOCUMENT_6 = data.FLAG_DOCUMENT_6.astype('category')
    data.FLAG_DOCUMENT_7 = data.FLAG_DOCUMENT_7.astype('category')
    data.FLAG_DOCUMENT_8 = data.FLAG_DOCUMENT_8.astype('category')
    data.FLAG_DOCUMENT_9 = data.FLAG_DOCUMENT_9.astype('category')
    data.FLAG_DOCUMENT_10 = data.FLAG_DOCUMENT_10.astype('category')
    data.FLAG_DOCUMENT_11 = data.FLAG_DOCUMENT_11.astype('category')
    data.FLAG_DOCUMENT_12 = data.FLAG_DOCUMENT_12.astype('category')
    data.FLAG_DOCUMENT_13 = data.FLAG_DOCUMENT_13.astype('category')
    data.FLAG_DOCUMENT_14 = data.FLAG_DOCUMENT_14.astype('category')
    data.FLAG_DOCUMENT_15 = data.FLAG_DOCUMENT_15.astype('category')
    data.FLAG_DOCUMENT_16 = data.FLAG_DOCUMENT_16.astype('category')
    data.FLAG_DOCUMENT_17 = data.FLAG_DOCUMENT_17.astype('category')
    data.FLAG_DOCUMENT_18 = data.FLAG_DOCUMENT_18.astype('category')
    data.FLAG_DOCUMENT_19 = data.FLAG_DOCUMENT_19.astype('category')
    data.FLAG_DOCUMENT_20 = data.FLAG_DOCUMENT_20.astype('category')
    data.FLAG_DOCUMENT_21 = data.FLAG_DOCUMENT_21.astype('category')

    data.AMT_REQ_CREDIT_BUREAU_HOUR = data.AMT_REQ_CREDIT_BUREAU_HOUR.astype('category')
    data.AMT_REQ_CREDIT_BUREAU_DAY = data.AMT_REQ_CREDIT_BUREAU_DAY.astype('category')
    data.AMT_REQ_CREDIT_BUREAU_WEEK = data.AMT_REQ_CREDIT_BUREAU_WEEK.astype('category')
    data.AMT_REQ_CREDIT_BUREAU_MON = data.AMT_REQ_CREDIT_BUREAU_MON.astype('category')
    data.AMT_REQ_CREDIT_BUREAU_QRT = data.AMT_REQ_CREDIT_BUREAU_QRT.astype('category')
    data.AMT_REQ_CREDIT_BUREAU_YEAR = data.AMT_REQ_CREDIT_BUREAU_YEAR.astype('category')

    data.CNT_FAM_MEMBERS = data.CNT_FAM_MEMBERS.astype('category').cat.reorder_categories([ 1.,  2.,  3.,  4.,  5.,  6., 7.,
                                                                                            8.,  9., 10., 11., 12., 13., 14.,
                                                                                            15., 16., 20.])
    data.REGION_RATING_CLIENT = data.REGION_RATING_CLIENT.astype('category').cat.reorder_categories([1, 2, 3])
    data.REGION_RATING_CLIENT_W_CITY = data.REGION_RATING_CLIENT_W_CITY.astype('category').cat.reorder_categories([1, 2, 3])
    
    
    # create new features
    data['CREDIT_INCOME_RATIO'] = data.AMT_CREDIT / data.AMT_INCOME_TOTAL
    data['ANNUITY_INCOME_RATIO'] = data.AMT_ANNUITY / data.AMT_INCOME_TOTAL
    data['PERCENT_EMPLOYED_TO_AGE'] = data.DAYS_EMPLOYED / data.DAYS_BIRTH

    return data


# define a function to create a table of the missing values
def missing_values_table(df: DataFrame) -> DataFrame:
    """ Displays the missing values for each column raw value and percent of total
    """
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")
    return mis_val_table_ren_columns