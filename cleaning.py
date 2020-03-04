"""Cleaning
Utilities for cleaning the data
"""

from typing import List, Union, Tuple

import numpy as np
from scipy.stats import mode
import pandas as pd

from numpy import ndarray
from pandas import Series
from pandas import DataFrame

def fill_occupation_type(df: DataFrame) -> Series:
    """Fill NAs in occupation type
    * If NAME_INCOME_TYPE == "Unemployed", fill OCCUPATION_TYPE with "None"
    * If NAME_INCOME_TYPE == "Pensioner" and OCCUPATION_TYPE is NA (not str), fill OCCUPATION_TYPE with "Pensioner"
    * Else fill with Unknown
    """
    if df['NAME_INCOME_TYPE'] == "Unemployed":
        return "None"
    elif (df['NAME_INCOME_TYPE'] == "Pensioner") and not isinstance(df['OCCUPATION_TYPE'], str):
        return "Pensioner"
    elif (not isinstance(df['OCCUPATION_TYPE'], str)):
        return "Unknown"
    else:
        return df['OCCUPATION_TYPE']

def encode_days_employed(days: float) -> str:
    """Encode an Employed column based on DAYS_EMPLOYED
    The values of 365E3 will be encoded as 'N', all others will be encoded as 'Y'.
    """
    if days > 200000:
        return '0'
    else:
        return '1'

def pos_to_zero(value: float) -> float:
    """Return 0 for positive numbers, else the number
    """
    if value >= 0.0:
        return 0
    else:
        return value

def encode_available(value: float) -> float:
    """Encode if values are available (np.nan)
    """
    # value exists
    if np.isnan(value):
        return '0'
    # value does not exist
    else:
        return '1'

def get_complete_columns(data: DataFrame) -> List[str]:
    """Get a list of columns without np.nan elements (complete columns)
    """
    return list(data.columns[~data.isnull().any()])
    
def get_mode(data_series: Union[ndarray, Series]) -> Union[str, int]:
    """ Get mode from a series of data
    """
    modal, _ = mode(data_series)
    return modal[0]

# reducing the child count feature to 3 categories
def cnt_child(series):
    if series == 0 :
        return 'No Children'
    elif 1 <= series < 5 :
        return '1-4 Children'
    else :
        return '5 or More Children'
    
# reducing family count feature to 4 categories    
def cnt_family(series):
    if series == 1:
        return '1 Family Member'
    elif series == 2: 
        return '2 Family Members'
    elif 3 >= series <= 5:
        return '3 - -5 Family Members'
    else :
        return '6 or more Family Members'
    
# reducing engineered feature LOAN_COUNT to 5 categories
def loan_count(series):
    if series == 0:
        return 'No Loans'
    elif 1 <= series <= 2:
        return '1-2 Loans'
    elif 3 <= series <= 5:
        return '3-5 Loans'
    elif 6 <= series <= 10:
        return '6-10 Loans'
    else : 
        return ' > 10 Loans'
    
def read_clean_data(path: str = './application_train.csv', preimpute: bool = True) -> DataFrame:
    """Reads data and cleans the data set
    
    Cleaning Steps
      * Read csv with Pandas (setting correct data types)
      * Drop columns that will not be used
      * Recode NA values that are not listed as `np.nan`
      * Formattings
      * Encode categorical variables
      * Create new features
      * perform imputations that do not consider distributions
      * impute features if preimpute is set.
    
    Inputs
        path: str, optional
            Path to the data files
        preimpute: bool, optional
            data impution is based on the total distributions (for backward compatibility)
        
    Returns 
        DataFrame with cleaned data
    """
    
    # read the data
    data = pd.read_csv(path,
                       dtype = {
                           'SK_ID_CURR':np.uint32,
                           'DAYS_EMPLOYED':np.int32,
                           'AMT_REQ_CREDIT_BUREAU_HOUR':np.float64,
                           'AMT_REQ_CREDIT_BUREAU_DAY':np.float64,
                           'AMT_REQ_CREDIT_BUREAU_WEEK':np.float64,
                           'AMT_REQ_CREDIT_BUREAU_MON':np.float64,
                           'AMT_REQ_CREDIT_BUREAU_QRT':np.float64,
                           'AMT_REQ_CREDIT_BUREAU_YEAR':np.float64,
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
    data['EMPLOYED'] = data.DAYS_EMPLOYED.apply(encode_days_employed)
    # Set the large values in daye emplyed to 0, negate all, and retype
    data['DAYS_EMPLOYED'] = (-data.DAYS_EMPLOYED.apply(pos_to_zero)).astype(np.uint16)
    
    # create existance encoding for EXT source columns
    #data['EXT_SOURCE_1_AV'] = data.EXT_SOURCE_1.apply(encode_available)
    #data['EXT_SOURCE_2_AV'] = data.EXT_SOURCE_2.apply(encode_available)
    #data['EXT_SOURCE_3_AV'] = data.EXT_SOURCE_3.apply(encode_available)
    
    # fill nas in EXT_SOURCE_* with 0
    data.EXT_SOURCE_1.fillna(0, inplace = True)
    data.EXT_SOURCE_2.fillna(0, inplace = True)
    data.EXT_SOURCE_3.fillna(0, inplace = True)    
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
##                                                                                                                         ##
### NOTE: we will run read clean data without imputing external source 2 so we can get more accurate regression precision  ##
###    data.EXT_SOURCE_2.fillna(0, inplace = True)                                                                         ##
##                                                                                                                         ##
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################  
    # reducing large numbers of categories
    
    #data['cnt_child'] = data.CNT_CHILDREN.apply(cnt_child).astype('category')
    #data['cnt_family'] = data.CNT_FAM_MEMBERS.apply(cnt_family).astype('category')

    # convert nominal / ordinal variables to categories
    data.CODE_GENDER.replace(to_replace = {'M':0,'F':1}, inplace = True)
    # impute CODE_GENDER with the mode
    if preimpute:
        data.CODE_GENDER = data.CODE_GENDER.fillna(get_mode(data.CODE_GENDER.dropna()))
        # cannot be set to int if np.nan exist
        data.CODE_GENDER = data.CODE_GENDER.astype('int')
    # impute NAME_TYPE_SUITE with the mode
    if preimpute:
        data.NAME_TYPE_SUITE = data.NAME_TYPE_SUITE.fillna(get_mode(data.NAME_TYPE_SUITE.dropna()))

    # set ORGANIZATION_TYPE NAs to 'None' (not associated employer) and make categorical
    data.ORGANIZATION_TYPE = data.ORGANIZATION_TYPE.fillna('None')
    #data.ORGANIZATION_TYPE = data.ORGANIZATION_TYPE.astype('category')
    
    # set missing values in OCCUPATION_TYPE to unknown as it wasn't provided by the client
    data.OCCUPATION_TYPE = data.apply(fill_occupation_type, axis = 1)

    # we will treat missing credit enquiries as no enquiry has occured
    data.AMT_REQ_CREDIT_BUREAU_HOUR =data.AMT_REQ_CREDIT_BUREAU_HOUR.fillna(0).astype(np.uint16)
    data.AMT_REQ_CREDIT_BUREAU_DAY = data.AMT_REQ_CREDIT_BUREAU_DAY.fillna(0).astype(np.uint16)
    data.AMT_REQ_CREDIT_BUREAU_WEEK = data.AMT_REQ_CREDIT_BUREAU_WEEK.fillna(0).astype(np.uint16)
    data.AMT_REQ_CREDIT_BUREAU_MON = data.AMT_REQ_CREDIT_BUREAU_MON.fillna(0).astype(np.uint16)
    data.AMT_REQ_CREDIT_BUREAU_QRT = data.AMT_REQ_CREDIT_BUREAU_QRT.fillna(0).astype(np.uint16)
    data.AMT_REQ_CREDIT_BUREAU_YEAR = data.AMT_REQ_CREDIT_BUREAU_YEAR.fillna(0).astype(np.uint16)
    
    #data.AMT_REQ_CREDIT_BUREAU_HOUR = data.AMT_REQ_CREDIT_BUREAU_HOUR.astype('category')
    #data.AMT_REQ_CREDIT_BUREAU_DAY = data.AMT_REQ_CREDIT_BUREAU_DAY.astype('category')
    #data.AMT_REQ_CREDIT_BUREAU_WEEK = data.AMT_REQ_CREDIT_BUREAU_WEEK.astype('category')
    #data.AMT_REQ_CREDIT_BUREAU_MON = data.AMT_REQ_CREDIT_BUREAU_MON.astype('category')
    #data.AMT_REQ_CREDIT_BUREAU_QRT = data.AMT_REQ_CREDIT_BUREAU_QRT.astype('category')
    #data.AMT_REQ_CREDIT_BUREAU_YEAR = data.AMT_REQ_CREDIT_BUREAU_YEAR.astype('category')

    # impute CNT_FAM_MEMBERS with the mode
    if preimpute:
        data.CNT_FAM_MEMBERS = data.CNT_FAM_MEMBERS.fillna(get_mode(data.CNT_FAM_MEMBERS.dropna()))

    # create new features
    data['CREDIT_INCOME_RATIO'] = data.AMT_CREDIT / data.AMT_INCOME_TOTAL
    data['ANNUITY_INCOME_RATIO'] = data.AMT_ANNUITY / data.AMT_INCOME_TOTAL
    data['PERCENT_EMPLOYED_TO_AGE'] = data.DAYS_EMPLOYED / data.DAYS_BIRTH
    data['CREDIT_LENGTH'] = data.AMT_CREDIT / data.AMT_ANNUITY

    # just set the NAs in OWN_CAR_AGE to 0 because it will be used as an interaction only
    data.OWN_CAR_AGE = data.OWN_CAR_AGE.fillna(0)
    
    # remaining imputions here
    if preimpute:
        data.OBS_30_CNT_SOCIAL_CIRCLE = data.OBS_30_CNT_SOCIAL_CIRCLE.fillna(np.median(data.OBS_30_CNT_SOCIAL_CIRCLE.dropna()))
        data.DEF_30_CNT_SOCIAL_CIRCLE = data.DEF_30_CNT_SOCIAL_CIRCLE.fillna(np.median(data.DEF_30_CNT_SOCIAL_CIRCLE.dropna()))
        data.OBS_60_CNT_SOCIAL_CIRCLE = data.OBS_60_CNT_SOCIAL_CIRCLE.fillna(np.median(data.OBS_60_CNT_SOCIAL_CIRCLE.dropna()))
        data.DEF_60_CNT_SOCIAL_CIRCLE = data.DEF_60_CNT_SOCIAL_CIRCLE.fillna(np.median(data.DEF_60_CNT_SOCIAL_CIRCLE.dropna()))
        data.AMT_GOODS_PRICE = data.AMT_GOODS_PRICE.fillna(np.median(data.AMT_GOODS_PRICE.dropna()))
        data.AMT_ANNUITY = data.AMT_ANNUITY.fillna(np.median(data.AMT_ANNUITY.dropna()))
        data.ANNUITY_INCOME_RATIO = data.ANNUITY_INCOME_RATIO.fillna(np.median(data.ANNUITY_INCOME_RATIO.dropna()))
        data.DAYS_LAST_PHONE_CHANGE = data.DAYS_LAST_PHONE_CHANGE.fillna(np.median(data.DAYS_LAST_PHONE_CHANGE.dropna()))

    return data

def impute_data(training_data: DataFrame, testing_data: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """Calculate imputations from the training data and apply to the testing data
    
    Inputs
        training_data: DataFrame
            training data
        testing_data: DataFrame
            testing data
        
    Returns 
        training_data: imputations based on the training data only
        testing_data: imputations based on the training data only
    """
    
    # calculate from training data, impute into testing data
    
    # make training imputations
    training_data.CODE_GENDER = training_data.CODE_GENDER.fillna(get_mode(training_data.CODE_GENDER.dropna()))
    training_data.NAME_TYPE_SUITE = training_data.NAME_TYPE_SUITE.fillna(get_mode(training_data.NAME_TYPE_SUITE.dropna()))
    training_data.OBS_30_CNT_SOCIAL_CIRCLE = training_data.OBS_30_CNT_SOCIAL_CIRCLE.fillna(np.median(training_data.OBS_30_CNT_SOCIAL_CIRCLE.dropna()))
    training_data.DEF_30_CNT_SOCIAL_CIRCLE = training_data.DEF_30_CNT_SOCIAL_CIRCLE.fillna(np.median(training_data.DEF_30_CNT_SOCIAL_CIRCLE.dropna()))
    training_data.OBS_60_CNT_SOCIAL_CIRCLE = training_data.OBS_60_CNT_SOCIAL_CIRCLE.fillna(np.median(training_data.OBS_60_CNT_SOCIAL_CIRCLE.dropna()))
    training_data.DEF_60_CNT_SOCIAL_CIRCLE = training_data.DEF_60_CNT_SOCIAL_CIRCLE.fillna(np.median(training_data.DEF_60_CNT_SOCIAL_CIRCLE.dropna()))
    training_data.AMT_ANNUITY = training_data.AMT_ANNUITY.fillna(np.median(training_data.AMT_ANNUITY.dropna()))
    training_data.ANNUITY_INCOME_RATIO = training_data.ANNUITY_INCOME_RATIO.fillna(np.median(training_data.ANNUITY_INCOME_RATIO.dropna()))
    training_data.DAYS_LAST_PHONE_CHANGE = training_data.DAYS_LAST_PHONE_CHANGE.fillna(np.median(training_data.DAYS_LAST_PHONE_CHANGE.dropna()))
    
    # make testing imputation based on training data
    testing_data.CODE_GENDER = testing_data.CODE_GENDER.fillna(get_mode(training_data.CODE_GENDER.dropna()))
    testing_data.NAME_TYPE_SUITE = testing_data.NAME_TYPE_SUITE.fillna(get_mode(training_data.NAME_TYPE_SUITE.dropna()))
    testing_data.OBS_30_CNT_SOCIAL_CIRCLE = testing_data.OBS_30_CNT_SOCIAL_CIRCLE.fillna(np.median(training_data.OBS_30_CNT_SOCIAL_CIRCLE.dropna()))
    testing_data.DEF_30_CNT_SOCIAL_CIRCLE = testing_data.DEF_30_CNT_SOCIAL_CIRCLE.fillna(np.median(training_data.DEF_30_CNT_SOCIAL_CIRCLE.dropna()))
    testing_data.OBS_60_CNT_SOCIAL_CIRCLE = testing_data.OBS_60_CNT_SOCIAL_CIRCLE.fillna(np.median(training_data.OBS_60_CNT_SOCIAL_CIRCLE.dropna()))
    testing_data.DEF_60_CNT_SOCIAL_CIRCLE = testing_data.DEF_60_CNT_SOCIAL_CIRCLE.fillna(np.median(training_data.DEF_60_CNT_SOCIAL_CIRCLE.dropna()))
    testing_data.AMT_ANNUITY = testing_data.AMT_ANNUITY.fillna(np.median(training_data.AMT_ANNUITY.dropna()))
    testing_data.ANNUITY_INCOME_RATIO = testing_data.ANNUITY_INCOME_RATIO.fillna(np.median(training_data.ANNUITY_INCOME_RATIO.dropna()))
    testing_data.DAYS_LAST_PHONE_CHANGE = testing_data.DAYS_LAST_PHONE_CHANGE.fillna(np.median(training_data.DAYS_LAST_PHONE_CHANGE.dropna()))

    return training_data, testing_data
    
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

# define a function to load features from bureau table
def load_bureau(path: str = './bureau.csv') -> DataFrame:
    """creates LOAN_COUNT, CREDIT_ACTIVE, CREDIT_DAY_OVERDUE, AMT_CREDIT_SUM, AMT_CREDIT_SUM_DEBT,
               AMT_CREDIT_SUM_LIMIT, AMT_CREDIT_SUM_OVERDUE
    """
    bureau = pd.read_csv(path, 
                         dtype = {
                             'SK_ID_CURR': np.uint32,
                             'SK_BUREAU_ID': np.uint32,
                             'CREDIT_ACTIVE': str,
                             'CREDIT_CURRENCY': str,
                             'DAYS_CREDIT': np.int32,
                             'CREDIT_DAY_OVERDUE': np.int32,
                             'DAYS_CREDIT_ENDDATE': np.float64,
                             'DAYS_ENDDATE_FACT': np.float64,
                             'AMT_CREDIT_MAX_OVERDUE': np.float64,
                             'CNT_CREDIT_PROLONG': np.int32,
                             'AMT_CREDIT_SUM': np.float64,
                             'AMT_CREDIT_SUM_DEBT': np.float64,
                             'AMT_CREDIT_SUM_LIMIT': np.float64,
                             'AMT_CREDIT_SUM_OVERDUE': np.float64,
                             'CREDIT_TYPE': str,
                             'DAYS_CREDIT_UPDATE': np.int32,
                             'AMT_ANNUITY': np.float64
                         })
    numerics = ['DAYS_CREDIT',
                'CREDIT_DAY_OVERDUE',
                'DAYS_CREDIT_ENDDATE',
                'DAYS_ENDDATE_FACT',
                'AMT_CREDIT_MAX_OVERDUE',
                'CNT_CREDIT_PROLONG',
                'AMT_CREDIT_SUM',
                'AMT_CREDIT_SUM_DEBT',
                'AMT_CREDIT_SUM_LIMIT',
                'AMT_CREDIT_SUM_OVERDUE',
                'DAYS_CREDIT_UPDATE',
                'AMT_ANNUITY'
               ]
    
    # reducing the number of categories
    data['credit_active'] = data.CREDIT_ACTIVE.apply(credit_active).astype('category')
    data['loan_cnt'] = data.LOAN_COUNT.apply(loan_count).astype('category')
    # fill all missing data with 0
    bureau[numerics] = bureau[numerics].fillna(0)
    
    return bureau

# define function that creates newFeatures dataframe from bureau
def create_newFeatures(bureau: DataFrame) -> DataFrame:
    #number of loans per applicant
        # start with grouping by applicant ID and counting list length
    loanCounts = bureau.groupby('SK_ID_CURR').count()
    #take just the credit bureau counts
    loanCounts.drop(labels = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY',
       'DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_ENDDATE',
       'DAYS_ENDDATE_FACT', 'AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG',
       'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT',
       'AMT_CREDIT_SUM_OVERDUE', 'CREDIT_TYPE', 'DAYS_CREDIT_UPDATE',
       'AMT_ANNUITY'], axis = 1, inplace = True)
    # rename column of interest
    loanCounts = loanCounts.rename(columns = {'SK_ID_BUREAU':'LOAN_COUNT'})
    
    # count number of active credit accounts, create boolean list
    active = bureau['CREDIT_ACTIVE'] == 'Active'
    # use boolean list to filter bureau
    activeAccounts = bureau[active]
    # collect number of active accounts per applicant
    active_accounts = activeAccounts.groupby('SK_ID_CURR').count()
    # isolate the count of active accounts
    active_accounts.drop(labels = ['SK_ID_BUREAU', 'CREDIT_CURRENCY',
       'DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_ENDDATE',
       'DAYS_ENDDATE_FACT', 'AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG',
       'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT',
       'AMT_CREDIT_SUM_OVERDUE', 'CREDIT_TYPE', 'DAYS_CREDIT_UPDATE',
       'AMT_ANNUITY'], axis = 1, inplace = True)
    # merging data
    data_bureau = loanCounts.merge(active_accounts, on = 'SK_ID_CURR', how = 'left') 
    
    # collect important sums for active credit accounts only   
    sums = activeAccounts.groupby('SK_ID_CURR').sum()
    # drop columns that aren't useful
    sums.drop(labels = ['SK_ID_BUREAU', 'DAYS_CREDIT_ENDDATE','DAYS_ENDDATE_FACT','DAYS_CREDIT',
                      'CNT_CREDIT_PROLONG','AMT_ANNUITY', 'AMT_CREDIT_MAX_OVERDUE', 'DAYS_CREDIT_UPDATE'],
             axis = 1, inplace = True)
    #merging data
    data_bureau = newFeatures.merge(sums, on = 'SK_ID_CURR', how = 'left')
    
    # number of different loan types
    grp = bureau[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(by = ['SK_ID_CURR'])['CREDIT_TYPE'].nunique().reset_index().rename(columns={'CREDIT_TYPE': 'BUREAU_LOAN_TYPES'})
    # merge with data_bureau
    data_bureau = data_bureau.merge(grp, on='SK_ID_CURR', how='left')
    
    
    return newFeatures

# define a function that merges newFeatures with application data
def merge_newFeatures(df: DataFrame) -> DataFrame:
    global newFeatures
    df = df.merge(newFeatures, on = 'SK_ID_CURR', how = 'left')
    #df = df.fillna(0)
    
    return df

    
    
    
    
    
    
    

    