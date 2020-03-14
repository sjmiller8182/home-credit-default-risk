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
from sklearn.preprocessing import LabelEncoder

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
    data['cnt_child'] = data.CNT_CHILDREN.apply(cnt_child).astype('object')
    data['cnt_family'] = data.CNT_FAM_MEMBERS.apply(cnt_family).astype('object')
    # dropping features with large numberso of mostly empty categories
    data = data.drop(labels = ['CNT_CHILDREN', 'CNT_FAM_MEMBERS'], axis = 1)

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
    
    #label encoding
    categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
    lbl = LabelEncoder()
    for col in categorical_columns:
        data[col] = lbl.fit_transform(data[col].astype(str))

    # impute CNT_FAM_MEMBERS with the mode
    if preimpute:
        data.cnt_family = data.cnt_family.fillna(get_mode(data.cnt_family.dropna()))

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

def merge_bureau(df):
    df = df.copy(deep=True)
    bureau = pd.read_csv('bureau.csv')
    
    # Combining numerical features
    grp = bureau.drop(['SK_ID_BUREAU'], axis = 1).groupby(
        by=['SK_ID_CURR']).mean().reset_index()
    grp.columns = ['BUREAU_'+column if column !='SK_ID_CURR' else column for column in grp.columns]
    data_bureau = df.merge(grp, on='SK_ID_CURR', how='left')
    data_bureau.update(data_bureau[grp.columns].fillna(0))
    # Combining categorical features
    bureau_categorical = pd.get_dummies(bureau.select_dtypes('object'))
    bureau_categorical['SK_ID_CURR'] = bureau['SK_ID_CURR']
    grp = bureau_categorical.groupby(by = ['SK_ID_CURR']).mean().reset_index()
    grp.columns = ['BUREAU_'+column if column !='SK_ID_CURR' else column for column in grp.columns]
    data_bureau = data_bureau.merge(grp, on='SK_ID_CURR', how='left')
    data_bureau.update(data_bureau[grp.columns].fillna(0))
    # Shape of application and bureau data combined
    print('The shape application and bureau data combined:',data_bureau.shape)

    # Number of past loans per customer
    grp = bureau.groupby(by = ['SK_ID_CURR'])['SK_ID_BUREAU'].count().reset_index().rename(columns = {'SK_ID_BUREAU': 'BUREAU_LOAN_COUNT'})
    data_bureau = data_bureau.merge(grp, on='SK_ID_CURR', how='left')
    data_bureau['BUREAU_LOAN_COUNT'] = data_bureau['BUREAU_LOAN_COUNT'].fillna(0)
    # Number of types of past loans per customer 
    grp = bureau[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(by = ['SK_ID_CURR'])['CREDIT_TYPE'].nunique().reset_index().rename(columns={'CREDIT_TYPE': 'BUREAU_LOAN_TYPES'})
    data_bureau = data_bureau.merge(grp, on='SK_ID_CURR', how='left')
    data_bureau['BUREAU_LOAN_TYPES'] = data_bureau['BUREAU_LOAN_TYPES'].fillna(0)
    # Debt over credit ratio 
    bureau['AMT_CREDIT_SUM'] = bureau['AMT_CREDIT_SUM'].fillna(0)
    bureau['AMT_CREDIT_SUM_DEBT'] = bureau['AMT_CREDIT_SUM_DEBT'].fillna(0)
    grp1 = bureau[['SK_ID_CURR','AMT_CREDIT_SUM']].groupby(
        by=['SK_ID_CURR'])['AMT_CREDIT_SUM'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM': 'TOTAL_CREDIT_SUM'})
    grp2 = bureau[['SK_ID_CURR','AMT_CREDIT_SUM_DEBT']].groupby(
        by=['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM_DEBT':'TOTAL_CREDIT_SUM_DEBT'})
    grp1['DEBT_CREDIT_RATIO'] = grp2['TOTAL_CREDIT_SUM_DEBT']/grp1['TOTAL_CREDIT_SUM']
    del grp1['TOTAL_CREDIT_SUM']
    data_bureau = data_bureau.merge(grp1, on='SK_ID_CURR', how='left')
    data_bureau['DEBT_CREDIT_RATIO'] = data_bureau['DEBT_CREDIT_RATIO'].fillna(0)
    data_bureau['DEBT_CREDIT_RATIO'] = data_bureau.replace([np.inf, -np.inf], 0)
    data_bureau['DEBT_CREDIT_RATIO'] = pd.to_numeric(data_bureau['DEBT_CREDIT_RATIO'], downcast='float')
    # Overdue over debt ratio
    bureau['AMT_CREDIT_SUM_OVERDUE'] = bureau['AMT_CREDIT_SUM_OVERDUE'].fillna(0)
    bureau['AMT_CREDIT_SUM_DEBT'] = bureau['AMT_CREDIT_SUM_DEBT'].fillna(0)
    grp1 = bureau[['SK_ID_CURR','AMT_CREDIT_SUM_OVERDUE']].groupby(
        by=['SK_ID_CURR'])['AMT_CREDIT_SUM_OVERDUE'].sum().reset_index().rename(
        columns={'AMT_CREDIT_SUM_OVERDUE': 'TOTAL_CUSTOMER_OVERDUE'})
    grp2 = bureau[['SK_ID_CURR','AMT_CREDIT_SUM_DEBT']].groupby(
        by=['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(
        columns={'AMT_CREDIT_SUM_DEBT':'TOTAL_CUSTOMER_DEBT'})
    grp1['OVERDUE_DEBT_RATIO'] = grp1['TOTAL_CUSTOMER_OVERDUE']/grp2['TOTAL_CUSTOMER_DEBT']
    del grp1['TOTAL_CUSTOMER_OVERDUE']
    data_bureau =data_bureau.merge(grp1, on='SK_ID_CURR', how='left')
    data_bureau['OVERDUE_DEBT_RATIO'] = data_bureau['OVERDUE_DEBT_RATIO'].fillna(0)
    data_bureau['OVERDUE_DEBT_RATIO'] = data_bureau.replace([np.inf, -np.inf], 0)
    data_bureau['OVERDUE_DEBT_RATIO'] = pd.to_numeric(data_bureau['OVERDUE_DEBT_RATIO'], downcast='float')
    print('Dimensions after adding new features: ', data_bureau.shape)
    return data_bureau

def merge_previous_application(df):
    df = df.copy(deep=True)
    previous_application = pd.read_csv('previous_application.csv')
    # Number of previous applications per customer
    grp = previous_application[['SK_ID_CURR','SK_ID_PREV']].groupby(
        by=['SK_ID_CURR'])['SK_ID_PREV'].count().reset_index().rename(columns={'SK_ID_PREV':'PREV_APP_COUNT'})
    data_bureau_prev = df.merge(grp, on =['SK_ID_CURR'], how = 'left')
    data_bureau_prev['PREV_APP_COUNT'] = data_bureau_prev['PREV_APP_COUNT'].fillna(0)
    # Combining numerical features
    grp = previous_application.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean().reset_index()
    prev_columns = ['PREV_'+column if column != 'SK_ID_CURR' else column for column in grp.columns ]
    grp.columns = prev_columns
    data_bureau_prev = data_bureau_prev.merge(grp, on =['SK_ID_CURR'], how = 'left')
    data_bureau_prev.update(data_bureau_prev[grp.columns].fillna(0))
    # Combining categorical features
    prev_categorical = pd.get_dummies(previous_application.select_dtypes('object'))
    prev_categorical['SK_ID_CURR'] = previous_application['SK_ID_CURR']
    prev_categorical.head()
    grp = prev_categorical.groupby('SK_ID_CURR').mean().reset_index()
    grp.columns = ['PREV_'+column if column != 'SK_ID_CURR' else column for column in grp.columns]
    data_bureau_prev = data_bureau_prev.merge(grp, on=['SK_ID_CURR'], how='left')
    data_bureau_prev.update(data_bureau_prev[grp.columns].fillna(0))
    print('Dimensions after adding previous_application: ', data_bureau_prev.shape)
    return data_bureau_prev    

def merge_POS_CASH(df):
    df = df.copy(deep=True)
    pos_cash = pd.read_csv('POS_CASH_balance.csv')
    # Combining numerical features
    grp = pos_cash.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean().reset_index()
    prev_columns = ['POS_'+column if column != 'SK_ID_CURR' else column for column in grp.columns ]
    grp.columns = prev_columns
    data_bureau_prev = df.merge(grp, on =['SK_ID_CURR'], how = 'left')
    data_bureau_prev.update(data_bureau_prev[grp.columns].fillna(0))
    # Combining categorical features
    pos_cash_categorical = pd.get_dummies(pos_cash.select_dtypes('object'))
    pos_cash_categorical['SK_ID_CURR'] = pos_cash['SK_ID_CURR']
    grp = pos_cash_categorical.groupby('SK_ID_CURR').mean().reset_index()
    grp.columns = ['POS_'+column if column != 'SK_ID_CURR' else column for column in grp.columns]
    data_bureau_prev = data_bureau_prev.merge(grp, on=['SK_ID_CURR'], how='left')
    data_bureau_prev.update(data_bureau_prev[grp.columns].fillna(0))
    print('Dimensions after adding POS_CASH_balance: ', data_bureau_prev.shape)
    return data_bureau_prev      

def merge_installments(df):
    df = df.copy(deep=True)
    installments = pd.read_csv('installments_payments.csv')
    # Combining numerical features and there are no categorical features in this dataset
    grp = installments.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean().reset_index()
    prev_columns = ['INSTA_'+column if column != 'SK_ID_CURR' else column for column in grp.columns ]
    grp.columns = prev_columns
    data_bureau_prev = df.merge(grp, on =['SK_ID_CURR'], how = 'left')
    data_bureau_prev.update(data_bureau_prev[grp.columns].fillna(0))
    print('Dimensions after adding installments: ', data_bureau_prev.shape)
    return data_bureau_prev  

def merge_credit_card_balance(df):
    df = df.copy(deep=True)
    credit_card = pd.read_csv('credit_card_balance.csv')
    # Combining numerical features
    grp = credit_card.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean().reset_index()
    prev_columns = ['CREDIT_'+column if column != 'SK_ID_CURR' else column for column in grp.columns ]
    grp.columns = prev_columns
    data_bureau_prev = df.merge(grp, on =['SK_ID_CURR'], how = 'left')
    data_bureau_prev.update(data_bureau_prev[grp.columns].fillna(0))
    # Combining categorical features
    credit_categorical = pd.get_dummies(credit_card.select_dtypes('object'))
    credit_categorical['SK_ID_CURR'] = credit_card['SK_ID_CURR']
    grp = credit_categorical.groupby('SK_ID_CURR').mean().reset_index()
    grp.columns = ['CREDIT_'+column if column != 'SK_ID_CURR' else column for column in grp.columns]
    data_bureau_prev = data_bureau_prev.merge(grp, on=['SK_ID_CURR'], how='left')
    data_bureau_prev.update(data_bureau_prev[grp.columns].fillna(0))
    print('Dimensions after adding credit_card_balance: ', data_bureau_prev.shape)
    return data_bureau_prev    

# https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtypes
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #else: df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
    
def downsampling_strategy(df):
    df = df.copy(deep=True)
    random_state = 1
    defaults = df.query('TARGET == 1')
    nominal = df.query('TARGET == 0').sample(
        n = np.round(0.5 * (defaults.TARGET.size / 0.5)).astype(int), random_state = random_state)
    # join dataframes and shuffle
    df = pd.concat([defaults, nominal]).sample(frac = 1, random_state = random_state)
    return df
    

    
def drop_em(df):
    df = df.copy(deep=True)
    dropped = ['BUREAU_CREDIT_ACTIVE_Active', 'DEBT_CREDIT_RATIO',      
  'PREV_NAME_CONTRACT_TYPE_Cash loans',      
  'PREV_NAME_CONTRACT_TYPE_Consumer loans',     
  'PREV_NAME_CONTRACT_TYPE_Revolving loans',      
  'PREV_NAME_CONTRACT_TYPE_XNA',      
  'PREV_WEEKDAY_APPR_PROCESS_START_FRIDAY',      
  'PREV_FLAG_LAST_APPL_PER_CONTRACT_N',      
  'PREV_NAME_CASH_LOAN_PURPOSE_Building a house or an annex',      
  'PREV_NAME_CASH_LOAN_PURPOSE_XAP',      
  'PREV_NAME_CONTRACT_STATUS_Approved',     
  'PREV_NAME_CONTRACT_STATUS_Unused offer',      
  'PREV_NAME_PAYMENT_TYPE_Cash through the bank',      
  'PREV_CODE_REJECT_REASON_CLIENT',      
  'PREV_NAME_CLIENT_TYPE_New',      
  'PREV_NAME_GOODS_CATEGORY_Additional Service',      
  'PREV_NAME_PORTFOLIO_Cards',      
  'PREV_NAME_PORTFOLIO_Cars',      
  'PREV_NAME_PORTFOLIO_Cash',      
  'PREV_NAME_PORTFOLIO_XNA',      
  'PREV_NAME_PRODUCT_TYPE_XNA',      
  'PREV_CHANNEL_TYPE_AP+ (Cash loan)',      
  'PREV_NAME_SELLER_INDUSTRY_Auto technology',      
  'CREDIT_AMT_RECIVABLE',        
  'PREV_NAME_PORTFOLIO_POS',        
  'CREDIT_AMT_TOTAL_RECEIVABLE',        
  'PREV_NAME_CONTRACT_STATUS_Refused',        
  'CREDIT_AMT_RECEIVABLE_PRINCIPAL',        
  'PREV_NAME_YIELD_GROUP_high',        
  'EMPLOYED',        
  'INSTA_DAYS_INSTALMENT',        
  'PREV_NAME_GOODS_CATEGORY_XNA',        
  'PREV_PRODUCT_COMBINATION_POS household with interest',        
  'OBS_60_CNT_SOCIAL_CIRCLE',       
  'PREV_NAME_CASH_LOAN_PURPOSE_XNA',        
  'CREDIT_SK_DPD_DEF',       
  'PREV_NAME_SELLER_INDUSTRY_XNA',       
  'PREV_NFLAG_LAST_APPL_IN_DAY',     
  'CREDIT_AMT_BALANCE',      
  'PREV_CHANNEL_TYPE_Country-wide',   
  'CREDIT_CNT_DRAWINGS_CURRENT',       
  'PREV_AMT_APPLICATION',    
  'PREV_CODE_REJECT_REASON_XAP',  
  'PREV_NAME_PRODUCT_TYPE_x-sell', 
  'PREV_NAME_GOODS_CATEGORY_Mobile', 
  'INSTA_AMT_PAYMENT', 
  'DAYS_EMPLOYED',
  'POS_CNT_INSTALMENT', 
  'PREV_FLAG_LAST_APPL_PER_CONTRACT_Y',
  'AMT_CREDIT', 
  'PREV_AMT_GOODS_PRICE',  
  'CREDIT_AMT_DRAWINGS_CURRENT', 
  'PREV_PRODUCT_COMBINATION_Cash', 
  'CREDIT_INCOME_RATIO', 
  'PREV_PRODUCT_COMBINATION_POS industry with interest', 
  'FLAG_DOCUMENT_3', 
  'PREV_NAME_SELLER_INDUSTRY_Connectivity',
  'REGION_RATING_CLIENT', 
  'PREV_DAYS_TERMINATION',
  'PREV_NAME_GOODS_CATEGORY_Clothing and Accessories',
  'POS_MONTHS_BALANCE',
  'CREDIT_AMT_PAYMENT_TOTAL_CURRENT',
  'REG_REGION_NOT_WORK_REGION',
  'PREV_DAYS_FIRST_DRAWING',
  'CREDIT_NAME_CONTRACT_STATUS_Active', 
   'PREV_NAME_SELLER_INDUSTRY_Consumer electronics', 
   'PREV_NAME_YIELD_GROUP_XNA', 
   'PREV_AMT_CREDIT',
   'REG_CITY_NOT_WORK_CITY', 
   'PREV_NAME_PRODUCT_TYPE_walk-in',
   'CREDIT_CNT_INSTALMENT_MATURE_CUM',
   'BUREAU_DAYS_CREDIT',
   'PREV_NAME_GOODS_CATEGORY_Furniture',
   'PREV_DAYS_DECISION']
    
    data_slim = df.drop(labels = dropped, axis = 1)
    return data_slim
    
    
    

    