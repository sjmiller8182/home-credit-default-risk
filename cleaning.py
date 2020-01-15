"""Cleaning
Utilities for cleaning the data
"""

import numpy as np
import pandas as pd
from pandas import DataFrame

def read_clean_data(path: str = './application_train.csv') -> DataFrame:
    """Reads data and cleans the data set
    
    Steps
      * Read csv with Pandas (setting correct data types)
      * Cleaning
        * Recode NA values that are not listed as `np.nan`
      * Formattings
        * Encode categorical variables
    Returns 
        DataFrame with cleaned data
    """
    
    # read the data
    data = pd.read_csv(path,
                       dtype = {
                           'SK_ID_CURR':np.uint32,
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

    # fix NAs that are not coded as np.nan
    #data.CODE_GENDER.replace('XNA', np.nan, inplace = True)
    #data.ORGANIZATION_TYPE.replace('XNA', np.nan, inplace = True)

    # Invert negations and correct data types
    data.DAYS_BIRTH = (-data.DAYS_BIRTH).astype(np.uint16)
    data.DAYS_EMPLOYED = (-data.DAYS_EMPLOYED).astype(np.uint16)
    data.DAYS_REGISTRATION = (-data.DAYS_REGISTRATION).astype(np.uint16)
    data.DAYS_ID_PUBLISH = (-data.DAYS_ID_PUBLISH).astype(np.uint16)
    data.DAYS_LAST_PHONE_CHANGE = (-data.DAYS_LAST_PHONE_CHANGE)

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
    
    return data
