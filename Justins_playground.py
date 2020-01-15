import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling as pdp
np.random.seed(20)

import sys
print(sys.version)
os.chdir('C:\\Users\\howar\\Documents\\Machine_Learning1\\home-credit-default-risk')
# reading in data
train = pd.read_csv('application_train.csv')
test = pd.read_csv('application_test.csv')
####### TRAIN initial exploration
# examining features
train.head()
print(train.info)
#checking NaN values
pd.options.display.max_rows = 200
na_vals = train.isna().sum()/len(train)*100
print(pd.Series.sort_values(na_vals, ascending= False))
"""
We can start by imputing the values with the fewest missing
EXT_SOURCE_3                    19.825307
AMT_REQ_CREDIT_BUREAU_QRT       13.501631
AMT_REQ_CREDIT_BUREAU_YEAR      13.501631
AMT_REQ_CREDIT_BUREAU_WEEK      13.501631
AMT_REQ_CREDIT_BUREAU_MON       13.501631
AMT_REQ_CREDIT_BUREAU_DAY       13.501631
AMT_REQ_CREDIT_BUREAU_HOUR      13.501631
NAME_TYPE_SUITE                  0.420148
OBS_30_CNT_SOCIAL_CIRCLE         0.332021
OBS_60_CNT_SOCIAL_CIRCLE         0.332021
DEF_60_CNT_SOCIAL_CIRCLE         0.332021
DEF_30_CNT_SOCIAL_CIRCLE         0.332021
EXT_SOURCE_2                     0.214626
AMT_GOODS_PRICE                  0.090403
AMT_ANNUITY                      0.003902
CNT_FAM_MEMBERS                  0.000650
DAYS_LAST_PHONE_CHANGE           0.000325
"""

ints = train.dtypes[train.dtypes == np.int64]
intslist = list(ints.index)

floats = train.dtypes[train.dtypes == np.float64]
floatslist = list(floats.index)

nominals = train.dtypes[train.dtypes == np.object]
nominalslist = list(nominals.index)
# checking nominals for high dimensionality
for col in nominalslist:
    print(train[col].value_counts())

# will recode all Y/N values to 1 = Y, 0 = N

cleanYN = {'FLAG_OWN_CAR': {'Y': 1, 'N': 0},
            'FLAG_OWN_REALTY': {'Y': 1, 'N': 0}
            }
train.replace(cleanYN, inplace = True)
# this did not work...trying alternative, what if they're cateogories?
# changing objects to categories, file size shrinks from 286Mb to 253Mb
for col in nominalslist:
    train[col] = train[col].astype('category')
# moving on to imputations
to_impute = ['EXT_SOURCE_3','AMT_REQ_CREDIT_BUREAU_QRT',
'AMT_REQ_CREDIT_BUREAU_YEAR',
'AMT_REQ_CREDIT_BUREAU_WEEK',
'AMT_REQ_CREDIT_BUREAU_MON',
'AMT_REQ_CREDIT_BUREAU_DAY',
'AMT_REQ_CREDIT_BUREAU_HOUR',
'NAME_TYPE_SUITE',
'OBS_30_CNT_SOCIAL_CIRCLE',         
'OBS_60_CNT_SOCIAL_CIRCLE',         
'DEF_60_CNT_SOCIAL_CIRCLE',         
'DEF_30_CNT_SOCIAL_CIRCLE',        
'EXT_SOURCE_2',               
'AMT_GOODS_PRICE',           
'AMT_ANNUITY',         
'CNT_FAM_MEMBERS', 
'DAYS_LAST_PHONE_CHANGE']
# check to make sure filling median makes sense
for col in to_impute:
    print(train[col].describe)
# AMT_REQ columns are known credit requests - impute median is OK
# NAME_TYPE_SUITE who was with client when they applied for loan
    # mode = Unaccompanied, .42% NaN values; will impute
# OBS_CNT_SOCIAL_CIRCLE - number of people in social circle who defaulted - not ok to impute
    # all NaN by the same amount...delete rows?
#EXT_SOURCE no definition provided...impute
imp_train = train
for col in to_impute:
    if imp_train[col].dtype == 'O':
        imp_train.fillna(imp_train[col].mode())
    else:
        imp_train[col].fillna(imp_train[col].median())

pd.options.display.max_rows = 200
na_vals = imp_train.isna().sum()/len(imp_train)*100
print(pd.Series.sort_values(na_vals, ascending= False))

# maybe we can put all the other data into this, too??

"""
We will look at the other datasets 
"""

"""
bureau
"""
bureau = pd.read_csv('bureau.csv')
bureau.head()
bureau.shape # (1716428, 17)
print(bureau.info)
na_vals = bureau.isna().sum()/len(bureau)*100
print(pd.Series.sort_values(na_vals, ascending= False))
# AMT_ANNUITY, AMT_CREDIT_MAX_OVERDUE over 60% Nan
"""
FINGINGS
bureau is a huge dataset of of transactions reported to the credit bureau
ex: SK_ID_CURR = 100001  has 7 credit transactions

Features:

% debt / credit ratio
% debt overdue
mean # loans prolonged
# loans per customer
# types of past loans / customer
mean number of past loans per type / customer
% active loans
mean number of days between successive past applications for each customer
mean # days in which credit expires in the future

Time Series Notes:
more recent data carries higher value than older data
"""

"""
bureau_balance.csv
can be joined with bureau via SK_ID_BUREAU
"""
b_balance = pd.read_csv('bureau_balance.csv')
b_balance.head()
b_balance.shape # (27299925, 3)
# Status of Credit Bureau loan during the month 
# (active, closed, DPD0-30,… 
# [C means closed, X means status unknown, 
# 0 means no DPD, 1 means maximal did during month between 1-30, 
# 2 means DPD 31-60,… 5 means DPD 120+ or sold or written off ] )

print(b_balance.info)

na_vals = b_balance.isna().sum()/len(b_balance)*100
print(pd.Series.sort_values(na_vals, ascending= False))
b_balance['STATUS'].unique() #['C', '0', 'X', '1', '2', '3', '5', '4']
"""
No missing values, another huge dataset
MONTHS_BALANCE = Month of balance relative to application date (-1 means the freshest balance date)
STATUS =  Status of Credit Bureau loan during the month 
(active, closed, DPD0-30,… 
[C means closed, X means status unknown, 
0 means no DPD, 1 means maximal did during month between 1-30, 
2 means DPD 31-60,… 5 means DPD 120+ or sold or written off ] )

this is how we create the time series data!

groupby SK_ID_BUREAU; each value of MONTHS_BALANCE is a backshift operator; 
STATUS can be recoded as ordinal values
X: no data
C: closed account
0: no days past due
1: 1 past due in last 30 days
2: past due for 31-60 days
3: past due for 3 months
4: past due for 4 months
5: past due for 120+ days / sold / written off
"""

card_balance = pd.read_csv('credit_card_balance.csv')
card_balance.head()
card_balance.shape # (3840312, 23)
print(card_balance.info)

na_vals = card_balance.isna().sum()/len(card_balance)*100
print(pd.Series.sort_values(na_vals, ascending= False))
"""
NaN report
AMT_PAYMENT_CURRENT           19.998063
AMT_DRAWINGS_OTHER_CURRENT    19.524872
CNT_DRAWINGS_POS_CURRENT      19.524872
CNT_DRAWINGS_OTHER_CURRENT    19.524872
CNT_DRAWINGS_ATM_CURRENT      19.524872
AMT_DRAWINGS_ATM_CURRENT      19.524872
AMT_DRAWINGS_POS_CURRENT      19.524872
CNT_INSTALMENT_MATURE_CUM      7.948208
AMT_INST_MIN_REGULARITY        7.948208
"""
payments = pd.read_csv('installments_payments.csv')
payments.head()
payments.shape # (13605401, 8)
print(payments.info)

na_vals = payments.isna().sum()/len(payments)*100
print(pd.Series.sort_values(na_vals, ascending= False))
""" 
mostly present, AMT_Payment, DAYS_ENTRY_PAYMENT .021% NaN
"""

POS_balance = pd.read_csv('POS_CASH_balance.csv')
POS_balance.head()
POS_balance.shape # (10001358, 8)
print(POS_balance.info)

na_vals = POS_balance.isna().sum()/len(POS_balance)*100
print(pd.Series.sort_values(na_vals, ascending= False))
""" NaN Report
CNT_INSTALMENT_FUTURE    0.260835
CNT_INSTALMENT           0.260675
"""

prev_app = pd.read_csv('previous_application.csv')
prev_app.head()
prev_app.shape # (1670214, 37)
print(prev_app.info)

na_vals = prev_app.isna().sum()/len(prev_app)*100
print(pd.Series.sort_values(na_vals, ascending= False))
"""
NaN Report

na_vals = prev_app.isna().sum()/len(prev_app)*100...
RATE_INTEREST_PRIVILEGED       99.643698
RATE_INTEREST_PRIMARY          99.643698
RATE_DOWN_PAYMENT              53.636480
AMT_DOWN_PAYMENT               53.636480
NAME_TYPE_SUITE                49.119754
DAYS_TERMINATION               40.298129
NFLAG_INSURED_ON_APPROVAL      40.298129
DAYS_FIRST_DRAWING             40.298129
DAYS_FIRST_DUE                 40.298129
DAYS_LAST_DUE_1ST_VERSION      40.298129
DAYS_LAST_DUE                  40.298129
AMT_GOODS_PRICE                23.081773
AMT_ANNUITY                    22.286665
CNT_PAYMENT                    22.286366
PRODUCT_COMBINATION             0.020716
"""
test = pd.read_csv('application_test.csv')
test.head()
test.shape # (1670214, 37)
print(test.info)

na_vals = test.isna().sum()/len(test)*100
print(pd.Series.sort_values(na_vals, ascending= False))


"""
Grouping by the type of loan : 'NAME_CONTRACT_TYPE'
Why do so many features have NaN values for home info? Do we need an 
ensemble model for the types of loans?
"""

loan_types = train.groupby('NAME_CONTRACT_TYPE').sum()
print(loan_types)
#sns.plt.show()
"""
Groupby Target to make some basic visualizations
"""

defaults = train.groupby('TARGET').sum()
defaults_cat = train.groupby('TARGET').count()

defaultsHouse_loan = pd.crosstab([train['NAME_HOUSING_TYPE'], train['NAME_CONTRACT_TYPE']],
                            train.TARGET.astype(bool))
print(defaultsHouse_loan)

revolving = train['NAME_CONTRACT_TYPE'] == 'Revolving loans'
loanDefaults = pd.crosstab([train['TARGET'], train['NAME_HOUSING_TYPE']])
print(loanDefaults)
"""
VISUALIZATIONS
"""
# We will begin visualizing variables initially judged to me important
# from the application_train.csv

# Defaults by house type and loan type
defaultsHouse_loan.plot(kind = 'barh')
plt.suptitle("Defaults by Housing and Loan Type ")
plt.xlabel('Default Count')
plt.ylabel('Housing and Loan Types')
plt.show()
"""
A vast majority of the loans are cash loans, which explains why 
the housing statistics are mostly NaN values. We might want to 
subset the data into cash loans to people with homes/apartments
"""

defaults_cat['NAME_CONTRACT_TYPE'].plot(kind = 'barh')
plt.suptitle('Total Defaults')
plt.ylabel('Loan Status')
plt.xlabel('Count')
plt.show()
# trying a subset : Cash loans to people in a house/apartment

cash = train['NAME_CONTRACT_TYPE']== ''
TARGET = imp_train.iloc[:,1]
train_large_scale = train_float_imp.iloc[:,0:4]
train_large_scale.loc[:,'TARGET'] = TARGET
train_large_scale.head()

sns.violinplot( y=train["AMT_INCOME_TOTAL"], x=train["TARGET"] )
sns.set()

stuff = pd.melt(train, 'NAME_CONTRACT_TYPE', var_name='AMT_INCOME_TOTAL')
stuff['TARGET'] = stuff['TARGET'].astype('category')
sns.swarmplot(x = 'NAME_CONTRACT_TYPE', y = 'AMT_INCOME_TOTAL', hue = 'TARGET', data = stuff)
sns.set(style="white")

g = sns.PairGrid(train_large_scale, diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_upper(sns.scatterplot, hue = 'TARGET')
g.map_diag(sns.kdeplot, lw=3)

sns.PairGrid(train_large_scale)
sns.pairplot(train_large_scale, kind="scatter", hue="TARGET", markers=["o", "D"], palette="Set2")
plt.show()



