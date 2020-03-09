"""Outliers
Functions for removing outliers
"""
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import stats

def outlier_removal_sigma_float(dataset: DataFrame) -> DataFrame:

    # remove outliers based on standard deviation under assumption of normally distributed data. To prevent
    # elimination of prior knowledge used for classification modeling, we will not transform the data. We assume
    # under the central limit theorem that this sample is representative of the overall population and that as the
    # sample size increases, the distribution will approximate toward a normal distribution centered around the mean.

    # dataset including only float64 (continous numeric) data. NAs are dropped so the z-test can be applied.
    # first, create the numeric set
    num_train = dataset.select_dtypes(include=['float32','float64'])
    # second, create dataset for non-numeric data (non-num_train outliers will join back to this, post-processing)
    cat_train = dataset.select_dtypes(exclude=['float32','float64'])

    # new dataset excluding observations having values beyond the 3rd standard deviation from the mean. This amount
    # allows to keep 99.73% of the data and eliminate the most fringe outliers.
    indx = (np.abs(stats.zscore(num_train)) < 3).all(axis=1)
    train_cleaned_numeric = pd.concat([num_train.loc[indx], cat_train.loc[indx]], axis=1)
    return(train_cleaned_numeric)

def outlier_removal_quartile_integer(dataset: DataFrame) -> DataFrame:
    # third, outlying integers will be dropped based on the inter-quartile range. This allows for more outliers
    # to be dropped than with the z-test hypothesis test.
    num_train_int = dataset.select_dtypes(include=['int32','int64'])
    cat_train_int = dataset.select_dtypes(exclude=['int32','int64'])

    # calculate IQR:
    Q1 = num_train_int.quantile(0.25)
    Q3 = num_train_int.quantile(0.75)
    IQR = Q3 - Q1

    idx = ~((num_train_int < (Q1 - 1.5 * IQR)) | (num_train_int > (Q3 + 1.5 * IQR))).any(axis=1)
    train_cleaned_int = pd.concat([num_train_int.loc[idx], cat_train_int.loc[idx]], axis=1)
    return(train_cleaned_int)
