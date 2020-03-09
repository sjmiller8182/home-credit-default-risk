"""Tables
Procedures for creating tables
"""

import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from numpy import ndarray

from sklearn.metrics import accuracy_score, precision_score, recall_score

def count_values_table(data_series: Series) -> DataFrame:
    """Transform a categorical series into a table of value counts.
    """
    count_val = data_series.value_counts()
    count_val_percent = 100 * data_series.value_counts() / len(data_series)
    count_val_table = pd.concat([count_val, count_val_percent.round(1)], axis=1)
    count_val_table_ren_columns = count_val_table.rename(
    columns = {0 : 'Count Values', 1 : '% of Total Values'})
    count_val_table_ren_columns = count_val_table_ren_columns.iloc[:, :-1]
    return count_val_table_ren_columns

def confusion_matrix(y_true: ndarray, y_pred: ndarray) -> DataFrame:
    """Create a confusion matrix in DataFrame form
    """
    # attempt coercion to ndarray is not correct type
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
        
    return pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

def classification_report(y_test, y_pred) -> None:
    """Print a basic classification report
    """
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("Classification Report")
    print('')
    print('Confusion Matrix')
    print(cm)
    print('')
    print('Accuracy: {:0.2f}%'.format(acc * 100),
          '\nPrecision: {:0.2f}%'.format(precision * 100),
          '\nRecall: {:0.2f}%'.format(recall * 100))