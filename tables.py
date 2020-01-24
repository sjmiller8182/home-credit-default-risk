"""Tables
Procedures for creating tables
"""

import pandas as pd
from pandas import DataFrame, Series

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
