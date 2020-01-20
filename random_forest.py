from sklearn.ensemble import RandomForestClassifier 
from pandas import DataFrame
import pandas as pd

def forest_components(dataset: DataFrame) -> DataFrame:
    
    # Instantiate the forest:
    forest = RandomForestClassifier()
    # Drop NaN from data frame for forest to work
    train = train.dropna()
    
    y_train = train['TARGET']
    # the forest didn't like str so not included for now:
    X_train = train.loc[:, train.columns != 'TARGET'].select_dtypes(include=['float64','int64'])

    forest.fit(X_train, y_train)

    feature_importances = pd.DataFrame(forest.feature_importances_,
                                       index = X_train.columns,
                                        columns=['importance']).sort_values('importance',ascending=False)

    return(feature_importances)
