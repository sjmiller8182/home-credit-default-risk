
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas import DataFrame

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier 

RF_PARAMETERS = {'n_estimators':100,'max_depth':None, 'min_samples_split':2,
                 'min_samples_leaf':1,'min_weight_fraction_leaf':0.0,
                 'max_features':'auto','max_leaf_nodes':None,
                 'min_impurity_decrease':0.0,'min_impurity_split':None,
                 'bootstrap':True}

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

class TreeImportances:
    """Estimate feature importances with an ensemble of random forest models
    
    **Process**
    
    * Tune a random forest on the whole dataset with grid or random search
    * Split the data into segments
    * Fit a random forest for each data segment
    * Aggregate importances from each random forest.
    
    """

    def __init__(self, scoring = None,
                       cv: int = 10,
                       random_state = None,
                       njobs = None,
                       verbose = True) -> None:
        """Class constructor
        """
        # set attributes
        self.search_type = None
        self.criterion = None
        self.params = None
        self.n_iter = None
        self.importance_estimators = None
        self.scoring = scoring
        self.cv_size = cv
        self.random_state = random_state
        self.random_generator = np.random.RandomState(random_state)
        self.verbose = verbose
        self.n_jobs = njobs
        self.__importances = None
        self.__best_model = None
        self.__best_parameters = None

    @property
    def importances(self, result_type = None):
        """Get the importances from the random forests
        """
        if self.__importances is None:
            raise Exception("Importances are not populated. Method `fit` must be run first.")
        else:
            return self.__importances

    @property
    def best_parameters(self):
        """Get the best parameters from tuning
        """
        if self.__best_parameters is None:
            raise Exception("Importances are not populated. Method `fit` must be run first.")
        else:
            return self.__best_parameters
    
    def plot_relative_importance(self, plot_type = 'line',
                                 figsize = (10,10),
                                 limit_size = None,
                                 save_path = None):
        """Plot the relative importance of features
        """
        fig, ax = plt.subplots(figsize = figsize)
        if limit_size is None:
            limit_size = len(self.__importances['importance_mean'])
        if plot_type == 'bar':
            y_pos = np.flip(np.arange(len(self.__importances['features'][: limit_size ])))
            ax.barh(y_pos, self.__importances['importance_mean'][: limit_size ]);
            ax.set_title('Feature Importance ('+ self.criterion +')');
            ax.set_yticks(y_pos)
            ax.set_yticklabels(self.__importances['features'][: limit_size ])
            ax.set_xlabel('Importance');
            ax.set_ylabel('Feature');
        elif plot_type == 'line':
            ax.plot(self.__importances['importance_mean']);
            ax.set_title('Feature Importance ('+ self.criterion +')');
            ax.set_xlabel('Feature (Index)');
            ax.set_ylabel('Importance');
        if save_path is not None:
            fig.savefig(save_path)
        
    
    @staticmethod
    def _arrange_params(best_settings, 
                        default_settings,
                        criterion, 
                        n_jobs, 
                        random_state, 
                        verbose):
        """Arrange RF params into list to serialize into model
        """
        # get tuned params and/or params to serialize into models
        # cannot expect that all parameters are search
        # fill in untuned parameters with the default
        settings = list()
        searched_params = best_settings.keys()
        rf_params = default_settings.keys()
        for param in rf_params:
            if param in searched_params:
                # get the best parameter value from search
                settings.append(best_settings[param])
            else:
                # get the default if this paramer wasn't searched
                settings.append(default_settings[param])
        # insert the criterion value
        settings.insert(1, criterion)
        return settings + [False, n_jobs, random_state, verbose]

    @staticmethod
    def _get_importance_estimators(importance_estimators: int,
                                   cv_size: int) -> int:
        """Set the number of models for determining the feature importances
        """
        if importance_estimators is None:
            if cv_size is not None:
                importance_estimators = cv_size
            else:
                importance_estimators = 10
        
        return importance_estimators

    @staticmethod
    def _create_splits(size:int,
                       importance_estimators: int,
                       random_generator: np.random.RandomState) -> np.ndarray:
        """Create splits for the data (shuffled)
        """
        # create splits on the indicies, shuffled
        ind = np.arange(size)
        # shuffle array inplace
        random_generator.shuffle(ind)
        return np.array_split(ind, importance_estimators)

    def fit(self, X, y,
                  search_type, 
                  params: Dict[str, List] = None, 
                  n_iter: int = 10, 
                  criterion: str = 'gini', 
                  importance_estimators: int = None):
        """Tune parameters and fit estimators
        """
        
        if self.verbose:
            grid_verbose = 2
        else:
            grid_verbose = 0
        # tune random forest by CV
        self.search_type = search_type
        if params is not None:
            self.params = params
        else:
            # use default search grid if none is provided
            self.params = {
                'n_estimators': list(np.linspace(start = 500, stop = 1500, num = 5).astype(int)),
                'max_features': ['auto', 'sqrt'],
                'max_depth': list(np.linspace(10, 110, num = 11, dtype = 'int')),
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }
        self.n_iter = n_iter
        self.criterion = criterion
        self.importance_estimators = importance_estimators

        model = RandomForestClassifier(random_state = self.random_state,
                                       n_jobs=self.n_jobs)
        
        if self.search_type == 'grid':
            searcher = GridSearchCV(estimator=model,
                                    param_grid=self.params,
                                    scoring=self.scoring,
                                    cv=self.cv_size,
                                    verbose=grid_verbose,
                                    refit=False)
        else:
            searcher = RandomizedSearchCV(estimator=model,
                                          param_distributions=self.params,
                                          n_iter=self.n_iter,
                                          scoring=self.scoring,
                                          cv=self.cv_size,
                                          verbose=grid_verbose,
                                          refit=False)
        # get best model and parameters (set internal state)
        searcher.fit(X, y)
        #self.__best_model = searcher.best_estimator_
        self.__best_parameters = searcher.best_params_
        
        if self.verbose > 0:
            print(f"The best score from searching is {searcher.best_score_:.4f}.")

        # destroy search object to free up memory
        del searcher
            
        # get tuned params and/or params to serialize into models
        tuned_settings = self._arrange_params(self.__best_parameters,
                                              RF_PARAMETERS,
                                              self.criterion,
                                              self.n_jobs,
                                              self.random_state,
                                              0)

        # set the number of models for determining the feature importances
        self.importance_estimators = self._get_importance_estimators(self.importance_estimators,
                                                                     self.cv_size)
        # create splits on the indicies, shuffled
        cross_val_splits = self._create_splits(X.shape[0],
                                               self.importance_estimators,
                                               self.random_generator)
        if self.verbose:
            print(f"Using {self.importance_estimators} importance estimators.")
        # fit the importance estimators
        # get most important features
        importances = list()
        X_array, y_array = np.asarray(X), np.asarray(y)
        for i, split in enumerate(cross_val_splits):
            if self.verbose:
                print(f"Fitting on split {i}.")
            model = RandomForestClassifier(*tuned_settings)
            model.fit(X_array[split, :], y_array[split])
            importances.append(model.feature_importances_)
        importances = np.column_stack(importances)
        
        
        # create dataframe header
        columns_splits = np.char.add(np.repeat('split_', self.importance_estimators),
                                     np.arange(0, self.importance_estimators, 1).astype(str))
        
        # create dataframe of feature importances
        df_temp = pd.DataFrame(importances, columns=columns_splits)
        df_temp['features'] = X.columns.values
        df_temp['importance_mean'] = np.mean(importances, axis = 1)
        df_temp['importance_sd'] = np.std(importances, axis = 1)
        df_temp = df_temp.reindex(columns = ['features', 'importance_mean', 'importance_sd'] + list(columns_splits))
        self.__importances = df_temp.sort_values('importance_mean', ascending = False).reset_index(drop = True)
