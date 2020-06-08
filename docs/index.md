# An Analysis of Loan Default Risk

Created by [Stuart Miller](https://github.com/sjmiller8182),
 [Paul Adams](https://github.com/PaulAdams4361),
 [Justin Howard](https://github.com/juhoward),
 and [Mel Schwan](https://github.com/schwan1).

This analysis of Home Credit's Default Risk dataset will focus on generating accurate loan default risk probabilities. 
Predicting loan defaults is essential to the profitability of banks and,
 given the competitive nature of the loan market,
 a bank that collects the right data can offer and service more loans. 
The target variable of the dataset is the binary label, `TARGET`, indicating whether the loan entered into default status (`1`) or not (`0`).

The final model will produce the probability of default for each loan and the predicted probabilities will be evaluated on the area under the ROC curve. 
We believe that a good predictive model is capable of achieving an accuracy between 70% and 80%.

## Data

The data was provided by Home Credit and is 
[hosted by Kaggle](https://www.kaggle.com/c/home-credit-default-risk/overview).

The dataset consists of 307,511 individual loans. 
For the purpose of this assignment, 
 the analysis will be limited to the initial training and test sets, 
 with the addition of three engineered features obtained from the bureau.csv.

Data Table | Number of Features
------------ | -------------
application_{train}.csv | 122
application_{test}.csv | 121
newFeatures.csv | 3
bureau.csv | 17
bureau_balance.csv | 3
POS_CASH_balance.csv | 8
installments_payments.csv | 8
credit_card_balance.csv | 23
previous_application.csv | 37

## Analysis

This project is broken into three main sections

1. Data Exploration and Understanding
2. Predictive Modeling
3. Clustering Analysis and Segmentation

### Data Exploration

Explored the data to understand the type of features and the relationships between features.

* Provided detailed description of the features and assessed data types.
* Assessed missing values and removed features with too much data corruption.
* Reported interesting features from univariate and multivariate analyses.
* Engineered new features from understanding of given features.

[Report: MiniLab1 CRISP-DM](./notebooks/MiniLab1_CRISP-DM.ipynb): Notebook for data exploration phase.

### Predictive Modeling

* Assessed importance of features with logistic regression and random forests.
* Created three models for predicting loan defaults.
* Tuned model hyperparameters to improve performance.
* Assessed model performances and selected the best model for the application.
* Created and tuned three models for a regression task (secondary objective).
* Provided a model deployment plan.

[Report: MiniLab2 CRISP-DM](./notebooks/MiniLab2_CRISP-DM.ipynb): Notebook for predictive model development.

### Clustering Analysis and Segmentation

* Evaluated several methods for dimensionality reduction.
* Reduced the 300+ dimensions to 2 dimensions with an autoencoder.
* Clustered on the reduced dimension set.
* Provided an interpretation of the cluster.

[Report: Lab3 CRISP-DM](./notebooks/Lab3_CRISP-DM.ipynb): Notebook for clustering and segmentation.


## Main Conclusions

### Predictive Modeling

* We were able to create a list of important features.
* We developed a logistic model with AUC of 0.76. This model did not show evidence of over fitting.

### Clustering and Segmentation

* We were able to create 8 well defined clusters, which can be easily interpreted.
* We were also able to develop a model to classify new customers into the clusters with reasonable AUC.

