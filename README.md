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
 with the addition of three engineered features obtained from the bureau.csv file and labeled ```newFeatures.csv``` .

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


- The data is a collection of nine tables arranged in a schema that is defined here:
![Home Credit Schema](https://storage.googleapis.com/kaggle-media/competitions/home-credit/home_credit.png)

- The data are a mixture of categorical, count, and continous features. The scale of the features vary widely. Large scale data from income features will need scaling to be more precisely compared to binary values. Data types encompass the entire range nominal categories to ordinal.  
	- A detailed description of all features in the dataset can be found [here](/HomeCredit_columns_description.csv).
	- A list of all features and their associated datatypes can be found [here]().

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

**Full Reports**

* [Report (html)](./html/MiniLab1 CRISP-DM\ .html)
* [NoteBook](./notebooks/MiniLab1_CRISP-DM.ipynb)

### Predictive Modeling



### Clustering Analysis and Segmentation



## Conclusion

