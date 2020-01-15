# An Analysis of Loan Default Risk: Executive Summary

## Problem Statement
This analysis of Home Credit's Default Risk dataset will focus on generating accurate loan default risk probabilities. Predicting loan defaults is essential to the profitability of banks and, given the competitive nature of the loan market, a bank that collects the right data can offer and service more loans. The target variable of the dataset is the binary label, 'TARGET', indicating whether the loan entered into default status or not.  
## Data Source
```bash
https://www.kaggle.com/c/home-credit-default-risk/overview
```
Given the binary nature of the target variable, the analytic task is that of classification. The final model will produce the probability of default for each loan and the predicted probabilities will be evaluated on the area under the Reciever Operating Characteristics curve between the predicted probabilitity of default and whether the loans defaulted or not. We believe that a good predictive model is capable of achieving an accuracy between 70% and 80%.

## Data Understanding
#### 1. Exploratory Analysis: Meaning of the data
The dataset consists of 307,511 individual loans. For the purpose of this assignment, the analysis will be limited to the initial training and test sets, with the addition of three engineered features obtained from the bureau.csv file and labeled ```newFeatures.csv``` .

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

- The data are a mixture of binary indicators, integer values, and continous floating values. The scale of the data varies. Large scale data from income features will need scaling to be more precisely compared to binary values. Data types encompass the entire range nominal categories to ordinal.  
	- A detailed description of all features in the dataset can be found [here](/HomeCredit_columns_description.csv).
	- A list of all features and their associated datatypes can be found [here]().

## Conclusions