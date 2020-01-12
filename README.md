# An Analysis of Loan Default Risk

## Problem Statement
This analysis of Home Credit's Default Risk dataset will focus on generating accurate loan default risk probabilities. Predicting loan defaults is essential to the profitability of banks and, given the competitive nature of the loan market, a bank that collects the right data can offer and service more loans. The target variable of the dataset is the binary label, 'TARGET', indicating whether the loan entered into default status or not.  
## Data Source
```bash
https://www.kaggle.com/c/home-credit-default-risk/overview
```
Given the binary nature of the target variable, the analytic task is that of classification. The final model will produce the probability of default for each loan and the predicted probabilities will be evaluated on the area under the Reciever Operating Characteristics curve between the predicted probabilitity of default and whether the loans defaulted or not. We believe that a good predictive model is capable of achieving an accuracy between 70% and 80%.

## Data Understanding
#### 1. Exploratory Analysis: Meaning of the data
The dataset consists of 307,511 individual loans

Data Table | Number of Features
------------ | -------------
application_{train|test}.csv |
bureau.csv |
bureau_balance.csv |
POS_CASH_balance.csv |
credit_card_balance.csv |
previous_application.csv |
installments_payments.csv |
HomeCredit_columns_description.csv |

- The data is a collection of nine tables arranged in a schema that is defined here:
![Home Credit Schema](https://storage.googleapis.com/kaggle-media/competitions/home-credit/home_credit.png)

- The data are a mixture of binary indicators, integer values, and continous floating values. The scale of the data varies from nominal categories to ordinal 
	- A detailed description of all features in the dataset can be found [here](/HomeCredit_columns_description.csv).
	- A list of all features and their associated datatypes can be found [here]().
#### 2. Missing Data & Errors
The data is fairly clean, but many categorical features contain NaN values that represent the absence of data. There are sparse features (all FLAG features are binary indicator variables), nominal features (NAME_CONTRACT_TYPE, ORGANIZATION_TYPE), and ordinal features (REGION_RATING_CLIENT,CREDIT_DAY_OVERDUE)

## Univariate Statistics: Critical Findings
- list of visualizations and their interpretations (explain why each finding is important)
## Bivariate Statistics: Critical Findings
#### 1.  relationships between attributes
- list of bivariate visualizations and their interpretations (explain why each finding is important)
#### 2. relationships between target variable and attributes
- list of bivariate visualizations and their interpretations (explain why each finding is important)
## Feature Engineering
- list of potential features and justifications
## Dimensionality Reduction
#### 1. PCA
#### 2. LDA


2. Predictive Analysis
- supporting objective
- supporting objective

## Conclusions