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



