### Feature Descriptions

#### Loan Application Features

The following features are loan application attributes from `application_train.csv`.
Descriptions, types, and and units are given for each feature.

| Feature                      | Description | Type | Units |
|------------------------------|-------------|------|-------|
| SK_ID_CURR                   |  ID of loan in our sample | Category | N/A |
| TARGET                       |  Target Variable (1 - difficulty paying loan, 0 - all other cases) |  Category | N/A  |
| NAME_CONTRACT_TYPE           |  Identification if loan is cash or revolving |  Category | N/A  |
| CODE_GENDER                  |  Gender of the client (M - male, F - female) |  Category | N/A  |
| FLAG_OWN_CAR                 |  Flag if the client owns a car |  Category | N/A  |
| FLAG_OWN_REALTY              |  Flag if client owns a house or flat |  Category | N/A  |
| CNT_CHILDREN                 |  Number of children the client has |  Coninuous | N/A  |
| AMT_INCOME_TOTAL             |  Income of the client |  Coninuous | US Dollar  |
| AMT_CREDIT                   |  Credit amount of the loan |  Coninuous | US Dollar  |
| AMT_ANNUITY                  |  Loan annuity |  Coninuous | US Dollar  |
| AMT_GOODS_PRICE              |  For consumer loans it is the price of the goods for which the loan is given |  Coninuous | US Dollar  |
| NAME_TYPE_SUITE              |  Who was accompanying client when he was applying for the loan |  Category | N/A  |
| NAME_INCOME_TYPE             |  Clients income type (businessman, working, maternity leave) |  Category | N/A  |
| NAME_EDUCATION_TYPE          |  Level of highest education the client achieved |  Category | N/A  |
| NAME_FAMILY_STATUS           |  Family status of the client |  Category | N/A  |
| NAME_HOUSING_TYPE            |  What is the housing situation of the client (renting, living with parents, ...) |  Category | N/A  |
| REGION_POPULATION_RELATIVE   |  Normalized population of region where client lives (higher number means the client lives in more populated region) |  Coninuous | Days  |
| DAYS_BIRTH                   |  Client's age in days at the time of application |  Coninuous | Days  |
| DAYS_EMPLOYED                |  How many days before the application the person started current employment |  Coninuous | Days  |
| DAYS_REGISTRATION            |  How many days before the application did client change his registration |  Coninuous | Days  |
| DAYS_ID_PUBLISH              |  How many days before the application did client change the identity document with which he applied for the loan |  Coninuous | Days  |
| OWN_CAR_AGE                  |  Age of client's car |  Coninuous | Months  |
| FLAG_MOBIL                   |  Did client provide mobile phone (Y, N) |  Category | N/A  |
| FLAG_EMP_PHONE               |  Did client provide work phone (Y, N) |  Category | N/A  |
| FLAG_WORK_PHONE              |  Did client provide home phone (Y, N) |  Category | N/A  |
| FLAG_CONT_MOBILE             |  Was mobile phone reachable (Y, N) |  Category | N/A  |
| FLAG_PHONE                   |  Did client provide home phone (Y, N) |  Category | N/A  |
| FLAG_EMAIL                   |  Did client provide email (Y, N) |  Category | N/A  |
| CNT_FAM_MEMBERS              |  What kind of occupation does the client have |  Category | N/A  |
| OCCUPATION_TYPE              |  How many family members does client have |  Category | N/A  |
| REGION_RATING_CLIENT         |  Our rating of the region where client lives (1,2,3) |  Category | N/A  |
| REGION_RATING_CLIENT_W_CITY  |  Our rating of the region where client lives with taking city into account (1,2,3) |  Category | N/A  |
| WEEKDAY_APPR_PROCESS_START   |  On which day of the week did the client apply for the loan |  Category | N/A  |
| HOUR_APPR_PROCESS_START      |  Approximately at what hour did the client apply for the loan |  Category | N/A  |
| REG_REGION_NOT_LIVE_REGION   | Flag if client's permanent address does not match contact address (1=different, 0=same, at region level)  |  Category | N/A  |
| REG_REGION_NOT_WORK_REGION   |  Flag if client's permanent address does not match work address (1=different, 0=same, at region level) |  Category | N/A  |
| LIVE_REGION_NOT_WORK_REGION  |  Flag if client's contact address does not match work address (1=different, 0=same, at region level) |  Category | N/A  |
| REG_CITY_NOT_LIVE_CITY       |  Flag if client's permanent address does not match contact address (1=different, 0=same, at city level) |  Category | N/A  |
| REG_CITY_NOT_WORK_CITY       |  Flag if client's permanent address does not match work address (1=different, 0=same, at city level) |  Category | N/A  |
| LIVE_CITY_NOT_WORK_CITY      |  Flag if client's contact address does not match work address (1=different, 0=same, at city level) |  Category | N/A  |
| ORGANIZATION_TYPE            | Type of organization where client works  |  Category | N/A  |
| EXT_SOURCE_1                 |  Normalized score from external data source |  Coninuous | N/A  |
| EXT_SOURCE_2                 | Normalized score from external data source  |  Coninuous | N/A  |
| EXT_SOURCE_3                 |  Normalized score from external data source |  Coninuous | N/A  |
| OBS_30_CNT_SOCIAL_CIRCLE     | How many observation of client's social surroundings with observable 30 DPD (days past due) default  |  Coninuous | N/A  |
| DEF_30_CNT_SOCIAL_CIRCLE     |  How many observation of client's social surroundings defaulted on 30 DPD (days past due)  |  Coninuous | N/A  |
| OBS_60_CNT_SOCIAL_CIRCLE     |  How many observation of client's social surroundings with observable 60 DPD (days past due) default |  Coninuous | N/A  |
| DEF_60_CNT_SOCIAL_CIRCLE     |  How many observation of client's social surroundings defaulted on 60 (days past due) DPD | Coninuous  | N/A  |
| DAYS_LAST_PHONE_CHANGE       |  How many days before application did client change phone | Coninuous  | N/A  |
| FLAG_DOCUMENT_2              | Did client provide document 2  |  Category | N/A  |
| FLAG_DOCUMENT_3              | Did client provide document 3  |  Category | N/A  |
| FLAG_DOCUMENT_4              | Did client provide document 4  |  Category | N/A  |
| FLAG_DOCUMENT_5              | Did client provide document 5  |  Category | N/A  |
| FLAG_DOCUMENT_6              | Did client provide document 6  |  Category | N/A  |
| FLAG_DOCUMENT_7              | Did client provide document 7  |  Category | N/A  |
| FLAG_DOCUMENT_8              | Did client provide document 8  |  Category | N/A  |
| FLAG_DOCUMENT_9              | Did client provide document 9  |  Category | N/A  |
| FLAG_DOCUMENT_10             | Did client provide document 10  |  Category | N/A  |
| FLAG_DOCUMENT_11             | Did client provide document 11  |  Category | N/A  |
| FLAG_DOCUMENT_12             | Did client provide document 12  | Category  | N/A  |
| FLAG_DOCUMENT_13             | Did client provide document 13  | Category  | N/A  |
| FLAG_DOCUMENT_14             | Did client provide document 14  | Category  | N/A  |
| FLAG_DOCUMENT_15             | Did client provide document 15  |  Category | N/A  |
| FLAG_DOCUMENT_16             | Did client provide document 16  |  Category | N/A  |
| FLAG_DOCUMENT_17             | Did client provide document 17  |  Category | N/A  |
| FLAG_DOCUMENT_18             | Did client provide document 18  |  Category | N/A  |
| FLAG_DOCUMENT_19             | Did client provide document 19  |  Category | N/A  |
| FLAG_DOCUMENT_20             | Did client provide document 20  |  Category | N/A  |
| FLAG_DOCUMENT_21             | Did client provide document 21  |  Category | N/A  |
| AMT_REQ_CREDIT_BUREAU_HOUR   | Number of enquiries to Credit Bureau about the client one hour before application | Category  | N/A  |
| AMT_REQ_CREDIT_BUREAU_DAY    | Number of enquiries to Credit Bureau about the client one day before application (excluding one hour before application)  | Category  | N/A  |
| AMT_REQ_CREDIT_BUREAU_WEEK   | Number of enquiries to Credit Bureau about the client one week before application (excluding one day before application)  |  Category | N/A  |
| AMT_REQ_CREDIT_BUREAU_MON    |  Number of enquiries to Credit Bureau about the client one month before application (excluding one week before application) |  Category | N/A  |
| AMT_REQ_CREDIT_BUREAU_QRT    | Number of enquiries to Credit Bureau about the client 3 month before application (excluding one month before application)  |  Category | N/A  |
| AMT_REQ_CREDIT_BUREAU_YEAR   |  Number of enquiries to Credit Bureau about the client one day year (excluding last 3 months before application) |  Category | N/A  |

#### Engineered Loan Application Features

The following features were engineered from the loan application features (from `application_train.csv`).

| Engineered Feature           | Description | Type | Units | Formula |
|------------------------------|-------------|------|-------|---------|
| CREDIT_INCOME_INCOME_RATIO   | The percentage credit relative to client's income | Numeric | N/A | `AMT_CREDIT` / `AMT_INCOME_TOTAL` |
| ANNUITY_INCOME_INCOME_RATIO  | The percentage annunity relative to client's income | Numeric | N/A | `AMT_ANNUITY` / `AMT_INCOME_TOTAL` | 
| PERCENT_EMPLOYED_TO_AGE      | The fraction of client's days employed. | Numeric | N/A | `DAYS_EMPLOYED` / `DAYS_BIRTH` |