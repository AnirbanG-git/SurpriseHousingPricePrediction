# Surprise Housing Market Analysis
> Surprise Housing, a US-based housing company, leverages data analytics to identify and invest in undervalued properties. Entering the Australian market, the company aims to use its analytical prowess to predict the actual value of prospective properties, enabling strategic investments that promise high returns.


## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
- **Background:** Surprise Housing has collected a dataset from the sale of houses in Australia to support its market entry strategy. The project's goal is to build a regression model that can predict the actual value of houses based on various independent variables.
- **Business Problem:** The company seeks to understand how house prices vary with different variables to inform investment decisions and identify high-return opportunities in the Australian real estate market.
- **Dataset Overview:** The dataset includes features related to property characteristics such as area, quality of construction, age, and other relevant attributes that influence house prices.

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Conclusions
### EDA
- **Class Imbalance:** Identified dominant categories within variables that may skew predictive models, highlighting the need for balancing techniques.
- **Sparse and High-Cardinality Features:** Noted challenges posed by rare categories and variables with many categories, suggesting potential benefits from combining sparse classes and reducing dimensionality.
- **Numerical Variable Skewness:** Observed right-skewed distributions of numerical variables, indicating the need for data transformation.
- **Outlier Impact:** Detected outliers that could influence model predictions, necessitating careful assessment and potential data cleaning.
- **Feature Engineering Opportunity:** Recognized opportunities for creating interaction terms and investigating variable combinations for enriched modeling.

### Modelling
- **Significant Variables:** Identified key predictors of house prices, such as TotalSqFt, OverallQual, Age, and GarageCars_GarageArea.
- **Linear Relationships:** Observed clear linear relationships between several variables and SalePrice, underlining their predictive power.
- **Model Comparison:** The Lasso model outperformed the Ridge model in terms of R-squared, RMSE, MSE, and RSS, indicating better predictive accuracy and model fit.
- **Predictive Indicators:** Highlighted categories with consistently higher median SalePrice values as important predictors for property valuation.
- **Regularisation Impact:** Emphasized the benefits of employing Ridge Regression to address multicollinearity among predictors in our dataset.

## Recommendations
- **Significant Variables for Price Prediction:** The analysis identified TotalSqFt, OverallQual, Age, GarageCars_GarageArea, LotArea, and SaleCondition_Partial as significant predictors of house prices. These variables should be given priority in the evaluation of properties for investment.
- **Descriptive Power of Significant Variables:** The significant variables collectively offer a comprehensive understanding of house prices. Their combined effect captures not only the physical attributes of properties but also qualitative aspects and situational factors, providing a holistic view of the market dynamics. It is recommended that Surprise Housing focuses on properties that align well with these key predictors to maximize investment returns.

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
- Python - version 3.8.18
- pandas - version 2.0.3
- numpy - version 1.22.3
- matplotlib - 3.7.2
- seaborn - 0.12.2
- anaconda - 23.5.2

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->


## Contact
Created by [@AnirbanG-git] - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->
