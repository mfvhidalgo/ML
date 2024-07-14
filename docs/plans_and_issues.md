# Upcoming features
Lists any planned upcoming features and to-do's.

## Pre-modelling data analysis
* Check for homoscedasticity
* Q-Q plot
* plot histogram of each feature
* Pearson correlation coefficient heatmap of terms

## Post-modelling data analysis
* Residuals vs predicted and actual
* Normality of Residuals

## Features
* implement Pearson correlation coefficient heatmap
* implement logistic regression
* implement RF, XGBoost, NN

## Guides
* guide for using Data.xlsx.
* guide for using auto_mlr
* guide for using functions\mult_lin_reg_utils

## Misc
* add error handling if features_coded are not single letters or uses I

# Known issues

## mult_lin_reg_utils
* model_reduction with categorical terms has a different AICc value when there are interactions with categorical terms.
    * BIC is correct, meaning it is likely an issue with either model.aic, model.nobs, or len(model.params), but all were checked to be correct.
    * when trying to find the root cause, we observed that model.nobs and len(model.params) would need to be non-int values to get the correct AICc value, which is impossible.

# Temp
* add dexpy to requirements.txt
* add functions.statistics.evaluate_data