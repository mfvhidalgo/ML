from typing import List, Dict
from statistics import geometric_mean
from scipy import stats
from statsmodels.formula.api import ols
import numpy as np
import pandas as pd

CONF_INT_CRIT = 3.841 # 95% confidence interval value for chi-squared https://www.itl.nist.gov/div898/handbook/eda/section3/eda3674.htm

def best_boxcox_lambda(df_input: pd.DataFrame,
                       formula: str,
                       response: str,
                       lambdas: List = [-2,-1.5,-1,-0.5,0,0.5,1,1.5,2]
                       ) -> Dict:
    '''
    Finds the best lambda for Box-Cox transformations. Note that the complete power transform
    is used to find the best lambda (including the use of a geometric mean), but we prefer to 
    use the simpler lambda transform for the final modelling with box_cox_transform.
    This is because: (1) the best_boxcox_lambda is a statistical analysis/transformation
    while the box_cox_transform is a scientific/physical transformation, and
    (2) we do not know the geometric mean when doing a reverse box_cox_transform, so we skip it

    Parameters:
        df_input (pd.DataFrame): pandas DataFrame containing the data of the features and responses
        formula (str): patsy-style formula for ols
        response (str): column name of the response
        lambdas (List): list of lambda values to try

    Returns:
        Dict: the following keys have the following values:
                lambdas in conf int (List): lambdas with Ln(residual sum of squares) less than the critical value
                                            (i.e. within the confidence interval)
                lambdas (List): all lambdas tested
                ln resid sum squares (List): all Ln(residual sum of squares)
                confidence interval (float): critical Ln(residual sum of squares) value for confidence interval
                best lambda (float): lambda with the lowest Ln(residual sum of squares)
    '''

    df_original = df_input.copy()

    if df_original[response].min() < 0:
        df_original = df_original + abs(df_original[response].min()) + 1
    geom_mean_resp = geometric_mean(df_original[response]) 
    
    models = {} 
    ln_residSS = []
    for lmbda in lambdas:
        df = df_original.copy()
        if lmbda == 0:
            df[response] = stats.boxcox(df[response], lmbda=lmbda) * geom_mean_resp
        else:
            df[response] = stats.boxcox(df[response], lmbda=lmbda) / (geom_mean_resp**(lmbda-1))
        models[lmbda] = ols(formula, data=df).fit()
        ln_residSS.append(np.log(sum(models[lmbda].resid**2)))
    best_resid_lmbda = sorted(zip(ln_residSS,lambdas))[0]
    best_resid,best_lambda = best_resid_lmbda
    
    conf_int_limit = best_resid + CONF_INT_CRIT / models[best_lambda].df_resid
    lambdas_in_conf_int = []
    for lmbda, resid in zip(lambdas,ln_residSS):
        if conf_int_limit >= resid:
            lambdas_in_conf_int.append(lmbda)
    return {'lambdas in conf int':lambdas_in_conf_int,
            'lambdas':lambdas,
            'ln resid sum squares':ln_residSS,
            'confidence interval':conf_int_limit,
            'best lambda':best_lambda
            }

def box_cox_transform(series_input: pd.Series,
                      lmbda: float,
                      reverse: bool = False):
    '''
    Power transform based on Box-Cox transformation. A custom implementaton is needed
    because this functon is slightly different from sklearn's power_transform.
    Specifically, the Box-Cox transformation is normally calculated as (y**lmbda - 1) / lmbda.
    However, this implementation calculates it as (y**lmbda), since we view this as a
    "scientific" power transformation. For example, if the response is time and lmbda == -1,
    time**-1 is just rate, meaning we are still modeling something which has physical meaning.

    Parameters:
        data (pd.Series): pandas Series containing the response
        lmbda (float): lambda value for transformation
        reverse (bool): False if applying the transformation
                        and True if converting transformed data back to the original data
    '''
    
    series = series_input.copy()
        
    if series.min() < 0:
        series = series + abs(series.min()) + 1
        
    if lmbda == 0:
        if reverse:
            series = np.exp(series)
        else:
            series = np.log(series)
    else:
        if reverse:
            series = series**(1/lmbda)
        else:
            series = series**lmbda

    return series