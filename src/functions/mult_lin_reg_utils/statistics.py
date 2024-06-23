import pandas as pd
from typing import List, Dict
from statsmodels.formula.api import ols
import numpy as np

from .terms import list_to_formula

def calc_bic_aicc(model: ols,
                  key_stat: str = 'aicc'):
    '''
    Returns the bic or aicc

    Parameters:
        model (ols): a fit ols model from statsmodels.formula.api
        key_stat (str): either aicc or bic
    '''
    num_params = len(model.params)
    if key_stat == 'bic':
        return model.bic
    elif key_stat == 'aicc':
        return model.aic + (2 * num_params * num_params + 2 * num_params) / (model.nobs - num_params - 1)
    else:
        raise ValueError('key_stat can only be either aicc or bic')
    


def getModelStats(data: pd.DataFrame,
                  terms: List,
                  response: str,
                  key_stat: str ='aicc',
                  model_type: str = 'Process'):
    '''
    Calculates various useful statistics

    Parameters:
        data (pd.DataFrame): pandas DataFrame containing the data to fit
        terms (List): list of terms to fit the model with
        response (str): column name of the response in data
        key_stat (str): statistic used to either aicc or bic
    '''
    # creates model and returns key statistics, specifically a list of:
    # [r-squared, r-sqared-adj, r-squared-press, value of mod_stat, OLS model]
    # dat_melt is data table in melt format
    # mod_ter is a list of terms in the model, ex. ['A','B','A*B']
    # mod_ty is either process or mixture
    # mod_stat is either aicc or bic

    terms_str = f"{response}~{list_to_formula(terms)}"
    model = ols(terms_str, data=data).fit()

    # get values of n and k (used for calculation AICc and BIC), as well as R2adj and R2press
    num_runs = len(data)
    if model_type == 'Mixture':
        num_params = len(model.params) - 1 # k; '-1' because no intercept
        r2 = 1 - model.ssr/model.centered_tss # R-squared; ols does not know that model is mixture so need to calculate r2 statistics manually
        r2_adj = 1 - (1 - r2) * (num_runs - 1) / (num_runs - num_params - 1) # R-squared-adj
    elif model_type == 'Process':
        num_params = len(model.params)
        r2 = model.rsquared
        r2_adj = model.rsquared_adj
    else:
        raise ValueError('model_type can only be either Mixture or Process')
                
    if key_stat == 'bic':
        key_stat_val = num_params * np.log(num_runs) - 2 * model.llf # BIC
    elif key_stat == 'aicc':
        if num_runs - num_params - 1 != 0: # added since cannot divide by 0
            key_stat_val = -2 * model.llf + 2 * num_params + (2 * num_params * num_params + 2 * num_params) / (num_runs - num_params - 1) # AICc
        else:
            key_stat_val = int('inf')
    else:
        raise ValueError('key_stat can only be either aicc or bic')

    return {'r2':r2,
            'r2_adj':r2_adj,
            'key_stat_val':key_stat_val,
            'model':model}
