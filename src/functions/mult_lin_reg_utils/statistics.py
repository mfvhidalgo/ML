import pandas as pd
from typing import List, Dict
from statsmodels.formula.api import ols
from statsmodels.regression.linear_model import OLS
import numpy as np

from .terms import list_to_formula

def calc_bic_aicc(model: OLS,
                  key_stat: str = 'aicc') -> float:
    '''
    Returns the bic or aicc

    Parameters:
        model (ols): a fit ols model from statsmodels.formula.api
        key_stat (str): either aicc or bic
        
    Returns:
        float: value of key statistic
    '''
    num_params = len(model.params)
    if key_stat == 'bic':
        return model.bic
    elif key_stat == 'aicc':
        if model.nobs - num_params - 1 == 0:
            return int('inf')
        else:
            return model.aic + (2 * num_params * num_params + 2 * num_params) / (model.nobs - num_params - 1)
    else:
        raise ValueError('key_stat can only be either aicc or bic')

   
def calc_r2_press(model: OLS) -> float:
    """
    Calculates the R-squared predicted residual error sum of squares (PRESS).

    Args:
        model (ols): fit ols model

    Returns:
        float: R2_press
    """

    PRESS = 0
    starting_df = model.model.data.frame.copy()
    starting_df = starting_df.dropna()
    for removed_exp_num in list(starting_df.index):
        df = starting_df.copy()
        removed_row = df.loc[removed_exp_num]
        remaining_df = df.drop(index=removed_exp_num)
        model_after_removal = ols(model.model.formula, data=remaining_df).fit()
        removed_resp_pred = model_after_removal.predict(removed_row).values[0]
        removed_resp_actual = removed_row[model.model.endog_names]
        SS = (removed_resp_pred - removed_resp_actual)**2
        PRESS += SS
    return 1 - PRESS/model.centered_tss
