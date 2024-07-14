import pandas as pd
from typing import List, Dict
from statsmodels.formula.api import ols
from statsmodels.regression.linear_model import OLS
import numpy as np
import dexpy
import dexpy.factorial 
import dexpy.alias
import dexpy.power
from statsmodels.stats.outliers_influence import variance_inflation_factor as var_inf_fact
from patsy import dmatrix

from .terms import list_to_formula, patsy_to_list

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
        removed_resp_pred = model_after_removal.predict(df).loc[removed_exp_num]
        #removed_resp_pred = model_after_removal.predict(removed_row).values[0] replaced bec could not run predict with a df of 1 row and categorical terms
        removed_resp_actual = removed_row[model.model.endog_names]
        SS = (removed_resp_pred - removed_resp_actual)**2
        PRESS += SS
    return 1 - PRESS/model.centered_tss

def evaluate_data(formula: str,
                  data: pd.DataFrame,
                  signal_to_noise: float = 2,
                  alpha: float = 0.05) -> pd.DataFrame:
    """
    Calculates the power, VIF, and Ri2 (Ri_squared) given a formula and data.
    If the full model is aliased, removes terms until aliasing is removed
    then calculates power, VIF, and Ri2.

    Args:
        formula (str): right-hand-side of a patsy formula.
        data (pd.DataFrame): data of the input features.
        signal_to_noise (float): signla-to-noise ratio. Default is 2.
        alpha (float): significance level (risk for Type I error). Default is 0.5.

    Returns:
        pd.DataFrame: df containing the power, VIF, and Ri2 for each term in the model,
        including whether terms are aliased.
    """

    if '~' in formula:
        formula = formula.split('~')[1]
    formula = formula.replace(' ','')
    df = data.copy()
    aliases = dexpy.alias.alias_list(formula, df)[0]

    terms = patsy_to_list(formula)
    power_alias = pd.DataFrame({'power':[np.nan]*len(terms)},index=terms)
    power_alias['power'] = power_alias['power'].astype(object)
    if len(aliases) != 0:
        for alias in aliases:
            keep,aliased = alias.split(' = ')
            power_alias.loc[aliased,'power'] = 'ALIASED'

    aliased_terms = power_alias['power'].dropna().index
    terms_unaliased = terms.copy()
    [terms_unaliased.remove(aliased_term) for aliased_term in aliased_terms]

    formula_unaliased = list_to_formula(terms_unaliased).split('~')[1]
    powers = dexpy.power.f_power(formula_unaliased,
                                df,
                                signal_to_noise,
                                alpha)

    for term,power in zip(terms_unaliased,powers[1:]):
        power_alias.loc[term,'power'] = power

    matrix_unaliased = dmatrix(formula_unaliased, df, return_type='dataframe')

    power_alias['VIF'] = ['']*len(power_alias)
    power_alias['R2i'] = ['']*len(power_alias)
    for i,term in enumerate(matrix_unaliased.columns[1:]):
        vif = var_inf_fact(matrix_unaliased,i+1)
        power_alias.loc[term.replace(' ',''),'VIF'] = vif
        power_alias.loc[term.replace(' ',''),'R2i'] = 1-1/vif
    
    return power_alias

