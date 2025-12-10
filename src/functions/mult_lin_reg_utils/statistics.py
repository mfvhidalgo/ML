import pandas as pd
from typing import List, Dict
from statsmodels.formula.api import ols
from statsmodels.regression.linear_model import OLS
import numpy as np

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
   
def calc_r2_press(model, groupby_cols=None, eps=1e-12):
    """
    Compute predicted R² (R2_PRESS) for an OLS model using either:
    1) the single-calculation PRESS shortcut (no grouping, no refits), or
    2) leave-one-group-out (LOGO) PRESS by refitting per group.

    Parameters:
        model : statsmodels.regression.linear_model.RegressionResultsWrapper
            A fitted OLS results object, typically from `sm.OLS(...).fit()` or `smf.ols(...).fit()`.
        groupby_cols : None | str | list[str] | tuple[str, ...], optional
            Columns in the original fitting DataFrame that define groups to leave out together.
            - If `None` (default), computes the standard PRESS via the OLS identity:
            `PRESS = sum((e / (1 - h))**2)`, where `e` are residuals and `h` are hat diagonals.
            - If provided, performs **leave-one-group-out**: for each unique group, drops all rows in
            that group, refits using the same formula, predicts the held-out rows, accumulates the
            squared errors as grouped PRESS.
        eps : float, optional (default=1e-12)
            Numerical safeguard to avoid division by (near) zero when leverage `h` is extremely close
            to 1 in the single-calculation path. Denominator `1 - h` is clipped to `±eps` as needed.

    Returns:
        r2_press : float
            Predicted R² computed as `1 - PRESS / SST`.
    """
        
    press_total = 0.0
    starting_df = model.model.data.frame.copy()
    starting_df = starting_df.dropna()
    formula = model.model.formula

    if groupby_cols is None:
        resid = model.resid
        h = model.get_influence().hat_matrix_diag
        denom = 1.0 - h
        denom = np.where(np.abs(denom) < eps, np.sign(denom) * eps, denom)
        r_loo = resid / denom
        press_total = float(np.sum(r_loo**2))
        return 1.0 - press_total / model.centered_tss
        
    else:
        for g_key, dfg in starting_df.groupby(groupby_cols, sort=False):
            test_idx = dfg.index
            train_df = starting_df.drop(index=test_idx)

            model_train = ols(formula=formula, data=train_df).fit()
            y_pred = model_train.predict(dfg)
            y_true = dfg[formula.split('~')[0].strip()].to_numpy()

            err = (y_true - y_pred)
            press_g = float(np.sum(err**2))
            press_total += press_g

    return 1 - press_total/model.centered_tss