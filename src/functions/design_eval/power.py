import numpy as np
from scipy.optimize import fsolve
import dexpy.power
import pandas as pd
from patsy import dmatrix
from typing import List

def est_signal_to_noise(design: pd.DataFrame,
                        formula: str,
                        term: str,
                        goal_power: float = 0.995,
                        alpha: float = 0.05,
                        initial_guess: float = 1.0,
                        ) -> float:
    """
    Within a specific design of experiments design and patsy model,
    given the power of a term in that model,
    estimate the signal-to-noise ratio of that term.

    Args:
        design (pd.DataFrame): Pandas df containing the design of experiments design
        model (str): patsy formula of the model
        term (str): which term in model to estimate the signal-to-noise ratio for
        goal_power (float): power for term from which the signal-to-noise ratio will be estimated for. Defaults to 0.995.
        alpha (float): significance level. Defaults to 0.05
        initial_guess (float): initial guess of signal-to-noise. Defaults to 1.

    Returns:
        float: returns the estimated signal-to-noise ratio
    """
    X = dmatrix(formula, design)
    X_formula = X.design_info.describe().replace(' ','')
    index_term = X_formula.split('+').index(term)

    def calc_power(snr):
        return dexpy.power.f_power(formula, design, snr, alpha)[index_term]

    def to_minimize(snr):
        return calc_power(snr) - goal_power

    pred_signal_to_noise = fsolve(to_minimize, initial_guess)

    if len(pred_signal_to_noise) == 1:
        return pred_signal_to_noise[0]
    else:
        raise ValueError('Inputs may be incorrect since multiple signal-to-noise values were returned')

def get_power(design: pd.DataFrame,
              model: str,
              signal_to_noise: float,
              alpha: float = 0.05,
              ) -> List[List[str]]:
    X = dmatrix(model, design)
    terms = X.design_info.describe().split(' + ')
    powers = dexpy.power.f_power(model, design, signal_to_noise, alpha)
    return [terms,powers]




