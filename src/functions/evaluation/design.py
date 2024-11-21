import pandas as pd
import patsy
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict
from scipy.optimize import fsolve
import dexpy.power
import dexpy.alias
from patsy import dmatrix
from statsmodels.stats.outliers_influence import variance_inflation_factor

def evaluate_design(design: pd.DataFrame,
                    formula: str,
                    signal_to_noise: float = 2,
                    alpha: float = 0.05,
                    ) -> pd.DataFrame:
    """
    Wrapper which takes in an experimental design and a formula and checks for
    aliasing, power, and VIF.

    Uses default values for signal-to-noise and alpha, but these can be adjusted.

    Args:
        design (pd.DataFrame): Pandas dataframe of the actual experiments
        formula (str): patsy formula of the model, for example 'A + B + A:B'
        signal_to_noise (float, optional): signal-to-noise ratio. Defaults to 2.
        alpha (float, optional): significance level. Defaults to 0.05.

    Returns:
        pd.DataFrame: Pandas dataframe calculating the power and VIF for each term
                        in the formula. Any aliased terms are removed before running
                        the calculations.
    """
    # run this to get the aliased terms and what they are aliased with
    alias_list, _ = dexpy.alias.alias_list(formula, design)

    # reform formula after removing aliased terms
    alias_terms = []
    for alias_pair in alias_list:
        terms = alias_pair.replace(' ','').split('=')
        alias_terms.extend(terms)
    formula_list = formula.replace(' ','').split('+')
    unaliased_formula_list = [term for term in formula_list if not(term in alias_terms)]
    unaliased_formula = '+'.join(unaliased_formula_list)

    # calculate power and vif and combine to a single df    
    power = calculate_power(design=design,formula=unaliased_formula,signal_to_noise=signal_to_noise,alpha=alpha)
    power.set_index('Term', inplace=True)
    vif = calculate_VIF(design=design,formula=unaliased_formula)
    vif.set_index('Term', inplace=True)

    # add unaliased terms to the table for completeness
    unaliased_df = pd.concat([power,vif], axis=1)
    for term in alias_terms:
        unaliased_df.loc[term] = {'Power':'X','VIF':'X'}
    
    return unaliased_df
    
def calculate_covariance_matrix(design: pd.DataFrame,
                         formula: str,
                         ) -> pd.DataFrame:
    """
    Wrapper to calculate covariance and

    Args:
        design (pd.DataFrame): Pandas dataframe of the actual experiments
        formula (str): patsy formula of the model, for example 'A + B + A:B'

    Returns:
        pd.DataFrame: Pandas dataframe of the covariance matrix
    """

    X = patsy.dmatrix(formula, design, return_type='dataframe')

    covariance_matrix = np.cov(X, rowvar=False)
    covariance_df = pd.DataFrame(covariance_matrix,index=X.columns,columns=X.columns)

    return covariance_df

def calculate_correlation_matrix(design: pd.DataFrame,
                         formula: str,
                         ) -> pd.DataFrame:
    """
    Wrapper to calculate the covariance matrix

    Args:
        design (pd.DataFrame): Pandas dataframe of the actual experiments
        formula (str): patsy formula of the model, for example 'A + B + A:B'

    Returns:
        pd.DataFrame: Pandas dataframe of the correlation matrix
    """

    X = patsy.dmatrix(formula, design, return_type='dataframe')

    correlation_matrix = np.corrcoef(X, rowvar=False)
    correlation_df = pd.DataFrame(correlation_matrix,index=X.columns,columns=X.columns)

    return correlation_df

def plot_matrix_heatmap(df: pd.DataFrame,
                        cmap_colors: Dict = None) -> None:
    """
    Plot the heatmap with annotations

    Args:
        df (pd.DataFrame): Pandas dataframe of either the correlation or covariance matrix
    """

    if cmap_colors:
        colors = [(key, val) for key,val in cmap_colors.items()]
        cmap = LinearSegmentedColormap.from_list('', colors)
    else:
        cmap = 'coolwarm'

    sns.heatmap(df, annot=True, cmap=cmap)

def calculate_power(design: pd.DataFrame,
                    formula: str,
                    signal_to_noise: float = 2,
                    alpha: float = 0.05,
                    ) -> pd.DataFrame:
    """
    Calculates the power of every term in a model

    Args:
        design (pd.DataFrame): Pandas dataframe of the actual experiments
        formula (str): patsy formula of the model, for example 'A + B + A:B'
        signal_to_noise (float, optional): signal-to-noise ratio. Defaults to 2.
        alpha (float, optional): significance level. Defaults to 0.05.

    Returns:
        pd.DataFrame: Pandas dataframe where the first column contains the formula terms
                        and the second column contains the power of each term
    """

    X = patsy.dmatrix(formula, design, return_type='dataframe')
    terms = X.columns
    powers = dexpy.power.f_power(formula, design, signal_to_noise, alpha)
    return pd.DataFrame({'Term':terms,'Power':powers})

def calculate_VIF(design: pd.DataFrame,
                    formula: str
                    ) -> pd.DataFrame:
    """
    Calculates the variance inflation factor (VIF) of each term in the model.

    Args:
        design (pd.DataFrame): Pandas dataframe of the actual experiments

    Returns:
        pd.DataFrame: Pandas dataframe where the first column contains the formula terms
                        and the second column contains the VIF of each term
    """
    X = patsy.dmatrix(formula, design, return_type='dataframe')
    terms = X.columns
    vifs = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

    return pd.DataFrame({'Term':terms,'VIF':vifs})

def est_signal_to_noise(design: pd.DataFrame,
                        model: str,
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
    X = dmatrix(model, design)
    X_formula = X.design_info.describe().replace(' ','')
    index_term = X_formula.split('+').index(term)

    def calc_power(snr):
        return dexpy.power.f_power(model, design, snr, alpha)[index_term]

    def to_minimize(snr):
        return calc_power(snr) - goal_power

    pred_signal_to_noise = fsolve(to_minimize, initial_guess)

    if len(pred_signal_to_noise) == 1:
        return pred_signal_to_noise[0]
    else:
        raise ValueError('Inputs may be incorrect since multiple signal-to-noise values were returned')