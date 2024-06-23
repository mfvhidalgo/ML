from typing import List, Dict
import pandas as pd
from statsmodels.formula.api import ols

from .terms import list_to_orders, list_to_formula
from .statistics import calc_bic_aicc

def model_reduction(data: pd.DataFrame,
                    model: ols,
                    terms_list: List,
                    term_types: Dict,
                    key_stat: str,
                    direction: str) -> str:
    '''
    Takes in a list of terms and simplifies the model.

    Parameters:
        data (pd.DataFrame): pandas DataFrame containing the data.
        terms (List): list of terms (ideally coded) for the maximum / most complex model.
        term_types (dict): dict where the keys are the terms and the values are either 'Process' or 'Mixture'.
                           defines whether a term is a process or mixture term.
        key_stat (str): either 'aicc' or 'bic'. defines the statistic used to evaluate each model.
                        aicc stands for the corrected Akeike Information Criterion
                        and bic stands for the Baeysian Information Criterion
        direction (str): 'forwards' for forward term addition
                         or 'backwards' for backwards term elimination

    Returns:   
        str: the final reduced model.

    '''

    if key_stat != 'aicc' and key_stat != 'bic':
        raise ValueError('key_stat must be either aicc or bic')

    if direction == 'forward':
        forward_model_reduction()
    elif direction == 'backward':
        backward_model_reduction()
    else:
        raise ValueError('direction must be either forward or backward')
    
def forward_model_reduction(data: pd.DataFrame,
                            terms_list: List,
                            term_types: Dict,
                            response: str,
                            key_stat: str) -> str:
    df = data.copy()
    terms_by_order = list_to_orders(terms_list)

    cur_model_terms, remaining_terms = [], []

    for term in terms_by_order[1]:
        if term_types[term] == 'Mixture':
             cur_model_terms.append(term)

    model = ols(formula=list_to_formula(cur_model_terms,
                                        term_types = term_types,
                                        response=response),data=df).fit()
    best_key_stat = calc_bic_aicc(model,key_stat)

    for order,terms in terms_by_order.items():
        terms_to_try = list(set([term for term in terms + remaining_terms if not(term in cur_model_terms)]))

        while len(terms_to_try) > 0:
            changed_best_ter = False
            for try_term in terms_to_try:
                model = ols(formula=list_to_formula(cur_model_terms + [try_term],
                                                    term_types = term_types,
                                                    response=response),
                            data=df).fit()
                
                key_stat_value = calc_bic_aicc(model,key_stat)

                if key_stat_value < best_key_stat:
                    best_term,best_key_stat = try_term,key_stat_value
                    changed_best_ter = True
            if changed_best_ter:
                cur_model_terms.append(best_term)
                terms_to_try.remove(best_term) 
            else:
                remaining_terms = terms_to_try.copy()
                terms_to_try = []
    
    return ols(formula=list_to_formula(cur_model_terms,
                                       term_types = term_types,
                                       response=response),data=df).fit()
    
    

            

def backward_model_reduction(data: pd.DataFrame,
                            terms_list: str,
                            term_types: Dict,
                            key_stat: str) -> str:
    ...