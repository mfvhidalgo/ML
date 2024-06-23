from typing import List, Dict
import pandas as pd
from statsmodels.formula.api import ols

from .terms import list_to_orders, list_to_formula
from .statistics import calc_bic_aicc
from .hierarchy import get_all_lower_order_terms

def model_reduction(data: pd.DataFrame,
                    terms_list: List,
                    term_types: Dict,
                    response: str,
                    key_stat: str = 'aicc',
                    direction: str = 'forwards') -> ols:
    '''
    Simplifies the model via model reduction.

    Parameters:
        data (pd.DataFrame): df containing all the features, responses, and their respective values
        terms_list (List): list containing all the terms of the largest model.
        term_types (Dict): dict where the keys are the lowest-order terms and the values are either
                           'Mixture' or 'Process'
        response (str): the column name in data of the response to be modeled.
        key_stat (str): statistic used to determine if a term will be added or not. can be either
                        'aicc' for the corrected Akeike Information Criterion
                        or 'bic' for the Bayesian Information Criterion
        direction (str): direction of model reduction, either 'forwards' or 'backwards'
    
    Returns:   
        ols: the final reduced and fit model
    '''

    if key_stat != 'aicc' and key_stat != 'bic':
        raise ValueError('key_stat must be either aicc or bic')

    if direction == 'forwards':
        return forward_model_reduction(data,terms_list,term_types,response,key_stat)
    elif direction == 'backwards':
        return backward_model_reduction(data,terms_list,term_types,response,key_stat)
    else:
        raise ValueError('direction must be either forwards or backwards')
    
def forward_model_reduction(data: pd.DataFrame,
                            terms_list: List,
                            term_types: Dict,
                            response: str,
                            key_stat: str) -> ols:
    '''
    Applies forward model reduction. Mainly a helper function for model_reduction.

    Parameters:
        data (pd.DataFrame): df containing all the features, responses, and their respective values
        terms_list (List): list containing all the terms of the largest model.
        term_types (Dict): dict where the keys are the lowest-order terms and the values are either
                           'Mixture' or 'Process'
        response (str): the column name in data of the response to be modeled.
        key_stat (str): statistic used to determine if a term will be added or not. can be either
                        'aicc' for the corrected Akeike Information Criterion
                        or 'bic' for the Bayesian Information Criterion
    
    Returns:   
        ols: the final reduced and fit model
    '''
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
    
    formula = get_all_lower_order_terms(list_to_formula(cur_model_terms,
                                        term_types = term_types,
                                        response=response))

    return ols(formula=formula,data=df).fit()

def backward_model_reduction(data: pd.DataFrame,
                             terms_list: List,
                             term_types: Dict,
                             response: str,
                             key_stat: str) -> ols:
    
    '''
    Applies backwards model reduction. Mainly a helper function for model_reduction.

    Parameters:
        data (pd.DataFrame): df containing all the features, responses, and their respective values
        terms_list (List): list containing all the terms of the largest model.
        term_types (Dict): dict where the keys are the lowest-order terms and the values are either
                           'Mixture' or 'Process'
        response (str): the column name in data of the response to be modeled.
        key_stat (str): statistic used to determine if a term will be added or not. can be either
                        'aicc' for the corrected Akeike Information Criterion
                        or 'bic' for the Bayesian Information Criterion
    
    Returns:   
        ols: the final reduced and fit model
    '''
    
    df = data.copy()
    terms_by_order = list_to_orders(terms_list)
    cur_model_terms = terms_list.copy()

    model = ols(formula=list_to_formula(cur_model_terms,
                                        term_types = term_types,
                                        response=response),data=df).fit()
    best_key_stat = calc_bic_aicc(model,key_stat)

    remove_terms = []

    orders = list(terms_by_order.keys())
    orders.sort(reverse=True)

    for order in orders:
        terms = terms_by_order[order]
        remove_terms.extend(terms)
        changed_best_ter = True
        while changed_best_ter:
            changed_best_ter = False
            for remove_term in remove_terms:
                last_model_terms = cur_model_terms.copy()
                last_model_terms.remove(remove_term)

                removed_model = ols(formula=list_to_formula(last_model_terms,
                                        term_types = term_types,
                                        response=response),
                                    data=df).fit()
                
                removed_key_stat_value = calc_bic_aicc(removed_model,key_stat)

                if removed_key_stat_value < best_key_stat:
                    best_term_remove = remove_term
                    best_key_stat = removed_key_stat_value
                    changed_best_ter = True
                
            if changed_best_ter:
                remove_terms.remove(best_term_remove)
                cur_model_terms.remove(best_term_remove)

    formula = get_all_lower_order_terms(list_to_formula(cur_model_terms,
                                        term_types = term_types,
                                        response=response))

    return ols(formula=formula,data=df).fit()