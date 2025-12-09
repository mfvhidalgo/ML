from typing import List, Union, Any, Tuple
from itertools import combinations
import numpy as np


from .terms import list_to_formula,get_base_order,get_base_exponent,tuple_to_interaction, sort_terms

'''

Functions related to adding missing terms due to hierarchy.
For example, a model cannot be y = x**2 + constant. It should be y = x**2 + x + constant

'''

def get_all_combo_from_exponent(term_str: str,
                                  output_patsy: bool = True) -> List:
    '''
    Takes in a str from patsy (such as I(A**2)) and returns itself plus lower order terms
    For example, if term_str is C**3, it will return a list of [I(C**3),I(C**2),C].

    Parameters:
        term_str (str): string containing a term in a patsy model
        output_patsy (bool): if True, will surround each returned term (that have exponents) with 'I()'.
                             if False, will return terms as is.

    Returns:
        List: list of terms containing the starting term and any lower order terms
    '''
    model_terms = []
    hasPower = False
    base_term,exponent = get_base_order(term_str)

    if exponent == 1:
        return [f'{term_str}']
    else:
        if int(exponent) == exponent: # remove float exponents if exponent is basically an int. useful for later when removing duplicates
            exponent = int(exponent)
        model_terms.append(f'I({base_term}**{exponent})')
        exponent -= 1
        while exponent > 1:
            if output_patsy:
                sub_term = f'I({base_term}**{exponent})'
            else:
                sub_term = f'{base_term}**{exponent}'
            if not(sub_term in model_terms):
                model_terms.append(sub_term)
            exponent -= 1
        if not(base_term in model_terms):
            model_terms.append(base_term)
        return model_terms

def get_all_combo_from_interaction(term_str: str) -> List:
    '''
    Takes a term (str) with interactions (separated with ':') and gets all the permutations of
    lower-order terms

    Parameters:
        term_str (str): term in the patsy model
  
    Returns:
        List: list of all the terms, including lower-order terms
    '''
    base_terms = term_str.split(':')
    max_order = len(base_terms)
    model_terms = base_terms.copy()
    
    for base_term in base_terms:
        base,exponent = get_base_order(base_term)
        if exponent > 1:
            model_terms.extend(get_all_combo_from_exponent(base_term))
    base_terms = list(dict.fromkeys(model_terms))
    for order in range(2, max_order + 1):
        combos = combinations(base_terms, order)
        for combo in combos:
            add_term = tuple_to_interaction(combo)
            if add_term != '':
                model_terms.append(add_term)

    return list(dict.fromkeys(model_terms))

def get_all_lower_order_terms(formula: str) -> str:
    '''
    Takes in a str of patsy formula and returns a list of individual terms,
    including any lower-order terms needed to be added due to heirarchy rules.

    For example, if formula = 'y ~ A + B + I(C**3)+ E:D', the function will return
    'y~A+B+C+D+E+I(C**2)+D:E+I(C**3)'

    Parameters:
        formula (str): string containing a patsy formula (including the 'response ~').
                       Can contain individual terms,
                       exponents in the form of I(np.power(term,exponent)) or I(term**exponent),
                       or interaction terms like term1:term2
  
    Returns:
        str: str of the patsy formula containing all terms,
        including their respective lower-order terms
    '''
    formula = formula.replace(' ','')
    response_str,terms_str = formula.split('~')
    terms = terms_str.split('+')

    model_terms = []
    for term in terms:
        if ('**' in term) and not(':' in term):
            model_terms.extend(get_all_combo_from_exponent(term))
        elif ':' in term:
            model_terms.extend(get_all_combo_from_interaction(term))
        else:
            if not(term in model_terms):
                model_terms.append(term)
    model_terms = sort_terms(list(dict.fromkeys(model_terms)))
    return list_to_formula(model_terms,response=response_str)



