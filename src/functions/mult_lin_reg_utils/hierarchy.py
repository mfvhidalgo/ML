from typing import List, Union, Any, Tuple
from itertools import combinations
import numpy as np

from .misc_func import temp_remove_this_item
from .formulas import list_to_formula

def get_base_order(term_str: str) -> List:
    '''
    Helper function. Expansion of get_base_exponent to include
    getting the base and order of interaction terms.
    Assumes that the base will be whichever letter is alphabetical.
    For example, 'C,I(A**2):B' will return I(A**2):B:C 

    '''
    if ':' in term_str:
        terms = term_str.split(':')
    else:
        terms = [term_str]
    
    bases,exponents = [],[]
    for term in terms:
        base,exponent = get_base_exponent(term)
        bases.append(base)
        exponents.append(exponent)
    bases.sort()
    return [bases[0],
            np.sum(exponents)]

def get_base_exponent(term_str: str) -> List:
    '''
    Takes in a term (str) such as C**3 and returns a list of [base,exponent] (such as [C,3]).
    Only works with powers like np.power(A,2) and A**2.
    '''
    base_term,exponent = term_str,1
    if 'np.power(' in term_str:
        comma_index = term_str.index(',')
        base_term = term_str[9:comma_index]
        exponent = float(term_str[comma_index+1:-1])

    if '**' in term_str:
        base_term,exponent = term_str[2:-1].split('**')
        exponent = float(exponent)
    
    if int(exponent) == exponent: # remove float exponents if exponent is basically an int. useful for later when removing duplicates
        exponent = int(exponent)

    return [base_term,exponent]

def get_lower_order_from_exponent(term_str: str,
                                  outpit_patsy: bool = True) -> List:
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
            if outpit_patsy:
                sub_term = f'I({base_term}**{exponent})'
            else:
                sub_term = f'{base_term}**{exponent}'
            if not(sub_term in model_terms):
                model_terms.append(sub_term)
            exponent -= 1
        if not(base_term in model_terms):
            model_terms.append(base_term)
        return model_terms

def list_to_interaction(combo: Tuple) -> str:
    '''
    A helper function to convert a list of terms into a sorted interaction term.
    Also removes interaction terms if they are interacting with a subset of themselves,
    such as, I(C**3):C, which is an artifact from using itertools.combinations
    with get_lower_order_from_exponent.
    
    Parameters
        combo (Tuple): element generated from an itertools.combinations object showing the terms which
                       are part of the interaction term
    
    Returns:
        str: interaction term in patsy format 
    '''
    bases = []
    for part_of_combo in combo:
        base,exponent = get_base_order(part_of_combo)
        bases.append(base)
    
    add_combo,add_combo_bases = [],[]
    for part_of_combo,base in zip(combo,bases):
        remaining_bases = temp_remove_this_item(bases,base)
        if not(base in remaining_bases):
            add_combo.append(part_of_combo)
            add_combo_bases.append(base)
    
    return ':'.join([ac for acb, ac in sorted(zip(add_combo_bases, add_combo))])

def get_lower_order_from_interaction(term_str: str) -> List:
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
            model_terms.extend(get_lower_order_from_exponent(base_term))
    base_terms = list(dict.fromkeys(model_terms))
    for order in range(2, max_order + 1):
        combos = combinations(base_terms, order)
        for combo in combos:
            add_term = list_to_interaction(combo)
            if add_term != '':
                model_terms.append(add_term)

    return list(dict.fromkeys(model_terms))

def sort_terms(terms: List) -> List:
    '''
    Helper function which takes in a list of terms, sorts them by order, then alphabetically.
    For example, ['A','C','I(A**2)','B'] will return ['A','B','C','I(A**2)']

    Parameters
        terms (List): list of terms

    Returns
        List: list of terms sorted by order then alphabetical order
    '''
    grouped_terms = {}
    for term in terms:
        base,exponent = get_base_order(term)
        if exponent not in grouped_terms:
            grouped_terms[exponent] = {'term':[],
                                        'base':[]
                                        }
        grouped_terms[exponent]['term'].append(term)
        grouped_terms[exponent]['base'].append(base)
    
    sorted_grouped_terms = {}
    for exponent,term_base in grouped_terms.items():
        sorted_grouped_terms[exponent] = [term for _, term in sorted(zip(term_base['base'], term_base['term']))]
    
    sorted_terms = []
    keys = list(sorted_grouped_terms.keys())
    keys.sort()
    for exponent in keys:
        for term in sorted_grouped_terms[exponent]:
            sorted_terms.append(term)
    
    return sorted_terms

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
            model_terms.extend(get_lower_order_from_exponent(term))
        elif ':' in term:
            model_terms.extend(get_lower_order_from_interaction(term))
        else:
            if not(term in model_terms):
                model_terms.append(term)
    model_terms = sort_terms(list(dict.fromkeys(model_terms)))
    return f'{response_str}~{list_to_formula(model_terms)}'



