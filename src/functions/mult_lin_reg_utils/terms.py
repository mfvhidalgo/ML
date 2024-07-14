from typing import List, Union, Any, Tuple, Dict
import numpy as np
from itertools import combinations, combinations_with_replacement
from collections import Counter

from .misc_func import temp_remove_this_item

'''

Functions related to the processing of terms in a patsy model, especially helper functions for hierarchy.py.

'''

def list_to_formula(terms: List,
                    term_types: Dict = {},
                    response: str = 'response') -> str:
    '''
    Takes in a list of terms and returns it as the right side (after the ~) of a patsy formula
    For example, if terms = [A, B, A:B], the function will return
    'A + B + A:B'

    Parameters:
        terms (List): list containing all terms in the patsy model
        term_types (Dict): dict where the keys are terms
                           and values show whether that term is a Mixture or Process term.
                           By default is blank so assumes all terms are Process terms.
        response (str): the left part of the ~
  
    Returns:
        str: string of the right side of the patsy formula
    '''

    formula = ''
    terms = list(dict.fromkeys(terms))
    for term in terms:
        formula += f'{str(term)}+'
    if formula == '':
        return f"{response}~1"
    else:
        if formula[-1] == '+':
            formula = formula[:-1]
        if 'Mixture' in term_types.values():
            formula = f"{response}~{formula}-1"
        return f"{response}~{formula}"

def get_base_order(term_str: str) -> List:
    '''
    Helper function. Expansion of get_base_exponent to include
    getting the base and order of interaction terms.
    Assumes that the base will be whichever letter is alphabetical
    among the lowest ordered terms.
    For example 'B:I(A**2):I(B**3)' returns ['B',6]

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
    
    sorted_zipped = sorted(zip(exponents,bases),reverse=True)
    exponents,bases = zip(*sorted_zipped)
    bases_min_exponent = list(bases[exponents.index(min(exponents)):])
    bases_min_exponent.sort()
    return [bases_min_exponent[0],
            np.sum(exponents)]

def get_base_exponent(term_str: str) -> List:
    '''
    Takes in a term (str) such as C**3 and returns a list of [base,exponent] (such as [C,3]).
    Only works with powers like np.power(A,2) and A**2.
    '''
    if ':' in term_str:
        raise ValueError('term contains interaction and not just an exponent')
    else:
        base_term,exponent = term_str,1
        if 'np.power(' in term_str:
            comma_index = term_str.index(',')
            base_term = term_str[9:comma_index]
            exponent = float(term_str[comma_index+1:-1])

        if '**' in term_str:
            base_term,exponent = term_str[2:-1].split('**')
            exponent = float(exponent)
        
        if '[T.' in term_str:
            base_term = term_str.split('[T.')[0]
            base_term = 1
        
        if int(exponent) == exponent: # remove float exponents if exponent is basically an int. useful for later when removing duplicates
            exponent = int(exponent)

        return [base_term,exponent]

def tuple_to_interaction(combo: Tuple) -> str:
    '''
    A helper function to convert a list of terms into a sorted interaction term.
    Also removes interaction terms if they are interacting with a subset of themselves,
    such as, I(C**3):C, which is an artifact from using itertools.combinations.
    
    Parameters
        combo (Tuple): element generated from an itertools.combinations object showing the terms which
                       are part of the interaction term
    
    Returns:
        str: interaction term in patsy format 
    '''
    bases = [get_base_order(part_of_combo)[0] for part_of_combo in combo]
    if len(bases) == len(set(bases)):
        return ':'.join([term for _, term in sorted(zip(bases, combo))])
    else:
        return ''
    '''
    add_combo,add_combo_bases = [],[]
    for part_of_combo,base in zip(combo,bases):
        remaining_bases = temp_remove_this_item(bases,base)
        if not(base in remaining_bases):
            add_combo.append(part_of_combo)
            add_combo_bases.append(base)
    
    return ':'.join([ac for _, ac in sorted(zip(add_combo_bases, add_combo))])
    '''

def tuple_to_term(combo: Tuple,
                  return_order: bool = False) -> Union[str,List]:
    '''
    Similar to tuple_to_interaction in that it takes a tuple and combines them into an interaction term.
    This has the added function of combining similar base terms (i.e. A:A becomes A**2)
    wheras tuple_to_interaction ignores these.
    '''
    all_terms = []
    for term in combo:
        add_terms = term.split(':') if ':' in term else [term]
        for add_term in add_terms:
            base,exponent = get_base_exponent(add_term)
            for exp in range(1,exponent+1):
                all_terms.append(base)

    counts = Counter(all_terms)

    terms = []
    order = 0
    for term, count in counts.items():
        if count > 1:
            terms.append(f'I({term}**{count})')
            order += count
        else:
            terms.append(term)
            order += 1
    if return_order:
        return [":".join(sort_terms(terms)),order]
    else:
        return ":".join(sort_terms(terms))

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
                                        'base':[],
                                        'base_order':[]
                                        }
        adder = '0' if ':' in term else '1'
        grouped_terms[exponent]['base_order'].append(f'{adder}{base}')
        grouped_terms[exponent]['term'].append(term)
        grouped_terms[exponent]['base'].append(base)
    
    sorted_grouped_terms = {}
    for exponent,term_base in grouped_terms.items():
        sorted_grouped_terms[exponent] = [term for _, term in sorted(zip(term_base['base_order'], term_base['term']))]
    
    sorted_terms = []
    keys = list(sorted_grouped_terms.keys())
    keys.sort()
    for exponent in keys:
        for term in sorted_grouped_terms[exponent]:
            sorted_terms.append(term)
    
    return sorted_terms

def get_all_higher_order_terms(base_terms: List,
                               max_order: int,
                               max_exponent: int = 'max_order') -> List:
    if max_exponent == 'max_order':
        max_exponent = max_order
    
    all_terms = []
    for order in range(1,max_order+1):
        combos = combinations_with_replacement(base_terms,order)
        for combo in combos:
            term = tuple_to_term(combo)
            if ':' in term:
                all_terms.append(term)
            else:
                base,exponent = get_base_exponent(term)
                if exponent <= max_exponent:
                    all_terms.append(term)

    return sort_terms(all_terms)

def list_to_orders(terms_list: List) -> Dict:
    '''
    Helper function, converts a list of terms into a dict
    where each key is the order and the values are lists of terms of that order.
    For example, ['A','B','C','I(C**2)','A:B','A:I(C**2)'] will return
    {1:['A','B','C'], 2:['I(C**2)','A:B',], 3:['A:I(C**2)']})

    Parameters:
        terms_list (List): list of terms

    Returns:
        Dict: dictionary where the keys are the orders
        and the values are lists of terms of that value
    '''

    terms = {}
    for term in terms_list:
        base,order = get_base_order(term)
        if not(order in terms):
            terms[order] = []
        terms[order].append(term)
    
    return terms

def patsy_to_list(formula:str):
    if '~' in formula:
        right_hand_side = formula[formula.index('~')+1:]
    else:
        right_hand_side = formula
    if '-' in right_hand_side:
        right_hand_side = right_hand_side.split('-')[0]
        terms = right_hand_side.split('+')
        terms.append('-1')
        return terms
    return right_hand_side.split('+')