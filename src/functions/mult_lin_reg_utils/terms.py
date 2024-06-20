from typing import List, Union, Any, Tuple
import numpy as np
from itertools import combinations, combinations_with_replacement
from collections import Counter

from .misc_func import temp_remove_this_item

'''

Functions related to the processing of terms in a patsy model, especially helper functions for hierarchy.py.

'''

def list_to_formula(terms: List) -> str:
    '''
    Takes in a list of terms and returns it as the right side (after the ~) of a patsy formula
    For example, if terms = [A, B, A:B], the function will return
    'A + B + A:B'

    Parameters:
        terms (List): list containing all terms in the patsy model
  
    Returns:
        str: string of the right side of the patsy formula
    '''
    formula = ''
    terms = list(dict.fromkeys(terms))
    for term in terms:
        formula += f'{str(term)}+'
    if formula[-1] == '+':
        formula = formula[:-1]
    return formula

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

def tuple_to_term(combo: Tuple) -> str:
    '''
    Similar to tuple_to_interaction in that it takes a tuple and combines them into an interaction term.
    This has the added function of combining similar base terms (i.e. A:A becomes A**2)
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
    for term, count in counts.items():
        if count > 1:
            terms.append(f'I({term}**{count})')
        else:
            terms.append(term)
    
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

def get_all_higher_order_terms(base_terms: List,max_order: int) -> List:
    all_terms = []
    for order in range(1,max_order+1):
        combos = combinations_with_replacement(base_terms,order)
        for combo in combos:
            all_terms.append(tuple_to_term(combo))
    return sort_terms(all_terms)
