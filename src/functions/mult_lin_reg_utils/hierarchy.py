from typing import List, Union, Any
from itertools import combinations
from .misc_func import temp_remove_this_item
from .formulas import list_to_formula

def get_base_exponent(term_str: str) -> List:
    '''
    Takes in a term (str) such as C**3 and returns a list of [base,exponent] (such as [C,3])
    '''
    base_term,exponent = term_str,0
    if 'np.power(' in term_str:
        comma_index = term_str.index(',')
        base_term = term_str[9:comma_index]
        exponent = float(term_str[comma_index+1:-1])

    if '**' in term_str:
        base_term,exponent = term_str.split('**')
        exponent = float(exponent)
    
    return [base_term,exponent]

def get_lower_order_from_exponent(term_str: str,
                                  input_patsy: bool = False,
                                  outpit_patsy: bool = True) -> List:
    '''
    Takes in a str within I() from patsy and returns itself plus lower order terms
    For example, if term_str is C**3, it will return a list of [I(C**3),I(C**2),C].

    Parameters:
        term_str (str): string containing a term in a patsy model
        input_patsy (bool): if True, term_str is surrounded by I().
        output_patsy (bool): if True, will surround each returned term (that have exponents) with 'I()'.
                             if False, will return terms as is.

    Returns:
        List: list of terms containing the starting term and any lower order terms
    '''
    model_terms = []
    hasPower = False
    if input_patsy:
        term_str = term_str[2:-1]
    base_term,exponent = get_base_exponent(term_str)

    if exponent == 0:
        if outpit_patsy:
            return [f'I({term_str})'] # in case there are non-exponent items like log under I()
        else:
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

def get_lower_order_from_interaction(term_str):
    '''
    Takes a term (str) with interactions (separated with ':') and gets all the permutations of
    lower-order terms

    Parameters:
        term_str (str): term in the patsy model
  
    Returns:
        List: list of all the terms, including lower-order terms
    '''
    base_terms = term_str.split(':')
    model_terms = base_terms.copy()
    
    for base_term in base_terms:
        if 'I(' == base_term[0:2]:
            model_terms.extend(get_lower_order_from_exponent(base_term[2:-1]))
        else:
            if not(base_term in model_terms):
                model_terms.append(base_term)
    base_terms.extend(model_terms)
    base_terms = list(dict.fromkeys(base_terms))
    for order in range(2, len(base_terms) + 1):
        combos = combinations(base_terms, order)
        for combo in combos:
            bases = []
            for part_of_combo in combo:
                if part_of_combo[0:2] =='I(':
                    base,exponent = get_base_exponent(part_of_combo[2:-1])
                else:
                    base,exponent = get_base_exponent(part_of_combo)
                bases.append(base)
            
            add_combo = []
            for part_of_combo,base in zip(combo,bases):
                remaining_bases = temp_remove_this_item(bases,base)
                if not(base in remaining_bases):
                    add_combo.append(part_of_combo)
            add_terms = ':'.join(add_combo)
            if add_terms != '':
                model_terms.append(add_terms)

    return list(dict.fromkeys(model_terms))

def get_all_lower_order_terms(formula: str) -> List:
    '''
    Takes in a str of patsy formula and returns a list of individual terms,
    including any lower-order terms needed to be added due to heirarchy rules.

    For example, if formula = 'y ~ A + B + I(C**3)+ E:D', the function will return
    [A, B, I(C**3), I(C**2), C, E, D]

    Parameters:
        formula (str): string containing a patsy formula (including the 'response ~').
                       Can contain individual terms,
                       exponents in the form of I(np.power(term,exponent)) or I(term**exponent),
                       or interaction terms like term1:term2
  
    Returns:
        dict: dict where dict['terms'] is the list of terms in the model
              including any lower order terms
              and dict['formula'] is the new formula
    '''
    response_str,terms_str = formula.split('~')
    terms = terms_str.replace(' ','').split('+')

    model_terms = []
    for term in terms:
        if ('I(' == term[0:2]) and not(':' in term):
            model_terms.extend(get_lower_order_from_exponent(term[2:-1]))
        elif ':' in term:
            model_terms.extend(get_lower_order_from_interaction(term))
        else:
            if not(term in model_terms):
                model_terms.append(term)
    
    model_terms = list(dict.fromkeys(model_terms))

    return {'terms':model_terms,
            'formula':f'{response_str}~{list_to_formula(model_terms)}'}


