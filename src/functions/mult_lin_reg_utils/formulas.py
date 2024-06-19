from typing import List, Union, Any

def list_to_formula(terms: List) -> str:
    '''
    Takes in a list of terms and returns it as the right side (after the ~) of a patsy formula
    For example, if terms = [A, B, A:B], the function will return
    'A + B + C'

    Parameters:
        terms (List): list containing all terms in the patsy model
  
    Returns:
        str: string of the right side of the patsy formula
    '''
    formula = ''
    terms = list(dict.fromkeys(terms))
    for term in terms:
        formula += f'{str(term)}+'
    return formula[:-1]