from typing import List, Union, Any
from itertools import combinations

def get_lambdas(value: Union[str, float, int]) -> List[float]:
    '''
    Takes in a comma-separated-list separates them.
    Alternatively, takes in a single int/float and returns it
    '''

    if isinstance(value,(int,float)):
        return value
    
    elif isinstance(value,str):
        return [float(val) for val in value.split(',')]
    
    else:
        raise TypeError('Lambda value enters is not an str, int, or float')