import pandas as pd
import numpy as np
from typing import List
import itertools

from mult_lin_reg_utils.terms import sort_terms, list_to_formula
from mult_lin_reg_utils.hierarchy import get_all_lower_order_terms


df = pd.DataFrame({'A':[-1,-1,1,1],'B':[-1,1,-1,1]})

class random_model:
    def __init__(self,
                 n_features: int,
                 interactions: List[int] = None, # set as [1] to not have any interactions
                 polynomials: List[int] = None, # set as [1] to not have any higher polynomials
                 size: float = 0.5,
                 seed: int = np.random.randint(10000000)):
        self.n_features = n_features
        self.size = size

        if not(interactions):
            self.interactions = list(range(1,n_features))
        else:
            if max(interactions) > n_features:
                raise ValueError('Interaction level cannot be more than the number of features.')
            self.interactions = list(interactions)
        if not(polynomials):
            self.polynomials = list(range(1,n_features))
        else:
            self.polynomials = list(polynomials)
        
        self.seed = seed
        np.random.seed(self.seed)

        self.features = [chr(65+i) for i in range(n_features)]
        self.terms = self.features.copy()
    
        self.polynomials.remove(1)
        for interaction in self.interactions:
            for combo in itertools.combinations_with_replacement(self.features, interaction):
                 self.terms.append(':'.join(sorted(set(combo))))
        
        self.interactions.remove(1)
        for polynomial in self.polynomials:
            for feature in self.features:
                self.terms.append(f'I({feature}**{polynomial})')

        self.terms = sort_terms(list(set(self.terms)))

        self.full_model = list_to_formula(self.terms)

        self.simple_terms = sort_terms(np.random.choice(self.terms, int(np.ceil(len(self.terms)*self.size))))
        self.simple_model = get_all_lower_order_terms(list_to_formula(self.simple_terms))

        self.__orig_terms__,self.__orig_full_model__ = self.terms.copy(), self.full_model[:]
        self.__orig_simple_terms__,self.__orig_simple_model__ = self.simple_terms.copy(), self.simple_model[:]

        self.terms, self.full_model = self.__format_formulas_terms__(self.full_model)
        self.simple_terms, self.simple_model = self.__format_formulas_terms__(self.simple_model)

    def __format_formulas_terms__(self, orig_model):
        terms = orig_model.split('+')
        terms[0] = terms[0].split('~')[1]
        terms = [term.replace(':','*') for term in terms]
        terms = [term.replace('I(','')[:-1] if 'I(' in term else term for term in terms]
        
        terms = [f'{int(np.random.uniform(-50, 50))}*{term}' for term in terms]

        model = terms[0]

        for term in terms[1:]:
            if term[0] == '-':
                model = model + term
            else:
                model = f'{model}+{term}'
        return terms, model



