# python -m tests.functions.mult_lin_reg_utils.TestModelReduction
import unittest
import pandas as pd

import src.functions.mult_lin_reg_utils.model_reduction as model_reduction
from src.functions.helper_funcs import load_data_xlsx

data_xlsx = load_data_xlsx('tests//functions//mult_lin_reg_utils//Data.xlsx')

data = data_xlsx['data']
'''
design_parameters = data_xlsx['design_parameters']
response_parameters = data_xlsx['response_parameters']
features = data_xlsx['features']
levels = data_xlsx['levels']
term_types = data_xlsx['term_types']
model_orders = data_xlsx['model_orders']
responses = data_xlsx['responses']
lambdas = data_xlsx['lambdas']
rescalers = data_xlsx['rescalers']
'''

response = 'C56mAhg'

class TestTerms(unittest.TestCase):
    def test_forward_model_reduction(self):
        model = model_reduction.forward_model_reduction(data,
                                                        ['A', 'B', 'C', 'A:B', 'A:C', 'B:C', 'I(A**2)', 'I(C**2)'],
                                                        term_types = {'A':'Process','B':'Process','C':'Process'},
                                                        response = response,
                                                        key_stat = 'bic')
        terms,values = list(model.params.index),list(model.params.values)
        fit = dict(zip(terms,values))

        actual_terms = ['Intercept', 'A', 'B', 'C', 'A:B', 'I(A ** 2)', 'I(C ** 2)'] 
        actual_vals = [162.78786285370947, -0.31209458333335327, -0.30569033714607574, -0.7669499682690404, 0.2676925332441158, -0.6726353410194186, -0.5053828629166617]
        actual = dict(zip(actual_terms,actual_vals))

        self.assertCountEqual(terms,actual_terms)
    
        for fit_term, actual_term in zip (terms,actual_terms):
            self.assertAlmostEqual(fit[fit_term], actual[actual_term], places=5)

        # same as above but aicc
        model = model_reduction.forward_model_reduction(data,
                                                        ['A', 'B', 'C', 'A:B', 'A:C', 'B:C', 'I(A**2)', 'I(C**2)'],
                                                        term_types = {'A':'Process','B':'Process','C':'Process'},
                                                        response = response,
                                                        key_stat = 'aicc')
        terms,values = list(model.params.index),list(model.params.values)
        fit = dict(zip(terms,values))
        
        actual_terms = ['Intercept', 'A', 'B', 'C', 'I(A ** 2)'] 
        actual_vals = [162.49372299873937, -0.31209458333334084, -0.29081850000000237, -0.7383119429590046, -0.6726353410193298]
        actual = dict(zip(actual_terms,actual_vals))

        self.assertCountEqual(terms,actual_terms)
    
        for fit_term, actual_term in zip (terms,actual_terms):
            self.assertAlmostEqual(fit[fit_term], actual[actual_term], places=5)
    
    def test_backward_model_reduction(self):
        model = model_reduction.backward_model_reduction(data,
                                                        ['A', 'B', 'C', 'A:B', 'A:C', 'B:C', 'I(A**2)', 'I(C**2)'],
                                                        term_types = {'A':'Process','B':'Process','C':'Process'},
                                                        response = response,
                                                        key_stat = 'bic')
        terms,values = list(model.params.index),list(model.params.values)
        fit = dict(zip(terms,values))

        actual_terms = ['Intercept', 'A', 'B', 'C', 'A:B', 'I(A ** 2)', 'I(C ** 2)'] 
        actual_vals = [162.78786285370947, -0.31209458333335327, -0.30569033714607574, -0.7669499682690404, 0.2676925332441158, -0.6726353410194186, -0.5053828629166617]
        actual = dict(zip(actual_terms,actual_vals))

        self.assertCountEqual(terms,actual_terms)
    
        for fit_term, actual_term in zip (terms,actual_terms):
            self.assertAlmostEqual(fit[fit_term], actual[actual_term], places=5)

        # same as above but aicc
        model = model_reduction.backward_model_reduction(data,
                                                        ['A', 'B', 'C', 'A:B', 'A:C', 'B:C', 'I(A**2)', 'I(C**2)'],
                                                        term_types = {'A':'Process','B':'Process','C':'Process'},
                                                        response = response,
                                                        key_stat = 'aicc')
        terms,values = list(model.params.index),list(model.params.values)
        fit = dict(zip(terms,values))
        
        actual_terms = ['Intercept', 'A', 'B', 'C', 'I(A ** 2)'] 
        actual_vals = [162.49372299873937, -0.31209458333334084, -0.29081850000000237, -0.7383119429590046, -0.6726353410193298]
        actual = dict(zip(actual_terms,actual_vals))

        self.assertCountEqual(terms,actual_terms)
    
        for fit_term, actual_term in zip (terms,actual_terms):
            self.assertAlmostEqual(fit[fit_term], actual[actual_term], places=5)

    def test_model_reduction(self):
        model = model_reduction.model_reduction(data,
                                                ['A', 'B', 'C', 'A:B', 'A:C', 'B:C', 'I(A**2)', 'I(C**2)'],
                                                term_types = {'A':'Process','B':'Process','C':'Process'},
                                                response = response,
                                                key_stat = 'bic',
                                                direction='forwards')
        terms,values = list(model.params.index),list(model.params.values)
        fit = dict(zip(terms,values))

        actual_terms = ['Intercept', 'A', 'B', 'C', 'A:B', 'I(A ** 2)', 'I(C ** 2)'] 
        actual_vals = [162.78786285370947, -0.31209458333335327, -0.30569033714607574, -0.7669499682690404, 0.2676925332441158, -0.6726353410194186, -0.5053828629166617]
        actual = dict(zip(actual_terms,actual_vals))

        self.assertCountEqual(terms,actual_terms)
    
        for fit_term, actual_term in zip (terms,actual_terms):
            self.assertAlmostEqual(fit[fit_term], actual[actual_term], places=5)

        model = model_reduction.model_reduction(data,
                                                ['A', 'B', 'C', 'A:B', 'A:C', 'B:C', 'I(A**2)', 'I(C**2)'],
                                                term_types = {'A':'Process','B':'Process','C':'Process'},
                                                response = response,
                                                key_stat = 'aicc',
                                                direction='backwards')
        terms,values = list(model.params.index),list(model.params.values)
        fit = dict(zip(terms,values))
        
        actual_terms = ['Intercept', 'A', 'B', 'C', 'I(A ** 2)'] 
        actual_vals = [162.49372299873937, -0.31209458333334084, -0.29081850000000237, -0.7383119429590046, -0.6726353410193298]
        actual = dict(zip(actual_terms,actual_vals))

        self.assertCountEqual(terms,actual_terms)
    
        for fit_term, actual_term in zip (terms,actual_terms):
            self.assertAlmostEqual(fit[fit_term], actual[actual_term], places=5)

if __name__ == '__main__':
    unittest.main()

