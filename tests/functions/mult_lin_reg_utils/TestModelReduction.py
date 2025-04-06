# python -m tests.functions.mult_lin_reg_utils.TestModelReduction
import unittest
import pandas as pd
import os

import src.ml_utils.mult_lin_reg_utils.model_reduction as model_reduction
from src.ml_utils.helper_funcs import load_data_xlsx

data_xlsx = load_data_xlsx('tests//functions//mult_lin_reg_utils//Data.xlsx')

data = data_xlsx['data']

design_parameters = data_xlsx['design_parameters']
response_parameters = data_xlsx['response_parameters']
features = data_xlsx['features']
levels = data_xlsx['levels']
term_types = data_xlsx['term_types']
responses = data_xlsx['responses']
lambdas = data_xlsx['lambdas']
rescalers = data_xlsx['rescalers']

response = 'C56mAhg'

terms_list = ['A','B','C','A:B','I(A**2)','I(C**2)']

data_xlsx_cat = load_data_xlsx('tests//functions//mult_lin_reg_utils//Datacat.xlsx')
data_cat = data_xlsx_cat['data']
design_parameters_cat = data_xlsx_cat['design_parameters']
response_parameters_cat = data_xlsx_cat['response_parameters']
features_cat = data_xlsx_cat['features']
levels_cat = data_xlsx_cat['levels']
term_types_cat = data_xlsx_cat['term_types']
responses_cat = data_xlsx_cat['responses']
lambdas_cat = data_xlsx_cat['lambdas']
rescalers_cat = data_xlsx_cat['rescalers']

terms_list_cat = ['A', 'B', 'C', 'A:B', 'A:C', 'B:C', 'I(A**2)']

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

    def test_auto_model_reduction(self):
        reduced_models = model_reduction.auto_model_reduction(data,
                                                              ['A', 'B', 'C', 'A:B', 'A:C', 'B:C', 'I(A**2)', 'I(C**2)'],
                                                              term_types = {'A':'Process','B':'Process','C':'Process'},
                                                              response = response,
                                                              key_stat = 'aicc_bic',
                                                              direction ='forwards_backwards',
                                                              lambdas = lambdas[response])
        
        best_model = reduced_models['best_models'][1]
        self.assertAlmostEqual(best_model['r2adj'].values[0], 0.749454, places=5)
        self.assertAlmostEqual(best_model['r2press'].values[0], 0.585008, places=5)
        self.assertEqual(best_model['key_stat'].values[0], 'bic')
        self.assertEqual(best_model['direction'].values[0], 'forwards')
        self.assertAlmostEqual(best_model['Intercept'].values[0], 162.787862,places=5)
        self.assertAlmostEqual(best_model['I(C ** 2)'].values[0], -0.505383,places=5)
        self.assertCountEqual(list(best_model.columns),['response','lambda','r2adj', 'r2press', 'd_r2s',
                                                        'key_stat', 'direction',
                                                        'num_terms','formulas', 'Intercept',
                                                        'A', 'B', 'C', 'I(A ** 2)',
                                                        'A:B', 'I(C ** 2)'])
        
        # categorical
        reduced_models = model_reduction.auto_model_reduction(data_cat,
                                                        terms_list_cat,
                                                        term_types = {'A':'Process','B':'Process','C':'Process'},
                                                        response = response,
                                                        key_stat = 'bic',
                                                        direction ='forwards_backwards',
                                                        lambdas = lambdas[response])
        best_model = reduced_models['best_models'][1]
        self.assertAlmostEqual(best_model.loc[0,'r2adj'],0.7255190104805536,places=5)
        self.assertAlmostEqual(best_model.loc[0,'r2press'],0.510058,places=5)
        self.assertAlmostEqual(best_model.loc[0,'A:B[T.low]'],-0.675102,places=5)
        self.assertCountEqual(best_model.columns,
                              ['response', 'lambda', 'r2adj', 'r2press', 'd_r2s', 'key_stat',
                                'direction', 'num_terms', 'formulas', 'Intercept', 'B[T.low]',
                                'C[T.low]', 'C[T.mid]', 'A', 'A:B[T.low]', 'I(A ** 2)'])
        
    def test_get_best_model(self):
        df = pd.DataFrame({'r2adj':[0.9,0.7,0.99],'r2press':[0.1,0.2,-245],
                    'd_r2s':[0.8,0.5,244.7],
                    'key_stat':['aicc','aicc','bic'],
                    'direction':['forwards','backwards','forwards'],
                    'num_terms':[5,7,1]})
        
        best_model = model_reduction.get_best_model(df)
        self.assertEqual(best_model['r2adj'].values[0], 0.7)
        self.assertEqual(best_model['r2press'].values[0], 0.2)
    
    def test_encoded_models_to_real(self):
        reduced_models = model_reduction.auto_model_reduction(data,
                                                        ['A', 'B', 'C', 'A:B', 'A:C', 'B:C', 'I(A**2)', 'I(C**2)'],
                                                        term_types = {'A':'Process','B':'Process','C':'Process'},
                                                        response = response,
                                                        key_stat = 'aicc_bic',
                                                        direction ='forwards_backwards',
                                                        lambdas = lambdas[response])
        
        lmbda = 1
        df_best_models = reduced_models['best_models'][lmbda]
        
        features_reversed = {}
        for key, value in features.items():
            features_reversed[value] = key

        real_data = data[list(features.values()) + list(responses)].copy()
        real_data.columns = [features_reversed[val] for val in list(features.values())] + list(responses)
        non_term_columns = ['response','lambda','r2adj','r2press','d_r2s','key_stat','direction','num_terms']
        
        df_best_models_real = model_reduction.encoded_models_to_real(df_best_models,
                                                                    term_types,
                                                                    response,
                                                                    real_data,
                                                                    non_term_columns)
        
        df_best_models_real = df_best_models_real.loc[df_best_models_real['key_stat']=='bic']
        
        self.assertAlmostEqual(df_best_models_real['A'].values[0], 0.0562,places=4)
        self.assertAlmostEqual(df_best_models_real['A:B'].values[0], 0.035692,places=4)
        self.assertAlmostEqual(df_best_models_real['I(A ** 2)'].values[0], -0.000747,places=6)
        self.assertAlmostEqual(df_best_models_real['Intercept'].values[0], 157.933645,places=2)
        self.assertAlmostEqual(df_best_models_real['r2adj'].values[0],
                               reduced_models['models'][lmbda]['bic']['forwards'].rsquared_adj,
                               places=3)

if __name__ == '__main__':
    unittest.main()

