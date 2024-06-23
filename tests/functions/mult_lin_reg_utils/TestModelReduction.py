# python -m tests.functions.mult_lin_reg_utils.TestModelReduction
import unittest
import pandas as pd

import src.functions.mult_lin_reg_utils.model_reduction as model_reduction

data = pd.read_excel('tests//functions//mult_lin_reg_utils//Data.xlsx')

class TestTerms(unittest.TestCase):
    def test_forward_model_reduction(self):
        self.assertEqual(list(model_reduction.forward_model_reduction(data,
                                                                 ['A', 'B', 'C', 'A:B', 'A:C', 'B:C', 'I(A**2)', 'I(C**2)'],
                                                                 term_types = {'A':'Process','B':'Process','C':'Process'},
                                                                 response = 'R1',
                                                                 key_stat = 'bic').params.index),
                         ['Intercept', 'C', 'B', 'I(A ** 2)', 'A', 'A:B', 'I(C ** 2)'])
    
    def test_forward_model_reduction(self):
        test_vals = list(model_reduction.forward_model_reduction(data,
                                                                ['A', 'B', 'C', 'A:B', 'A:C', 'B:C', 'I(A**2)', 'I(C**2)'],
                                                                term_types = {'A':'Process','B':'Process','C':'Process'},
                                                                response = 'R1',
                                                                key_stat = 'bic').params.values)
        actual_vals = [162.78786231737988, -0.7669503235619821, -0.3056904370850795, -0.6726350233835063, -0.3120946149626409, 0.2676924728186698, -0.5053828098375348]
        for test, actual in zip (test_vals,actual_vals):
            self.assertAlmostEqual(test, actual, places=5)
    
    def test_forward_model_reduction(self):
        self.assertEqual(list(model_reduction.forward_model_reduction(data,
                                                                 ['A', 'B', 'C', 'A:B', 'A:C', 'B:C', 'I(A**2)', 'I(C**2)'],
                                                                 term_types = {'A':'Process','B':'Process','C':'Process'},
                                                                 response = 'R1',
                                                                 key_stat = 'aicc').params.index),
                         ['Intercept', 'C', 'B', 'I(A ** 2)', 'A'])
    
    def test_forward_model_reduction(self):
        test_vals = list(model_reduction.forward_model_reduction(data,
                                                                ['A', 'B', 'C', 'A:B', 'A:C', 'B:C', 'I(A**2)', 'I(C**2)'],
                                                                term_types = {'A':'Process','B':'Process','C':'Process'},
                                                                response = 'R1',
                                                                key_stat = 'bic').params.values)
        actual_vals = [162.78786285370944, -0.7669499682690404, -0.3056903371460551, -0.67263502, -0.31209461]
        for test, actual in zip (test_vals,actual_vals):
            self.assertAlmostEqual(test, actual, places=5)

if __name__ == '__main__':
    unittest.main()

