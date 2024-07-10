# python -m tests.functions.mult_lin_reg_utils.TestPowerTransform
import unittest
import pandas as pd
import statsmodels.formula.api as smf

import src.functions.mult_lin_reg_utils.power_transform as power_transform
from src.functions.helper_funcs import load_data_xlsx

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

formula = f"{response}~A+B+C+A:B+I(A**2)+I(C**2)"

class TestPowerTransform(unittest.TestCase):
    def test_best_boxcox_lambda(self):
        lambda_values = power_transform.best_boxcox_lambda(data,
                                                           formula,
                                                           response)
        
        actual_lambda_values = {'lambdas in conf int': [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2],
                                'lambdas': [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2],
                                'ln resid sum squares': [0.8012016257306281, 0.8009991885563085, 0.8008286635785253, 0.8006900194012966, 0.8005832232875573, 0.8005082408540014, 0.8004650361019257, 0.8004535714267857, 0.8004738076284871],
                                'confidence interval': 1.149635389608604,
                                'best lambda': 1.5}

        for key in ['lambdas in conf int','lambdas']:
            self.assertCountEqual(lambda_values[key],actual_lambda_values[key])

        for key in ['confidence interval','best lambda']:
            self.assertAlmostEqual(lambda_values[key],actual_lambda_values[key], places=5)

        for test, actual in zip (lambda_values['ln resid sum squares'],actual_lambda_values['ln resid sum squares']):
            self.assertAlmostEqual(test, actual, places=5)

    def test_box_cox_transform(self):
        test_vals = power_transform.box_cox_transform(data[response],
                                                      0.5)
        actual_vals = [12.699073601795252,
                        12.776788045024915,
                        12.791061924212153,
                        12.749380120828208,
                        12.750394977033277,
                        12.672936080924346,
                        12.717023474402486,
                        12.7395435333491,
                        12.66467071055358,
                        12.73450216508347,
                        12.714790780350235,
                        12.725619932912183,
                        12.759354355313246,
                        12.678883215715775,
                        12.717613789033225]

        for test, actual in zip (test_vals,actual_vals):
            self.assertAlmostEqual(test, actual, places=5)

if __name__ == '__main__':
    unittest.main()
