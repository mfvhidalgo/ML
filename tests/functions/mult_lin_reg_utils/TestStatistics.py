# python -m tests.functions.mult_lin_reg_utils.TestStatistics
import unittest
from statsmodels.formula.api import ols
import pandas as pd

import src.functions.mult_lin_reg_utils.statistics as statistics

class TestStatistics(unittest.TestCase):

    def test_calc_r2_press(self):
        x = [0,.46,.27,1,.51,0,.5,0]
        y = [0,.47,.26,0,0,.5,.5,1]
        z = [1,.07,.47,0,.49,.5,0,0]
        resp = [93,95,99,100,104,97,108,101]

        df = pd.DataFrame({'x':x,
                        'y':y,
                        'z':z,
                        'resp':resp})

        model = ols('resp~x+y+z+x:y+x:z+y:z-1',df).fit()
        self.assertAlmostEqual(statistics.calc_r2_press(model),
                            -276.4972641859001,
                            places=5)

    def test_calc_r2_press_grouped(self):
        df = pd.DataFrame({'A':[85, 85, 85, 120, 120, 120, 145, 145, 145, 85, 85, 85, 120, 120, 120, 145, 145, 145],
                            'B': [2.7, 2.7, 2.7, 2.7, 2.7, 2.7, 2.7, 2.7, 2.7, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2],
                            'C': [39.44, 33.83, 28.22, 39.44, 33.83, 28.22, 39.44, 33.83, 28.22, 39.44, 33.83, 28.22, 39.44, 33.83, 28.22, 39.44, 33.83, 28.22],
                            '_10C': [72.04128926, 85.18668402, 89.82885543, 59.17462245, 81.77399638, 84.95119096, 30.73303112, 84.94575844, 69.42189115, 0.000889655, 21.63635015, 11.60130742, 3.496609912, 17.1987154, 14.89253066, 0.000892141, 15.86273027, 11.62351169]
                            })

        model = ols(formula='_10C~A+B+C',data=df).fit()

        self.assertAlmostEqual(statistics.calc_r2_press(model, groupby_cols=['A', 'B']),
                            0.843028138,
                            places=5)

if __name__ == '__main__':
    unittest.main()
