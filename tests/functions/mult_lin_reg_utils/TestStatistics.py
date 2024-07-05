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

if __name__ == '__main__':
    unittest.main()
