# python -m tests.functions.mult_lin_reg_utils.TestStatistics
import unittest
from statsmodels.formula.api import ols
import pandas as pd
import dexpy.factorial
import numpy as np

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

    def test_evaluate_data(self):
        formula = 'X1 + X2 + X3 + X4 + X1:X2 + X1:X3 + X1:X4 + X2:X3 + X2:X4 + X3:X4'
        signal_to_noise = 2
        alpha = 0.05
        df = dexpy.factorial.build_factorial(4, 8)
        df = pd.concat([df,df.iloc[-1,:].to_frame().T])
        evals = statistics.evaluate_data(formula=formula,
                                        data=df,
                                        signal_to_noise=signal_to_noise,
                                        alpha=alpha)
        for vif,vif_actual in zip(evals['VIF'],[1.04167]*7+['']*3):
            if vif_actual == '':
                self.assertEqual(vif,vif_actual)
            else:
                self.assertAlmostEqual(vif,vif_actual,places=5)   
            
        for power,power_actual in zip(evals['power'],[18.1]*7+['']*3):
            if vif_actual == '':
                self.assertEqual(vif,vif_actual)
            else:
                self.assertAlmostEqual(power,power_actual,places=1)       

        # case where there are not enough experiments to calculate power and VIF    
        df = dexpy.factorial.build_factorial(4, 8)
        evals = statistics.evaluate_data(formula=formula,
                                        data=df,
                                        signal_to_noise=signal_to_noise,
                                        alpha=alpha)
        for vif,vif_actual in zip(evals['VIF'],[1]*7+['']*3):
            if vif_actual == '':
                self.assertEqual(vif,vif_actual)
            else:
                self.assertAlmostEqual(vif,vif_actual,places=1)   
            
        for power,power_actual in zip(evals['power'],[np.nan]*7+['']*3):
            self.assertEqual(vif,vif_actual)

    def test_evaluate_data_no_alias(self):
        formula = 'A + B + C + A:B + A:C + B:C + I(A**2) + I(B**2) + I(C**2)'
        signal_to_noise = 2
        alpha = 0.05
        df = pd.DataFrame({'A':[-1,-0.3,1,-1,0.18,-1,1,0.08,1,1,0.1,-0.3],
                           'B':[1,-1,-1,-0.17,1,-1,-1,-0.09,1,0.3,-0.09,0.3],
                           'C':[-0.18,0.3,-1,1,1,-1,1,-0.1,-1,0.3,-0.08,-1]})
        evals = statistics.evaluate_data(formula=formula,
                                        data=df,
                                        signal_to_noise=signal_to_noise,
                                        alpha=alpha)
        for vif,vif_actual in zip(evals['VIF'],[1.11454,1.11524,1.11454,
                                                1.14541,1.14681,1.14541,
                                                1.29295,1.29247,1.29295]):
            if vif_actual == '':
                self.assertEqual(vif,vif_actual)
            else:
                self.assertAlmostEqual(vif,vif_actual,places=5)   
            
        for power,power_actual in zip(evals['power'],[0.306,0.306,0.306,
                                                      0.24,0.24,0.24,
                                                      0.364,0.365,0.364]):
            if vif_actual == '':
                self.assertEqual(vif,vif_actual)
            else:
                self.assertAlmostEqual(power,power_actual,places=0)       


if __name__ == '__main__':
    unittest.main()
