# python -m tests.functions.evaluation.TestPower

import unittest
import pandas as pd

import src.ml_utils.design_eval.power as power

class TestPower(unittest.TestCase):
    def setUp(self):
        self.design = pd.read_excel('tests//functions//evaluation//OD.xlsx',sheet_name='design')
        self.eval = pd.read_excel('tests//functions//evaluation//OD.xlsx',sheet_name='eval')
        self.model = 'A + C + D + B\
                + C:D + A:B + A:C + A:D + B:C + B:D \
                + I(np.power(A,2)) + I(np.power(C,2)) + I(np.power(B,2))  + I(np.power(D,2))'
        self.term = 'C:D'
        self.goal_power = 0.77
        print("setUp: Preparing the test environment")

    def tearDown(self):
        # Code to clean up after tests
        self.design = None
        self.eval = None
        self.model = None
        self.term = None
        self.goal_power = None
        print("tearDown: Cleaning up the test environment")

    def test_est_signal_to_noise(self):
        result = power.est_signal_to_noise(self.design,self.model,self.term,self.goal_power)
        self.assertAlmostEqual(round(result,2), 2, places=2)

    def test_get_power(self):
        terms, powers = power.get_power(self.design, self.model, 2)
        self.assertCountEqual(terms,
                             ['1','A','C','D','B',\
                              'C:D', 'A:B', 'A:C', 'A:D', 'B:C', 'B:D',\
                              'I(np.power(A, 2))', 'I(np.power(C, 2))', 'I(np.power(B, 2))', 'I(np.power(D, 2))'])
        self.assertCountEqual([round(power,3) for power in powers[1:]],
                             [0.9220, 0.9500, 0.9330, 0.9350, \
                              0.7700, 0.7520, 0.7630, 0.7520, 0.7710, 0.7510, \
                              0.9730, 0.9790, 0.9650, 0.9660])
        
if __name__ == '__main__':
    unittest.main()








