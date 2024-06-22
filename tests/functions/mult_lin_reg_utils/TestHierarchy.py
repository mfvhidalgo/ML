# python -m tests.functions.mult_lin_reg_utils.TestHierarchy
import unittest

import src.functions.mult_lin_reg_utils.hierarchy as hierarchy

class TestHierarchy(unittest.TestCase):

    def test_get_all_combo_from_exponent(self):
        self.assertCountEqual(hierarchy.get_all_combo_from_exponent('I(C**3)'), ['I(C**3)','I(C**2)','C'])
        self.assertCountEqual(hierarchy.get_all_combo_from_exponent('C'), ['C'])
        self.assertCountEqual(hierarchy.get_all_combo_from_exponent('np.power(C,5)'),
                             ['I(C**5)','I(C**4)','I(C**3)','I(C**2)','C'])
        
    def test_get_all_combo_from_interaction(self):
        self.assertCountEqual(hierarchy.get_all_combo_from_interaction('A:B'), ['A','B','A:B'])
        self.assertCountEqual(hierarchy.get_all_combo_from_interaction('A:I(B**2)'), ['A','B','A:B','I(B**2)','A:I(B**2)'])
        self.assertCountEqual(hierarchy.get_all_combo_from_interaction('A:B:C'), ['A','B','C','A:B','A:C','B:C','A:B:C'])
        self.assertCountEqual(hierarchy.get_all_combo_from_interaction('A:B:I(C**2)'), ['A','B','C',
                                                                                          'A:B','A:C','B:C','I(C**2)',
                                                                                          'A:B:C','A:I(C**2)','B:I(C**2)',
                                                                                          'A:B:I(C**2)'])
        self.assertCountEqual(hierarchy.get_all_combo_from_interaction('I(A**2):B:I(C**3)'), ['A','B','C',
                                                                                          'A:B','A:C','B:C','I(A**2)','I(C**2)',
                                                                                          'A:B:C','I(A**2):B','I(A**2):C','A:I(C**2)','B:I(C**2)','I(C**3)',
                                                                                          'I(A**2):B:C','I(A**2):I(C**2)','A:B:I(C**2)','A:I(C**3)','B:I(C**3)',
                                                                                          'I(A**2):B:I(C**2)','I(A**2):I(C**3)','A:B:I(C**3)',
                                                                                          'I(A**2):B:I(C**3)'])
        
        # check that output terms are sorted
        self.assertCountEqual(hierarchy.get_all_combo_from_interaction('I(C**2):A'),['A','C',
                                                                                       'A:C','I(C**2)',
                                                                                       'A:I(C**2)'])

    def test_get_all_lower_order_terms(self):
        self.assertEqual(hierarchy.get_all_lower_order_terms('y ~ A + B + I(C**3)+ E:D'),
                                                             'y~A+B+C+D+E+D:E+I(C**2)+I(C**3)')

if __name__ == '__main__':
    unittest.main()








