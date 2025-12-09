# python -m tests.functions.mult_lin_reg_utils.TestTerms
import unittest

import src.functions.mult_lin_reg_utils.terms as terms

class TestTerms(unittest.TestCase):
    def test_get_base_exponent(self):
        self.assertEqual(terms.get_base_exponent('I(C**3)'), ['C',3])
        self.assertEqual(terms.get_base_exponent('I(C**3.0)'), ['C',3])
        self.assertEqual(terms.get_base_exponent('I(C**3.5)'), ['C',3.5])
        self.assertEqual(terms.get_base_exponent('C'), ['C',1])
        self.assertEqual(terms.get_base_exponent('I(C**5)'), ['C',5])
        self.assertEqual(terms.get_base_exponent('np.power(C, 5)'), ['C',5])
        self.assertEqual(terms.get_base_exponent('np.power(C,5 )'), ['C',5])
        self.assertEqual(terms.get_base_exponent('np.sin(C)'), ['np.sin(C)',1])
        with self.assertRaises(ValueError) as context:
            terms.get_base_exponent('A:I(C**2)')
        self.assertIn('term contains interaction and not just an exponent', str(context.exception))
    
    def test_sort_terms(self):
        self.assertEqual(terms.sort_terms(['A','I(A**3)','B']),
                         ['A','B','I(A**3)'])
        self.assertEqual(terms.sort_terms(['A','I(A**3)','C:D']),
                         ['A','C:D','I(A**3)'])
        
    def test_tuple_to_interaction(self):
        self.assertEqual(terms.tuple_to_interaction(('A','I(A**3)','C:D')),
                         '')
        self.assertEqual(terms.tuple_to_interaction(('A','A:B','C:D')),
                         '')
        self.assertEqual(terms.tuple_to_interaction(('A','B','C:D')),
                         'A:B:C:D')
    
    def test_tuple_to_term(self):
        self.assertEqual(terms.tuple_to_term(('k','k','C')),
                         'C:I(k**2)')
        self.assertEqual(terms.tuple_to_term(('k','k','C','B','B','B')),
                         'C:I(k**2):I(B**3)')
        self.assertEqual(terms.tuple_to_term(('B:C','B')),
                         'C:I(B**2)')
        self.assertEqual(terms.tuple_to_term(('B:C','I(B**2)')),
                         'C:I(B**3)')
        self.assertEqual(terms.tuple_to_term(('B:C:A:K','I(B**2)','B')),
                         'A:C:K:I(B**4)')
        self.assertEqual(terms.tuple_to_term(('B:C:A:K','B:I(B**2)','B')),
                         'A:C:K:I(B**5)')
        self.assertEqual(terms.tuple_to_term(('B:C:A:K','A:I(B**2)','B')),
                         'C:K:I(A**2):I(B**4)')
        self.assertEqual(terms.tuple_to_term(('B:C:A:K','A:I(B**2)','B'),True),
                         ['C:K:I(A**2):I(B**4)',8])

    def test_get_base_order(self):
        self.assertEqual(terms.get_base_order(('A:B:C')),
                         ['A',3])
        self.assertEqual(terms.get_base_order(('B:I(A**2)')),
                         ['B',3])
        self.assertEqual(terms.get_base_order(('B:I(A**2):I(B**3)')),
                         ['B',6])

    def test_get_all_higher_order_terms(self):
        self.assertEqual(terms.get_all_higher_order_terms(('A','B','C'),2),
                         ['A','B','C','A:B','A:C','B:C','I(A**2)','I(B**2)','I(C**2)'])
        self.assertEqual(terms.get_all_higher_order_terms(('A','B','C'),3),
                         ['A','B','C',
                          'A:B','A:C','B:C',
                          'I(A**2)','I(B**2)','I(C**2)',
                          'A:B:C',
                          'A:I(B**2)','A:I(C**2)',
                          'B:I(A**2)','B:I(C**2)',
                          'C:I(A**2)','C:I(B**2)',
                          'I(A**3)','I(B**3)','I(C**3)'])
        self.assertEqual(terms.get_all_higher_order_terms(('A','B','C'),3,2),
                         ['A','B','C',
                          'A:B','A:C','B:C',
                          'I(A**2)','I(B**2)','I(C**2)',
                          'A:B:C',
                          'A:I(B**2)','A:I(C**2)',
                          'B:I(A**2)','B:I(C**2)',
                          'C:I(A**2)','C:I(B**2)'])

    def test_list_to_orders(self):
        self.assertEqual(terms.list_to_orders(['A','B','C']),
                         {1:['A','B','C']})
        self.assertEqual(terms.list_to_orders(['A','B','C','I(C**2)','A:B','A:I(C**2)']),
                         {1:['A','B','C'],
                          2:['I(C**2)','A:B',],
                          3:['A:I(C**2)']})

    def test_list_to_formula(self):
        self.assertEqual(terms.list_to_formula(''),
                         'response~1')



if __name__ == '__main__':
    unittest.main()

