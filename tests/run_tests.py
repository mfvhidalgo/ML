# python -m tests.run_tests
import unittest

test_dir = 'tests'
pattern = '*.py'

loader = unittest.TestLoader()
suite = loader.discover(start_dir=test_dir, pattern=pattern)

runner = unittest.TextTestRunner()
runner.run(suite)
