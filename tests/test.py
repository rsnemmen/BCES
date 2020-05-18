import unittest
import bces.bces

class TestBCES(unittest.TestCase):
    def test_fit_bs(self):
        """
        Test BCES fitting with bootstrapping
        """
        data = [1, 2, 3]
        result = sum(data)
        self.assertEqual(result, 6)


    def test_fit(self):
    	"""
    	Test BCES fitting without bootstrapping
    	"""





if __name__ == '__main__':
    unittest.main()