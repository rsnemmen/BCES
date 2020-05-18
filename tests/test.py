import unittest
import bces.bces as bces
import numpy as np

class TestBCES(unittest.TestCase):

	# reads test dataset
	data=np.load('../data.npz')
	xdata=data['x']
	ydata=data['y']
	errx=data['errx']
	erry=data['erry']
	covdata=data['cov']

	# correct fit parameters expected for dataset
	ans_pars=np.array([ 0.57955173, 17.88855826])
	ans_cov=np.array([[ 5.85029731e-04, -2.72808055e-02],
		[-2.72808055e-02,  1.27299029e+00]])


	def test_fit(self):
		"""
		Test BCES Y|X fit without bootstrapping.
		"""
		# fit
		a,b,erra,errb,covab=bces(xdata,errx,ydata,erry,covdata)

		np.testing.assert_array_almost_equal(ans_pars,np.array([a[0],b[0]]))
		#self.assertEqual(result, 6)




if __name__ == '__main__':
    unittest.main()