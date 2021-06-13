"""
Basic unit testing functionality.
"""
import bces.bces as B
import numpy as np

# reads test dataset
data=np.load('data.npz')
xdata=data['x']
ydata=data['y']
errx=data['errx']
erry=data['erry']
covdata=data['cov']

# correct fit parameters expected for dataset
ans_pars=np.array([ 0.57955173, 17.88855826])
ans_cov=np.array([[ 5.85029731e-04, -2.72808055e-02],
	[-2.72808055e-02,  1.27299029e+00]])

def test_yx():
	"""
	Test BCES Y|X fit without bootstrapping.
	"""
	# fit
	#a,b,erra,errb,covab=B.bcesp(xdata,errx,ydata,erry,covdata)
	a,b,erra,errb,covab=B.bces(xdata,errx,ydata,erry,covdata)

	np.testing.assert_array_almost_equal(ans_pars,np.array([a[0],b[0]]))

def test_yxboot():
	"""
	Test BCES Y|X fit with bootstrapping.
	"""
	# fit
	a,b,erra,errb,covab=B.bcesp(xdata,errx,ydata,erry,covdata)

	# check if the regression parameters match within 1 decimal
	np.testing.assert_array_almost_equal(ans_pars,np.array([a[0],b[0]]),1)
