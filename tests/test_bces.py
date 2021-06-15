"""
Basic unit testing functionality.
"""
import bces.bces as BCES
import numpy as np

# reads test dataset
data=np.load('data.npz')
xdata=data['x']
ydata=data['y']
errx=data['errx']
erry=data['erry']
covdata=data['cov']

# Correct fit parameters expected for dataset. The parameters in 
# ans_pars are such that y=Ax+B:
#
# ans_pars = [ A(y|x), B(y|x), 
#              A(x|y), B(x|y), 
#              A(ort), B(ort) ]
#
# i.e., each pair contains the expected result for one of the BCES
# regression methods.
#
ans_pars=np.array([ [0.57955173, 17.88855826], 
	                [0.26053751, 32.70952271], 
	                [0.50709256, 21.25491222] ])
# covariance matrix
ans_cov=np.array([[ 5.85029731e-04, -2.72808055e-02], 
				  [-2.72808055e-02,  1.27299029e+00]])

def test_yx():
	"""
	Test BCES Y|X fit without bootstrapping.
	"""
	# fit
	a,b,erra,errb,covab=BCES.bces(xdata,errx,ydata,erry,covdata)

	np.testing.assert_array_almost_equal([ans_pars[0,0],ans_pars[0,1]], np.array([a[0],b[0]]))

def test_xy():
	"""
	Test BCES X|Y fit without bootstrapping.
	"""
	# fit
	a,b,erra,errb,covab=BCES.bces(xdata,errx,ydata,erry,covdata)

	np.testing.assert_array_almost_equal([ans_pars[1,0],ans_pars[1,1]], np.array([a[1],b[1]]))

def test_ort():
	"""
	Test BCES orthogonal fit without bootstrapping.
	"""
	# fit
	a,b,erra,errb,covab=BCES.bces(xdata,errx,ydata,erry,covdata)

	np.testing.assert_array_almost_equal([ans_pars[2,0],ans_pars[2,1]], np.array([a[3],b[3]]))

def test_yxboot():
	"""
	Test BCES Y|X fit with bootstrapping.
	"""
	# fit
	a,b,erra,errb,covab=BCES.bcesp(xdata,errx,ydata,erry,covdata)

	# check if the regression parameters match within 1 decimal
	np.testing.assert_array_almost_equal([ans_pars[0,0],ans_pars[0,1]], np.array([a[0],b[0]]),1)

def test_ortboot():
	"""
	Test BCES orthogonal fit with bootstrapping.
	"""
	# fit
	a,b,erra,errb,covab=BCES.bcesp(xdata,errx,ydata,erry,covdata)

	# check if the regression parameters match within 1 decimal
	np.testing.assert_array_almost_equal([ans_pars[2,0],ans_pars[2,1]], np.array([a[3],b[3]]),1)

def test_bootstrap():
	"""
	Test if bootstrap is working correctly.
	"""
	import scipy.stats

	# number of bootstrap samples
	nboot=5000

	# bootstrapping procedure
	ts=[] # test statistic
	for i in range(nboot):
	    xsim=BCES.bootstrap(xdata)
	    tsim,asim,psim=scipy.stats.anderson_ksamp([xdata,xsim])
	    ts.append(tsim)

	ts=np.array(ts)

	# is the simulated (bootstrapped) dataset consistent with the
	# original one? 
	assert np.median(ts)<asim[0]