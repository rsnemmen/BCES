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

# correct fit parameters expected for dataset Y|X
ans_pars=np.array([ 0.57955173, 17.88855826])
ans_cov=np.array([[ 5.85029731e-04, -2.72808055e-02],
	[-2.72808055e-02,  1.27299029e+00]])

def test_yx():
	"""
	Test BCES Y|X fit without bootstrapping.
	"""
	# fit
	#a,b,erra,errb,covab=B.bcesp(xdata,errx,ydata,erry,covdata)
	a,b,erra,errb,covab=BCES.bces(xdata,errx,ydata,erry,covdata)

	np.testing.assert_array_almost_equal(ans_pars,np.array([a[0],b[0]]))

def test_yxboot():
	"""
	Test BCES Y|X fit with bootstrapping.
	"""
	# fit
	a,b,erra,errb,covab=BCES.bcesp(xdata,errx,ydata,erry,covdata)

	# check if the regression parameters match within 1 decimal
	np.testing.assert_array_almost_equal(ans_pars,np.array([a[0],b[0]]),1)

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