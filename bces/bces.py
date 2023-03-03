from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import scipy
import scipy.stats




# BCES fitting
# ===============

def bces(y1,y1err,y2,y2err,cerr):
	"""
Does the entire regression calculation for 4 slopes:
OLS(Y|X), OLS(X|Y), bisector, orthogonal.
Fitting form: Y=AX+B.

Usage:

>>> a,b,aerr,berr,covab=bces(x,xerr,y,yerr,cov)

Output:

- a,b : best-fit parameters a,b of the linear regression 
- aerr,berr : the standard deviations in a,b
- covab : the covariance between a and b (e.g. for plotting confidence bands)

Arguments:

- x,y : data
- xerr,yerr: measurement errors affecting x and y
- cov : covariance between the measurement errors
(all are arrays)
	"""
	# Arrays holding the code main results for each method:
	# Elements: 0-Y|X, 1-X|Y, 2-bisector, 3-orthogonal
	a,b,avar,bvar,covarxiz,covar_ba=np.zeros(4),np.zeros(4),np.zeros(4),np.zeros(4),np.zeros(4),np.zeros(4)
	# Lists holding the xi and zeta arrays for each method above
	xi,zeta=[],[]
	
	# Calculate sigma's for datapoints using length of conf. intervals
	sig11var = np.mean( y1err**2 )
	sig22var = np.mean( y2err**2 )
	sig12var = np.mean( cerr )
	
	# Covariance of Y1 (X) and Y2 (Y)
	covar_y1y2 = np.mean( (y1-y1.mean())*(y2-y2.mean()) )

	# Compute the regression slopes
	a[0] = (covar_y1y2 - sig12var)/(y1.var() - sig11var)	# Y|X
	a[1] = (y2.var() - sig22var)/(covar_y1y2 - sig12var)	# X|Y
	a[2] = ( a[0]*a[1] - 1.0 + np.sqrt((1.0 + a[0]**2)*(1.0 + a[1]**2)) ) / (a[0]+a[1])	# bisector
	if covar_y1y2<0:
		sign = -1.
	else:
		sign = 1.
	a[3] = 0.5*((a[1]-(1./a[0])) + sign*np.sqrt(4.+(a[1]-(1./a[0]))**2))	# orthogonal
	
	# Compute intercepts
	for i in range(4):
		b[i]=y2.mean()-a[i]*y1.mean()
	
	# Set up variables to calculate standard deviations of slope/intercept 
	xi.append(	( (y1-y1.mean()) * (y2-a[0]*y1-b[0]) + a[0]*y1err**2 ) / (y1.var()-sig11var)	)	# Y|X
	xi.append(	( (y2-y2.mean()) * (y2-a[1]*y1-b[1]) - y2err**2 ) / covar_y1y2	)	# X|Y
	xi.append(	xi[0] * (1.+a[1]**2)*a[2] / ((a[0]+a[1])*np.sqrt((1.+a[0]**2)*(1.+a[1]**2))) + xi[1] * (1.+a[0]**2)*a[2] / ((a[0]+a[1])*np.sqrt((1.+a[0]**2)*(1.+a[1]**2)))	)	# bisector
	xi.append(	xi[0] * a[3]/(a[0]**2*np.sqrt(4.+(a[1]-1./a[0])**2)) + xi[1]*a[3]/np.sqrt(4.+(a[1]-1./a[0])**2)	)	# orthogonal
	for i in range(4):
		zeta.append( y2 - a[i]*y1 - y1.mean()*xi[i]	)

	for i in range(4):
		# Calculate variance for all a and b
		avar[i]=xi[i].var()/xi[i].size
		bvar[i]=zeta[i].var()/zeta[i].size
		
		# Sample covariance obtained from xi and zeta (paragraph after equation 15 in AB96)
		covarxiz[i]=np.mean( (xi[i]-xi[i].mean()) * (zeta[i]-zeta[i].mean()) )
	
	# Covariance between a and b (equation after eq. 15 in AB96)
	covar_ab=covarxiz/y1.size
	
	return a,b,np.sqrt(avar),np.sqrt(bvar),covar_ab
	
	
	
	
def bootstrap(v):
	"""
Constructs Monte Carlo simulated data set using the
Bootstrap algorithm.                                                                                   

Usage:

>>> bootstrap(x)

where x is either an array or a list of arrays. If it is a
list, the code returns the corresponding list of bootstrapped 
arrays assuming that the same position in these arrays map the 
same "physical" object.
	"""
	if type(v)==list:
		vboot=[]	# list of boostrapped arrays
		n=v[0].size
		iran=scipy.stats.randint.rvs(0,n,size=n)	# Array of random indexes
		for x in v:	vboot.append(x[iran])
	else:	# if v is an array, not a list of arrays
		n=v.size
		iran=scipy.stats.randint.rvs(0,n,size=n)	# Array of random indexes
		vboot=v[iran]
	
	return vboot
	





def bcesboot(y1,y1err,y2,y2err,cerr,nsim=10000):
	"""
Does the BCES with bootstrapping.	

Usage:

>>> a,b,aerr,berr,covab=bcesboot(x,xerr,y,yerr,cov,nsim)

:param x,y: data
:param xerr,yerr: measurement errors affecting x and y
:param cov: covariance between the measurement errors (all are arrays)
:param nsim: number of Monte Carlo simulations (bootstraps)

:returns: a,b -- best-fit parameters a,b of the linear regression 
:returns: aerr,berr -- the standard deviations in a,b
:returns: covab -- the covariance between a and b (e.g. for plotting confidence bands)

.. note:: this method is definitely not nearly as fast as bces_regress.f. Needs to be optimized. Maybe adapt the fortran routine using f2python?
	"""
	import tqdm
	
	print("Bootstrapping progress:")
	
	"""
	My convention for storing the results of the bces code below as 
	matrixes for processing later are as follow:
	
	      simulation-method  y|x x|y bisector orthogonal
	          sim0           ...
	Am =      sim1           ...
	          sim2           ...
	          sim3           ...
	"""
	for i in tqdm.tqdm(range(nsim)):
		# This is needed for small datasets. With a dataset of e.g. 4 points,
		# bootstrapping can generate a mock array with 4 repeated points. This
		# will cause an error in the linear regression.
		allEquals=True
		while allEquals:
			[y1sim,y1errsim,y2sim,y2errsim,cerrsim]=bootstrap([y1,y1err,y2,y2err,cerr])

			allEquals=allEqual(y1sim)
		
		asim,bsim,errasim,errbsim,covabsim=bces(y1sim,y1errsim,y2sim,y2errsim,cerrsim)	
		
		if i==0:
			# Initialize the matrixes
			am,bm=asim.copy(),bsim.copy()
		else: 
			am=np.vstack((am,asim))
			bm=np.vstack((bm,bsim))
				

	if True in np.isnan(am):
		am,bm=checkNan(am,bm)
	
	# Bootstrapping results
	a=np.array([ am[:,0].mean(),am[:,1].mean(),am[:,2].mean(),am[:,3].mean() ])
	b=np.array([ bm[:,0].mean(),bm[:,1].mean(),bm[:,2].mean(),bm[:,3].mean() ])

	# Error from unbiased sample variances
	erra,errb,covab=np.zeros(4),np.zeros(4),np.zeros(4)
	for i in range(4):
		erra[i]=np.sqrt( 1./(nsim-1) * ( np.sum(am[:,i]**2)-nsim*(am[:,i].mean())**2 ))
		errb[i]=np.sqrt( 1./(nsim-1) * ( np.sum(bm[:,i]**2)-nsim*(bm[:,i].mean())**2 ))
		covab[i]=1./(nsim-1) * ( np.sum(am[:,i]*bm[:,i])-nsim*am[:,i].mean()*bm[:,i].mean() )
	
	return a,b,erra,errb,covab
	


def checkNan(am,bm):
	"""
	Sometimes, if the dataset is very small, the regression parameters in
	some instances of the bootstrapped sample may have NaNs i.e. failed
	regression (I need to investigate this in more details).
	
	This method checks to see if there are NaNs in the bootstrapped 
	fits and remove them from the final sample.
	"""
	import nmmn.lsd

	idel=nmmn.lsd.findnan(am[:,2])
	print("Bootstrapping error: regression failed in",np.size(idel),"instances. They were removed.")

	return np.delete(am,idel,0),np.delete(bm,idel,0)



def allEqual(x):
	"""
Check if all elements in an array are equal. 
Returns True if they are all the same.
	"""
	from itertools import groupby

	g = groupby(x)

	return next(g, True) and not next(g, False)






	
# Methods which make use of parallelization
# ===========================================


def ab(x):
	"""
This method is the big bottleneck of the parallel BCES code. That's the 
reason why I put these calculations in a separate method, in order to 
distribute this among the cores. In the original BCES method, this is 
inside the main routine.
	
Argument:
[y1,y1err,y2,y2err,cerr,nsim]
where nsim is the number of bootstrapping trials sent to each core.

:returns: am,bm : the matrixes with slope and intercept where each line corresponds to a bootrap trial and each column maps a different BCES method (ort, y|x etc).

Be very careful and do not use lambda functions when calling this 
method and passing it to multiprocessing or ipython.parallel!
I spent >2 hours figuring out why the code was not working until I
realized the reason was the use of lambda functions.
	"""
	y1,y1err,y2,y2err,cerr,nsim=x[0],x[1],x[2],x[3],x[4],x[5]
	
	for i in range(int(nsim)):
		# This is needed for small datasets. With datasets of 4 points or less,
		# bootstrapping can generate a mock array with 4 repeated points. This
		# will cause an error in the linear regression.
		allEquals=True
		while allEquals:
			[y1sim,y1errsim,y2sim,y2errsim,cerrsim]=bootstrap([y1,y1err,y2,y2err,cerr])

			allEquals=allEqual(y1sim)

		asim,bsim,errasim,errbsim,covabsim=bces(y1sim,y1errsim,y2sim,y2errsim,cerrsim)	
	
		if i==0:
			# Initialize the matrixes
			am,bm=asim.copy(),bsim.copy()
		else: 
			am=np.vstack((am,asim))
			bm=np.vstack((bm,bsim))
		
	return am,bm

	



def bcesp(y1,y1err,y2,y2err,cerr,nsim=10000):
	"""
Parallel implementation of the BCES with bootstrapping.
Divide the bootstraps equally among the threads (cores) of
the machine. It will automatically detect the number of
cores available.

Usage:

>>> a,b,aerr,berr,covab=bcesp(x,xerr,y,yerr,cov,nsim)

:param x,y: data
:param xerr,yerr: measurement errors affecting x and y
:param cov: covariance between the measurement errors (all are arrays)
:param nsim: number of Monte Carlo simulations (bootstraps)

:returns: a,b - best-fit parameters a,b of the linear regression 
:returns: aerr,berr - the standard deviations in a,b
:returns: covab - the covariance between a and b (e.g. for plotting confidence bands)

.. seealso:: Check out ~/work/projects/playground/parallel python/bcesp.py for the original, testing, code. I deleted some line from there to make the "production" version.

* v1 Mar 2012: serial version ported from bces_regress.f. Added covariance output.
* v2 May 3rd 2012: parallel version ported from nemmen.bcesboot.

.. codeauthor: Rodrigo Nemmen
	"""	
	import time	# for benchmarking
	import multiprocessing
	
	print("BCES,", nsim,"trials... ")
	tic=time.time()
	
	# Find out number of cores available
	ncores=multiprocessing.cpu_count()
	# We will divide the processing into how many parts?
	n=2*ncores
	
	"""
	Must create lists that will be distributed among the many
	cores with structure 
	core1 <- [y1,y1err,y2,y2err,cerr,nsim/n]
	core2 <- [y1,y1err,y2,y2err,cerr,nsim/n]
	etc...
	"""
	pargs=[]	# this is a list of lists!
	for i in range(n):
		pargs.append([y1,y1err,y2,y2err,cerr,nsim/n])
	
	# Initializes the parallel engine
	pool = multiprocessing.Pool(processes=ncores)	# multiprocessing package
			
	"""
	Each core processes ab(input)
		return matrixes Am,Bm with the results of nsim/n
		presult[i][0] = Am with nsim/n lines
		presult[i][1] = Bm with nsim/n lines
	"""
	presult=pool.map(ab, pargs)	# multiprocessing
	pool.close()	# close the parallel engine
	
	# vstack the matrixes processed from all cores
	i=0
	for m in presult:
		if i==0:
			# Initialize the matrixes
			am,bm=m[0].copy(),m[1].copy()
		else: 
			am=np.vstack((am,m[0]))
			bm=np.vstack((bm,m[1]))
		i=i+1
	
	if True in np.isnan(am):
		am,bm=checkNan(am,bm)

	# Computes the bootstrapping results on the stacked matrixes
	a=np.array([ am[:,0].mean(),am[:,1].mean(),am[:,2].mean(),am[:,3].mean() ])
	b=np.array([ bm[:,0].mean(),bm[:,1].mean(),bm[:,2].mean(),bm[:,3].mean() ])

	# Error from unbiased sample variances
	erra,errb,covab=np.zeros(4),np.zeros(4),np.zeros(4)
	for i in range(4):
		erra[i]=np.sqrt( 1./(nsim-1) * ( np.sum(am[:,i]**2)-nsim*(am[:,i].mean())**2 ))
		errb[i]=np.sqrt( 1./(nsim-1) * ( np.sum(bm[:,i]**2)-nsim*(bm[:,i].mean())**2 ))
		covab[i]=1./(nsim-1) * ( np.sum(am[:,i]*bm[:,i])-nsim*am[:,i].mean()*bm[:,i].mean() )
	
	print("%f s" % (time.time() - tic))
	
	return a,b,erra,errb,covab



