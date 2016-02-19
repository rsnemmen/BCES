import numpy as np
from scipy.stats import norm

try:
    #Use numba if it is present, makes choosing bootstrap indices much faster.
    import numba
    
    @numba.jit
    def bootstrap_indexes(data, n_samples=10000, weights=None):
        """
        Given data points data, where axis 0 is considered to delineate points, return
        an array where each row is a set of bootstrap indexes. This can be used as a list
        of bootstrap indexes as well.
    
        Numba makes choosing ridiculously fast.
        e.g.:
        %timeit choice(np.random.randint(0,100,1000),size=(n_samples,1000), replace=True)
        10 loops, best of 3: 129 ms per loop
        
        @numba.jit
        def choose():
            return(choice(np.random.randint(0,100,1000),size=(n_samples,1000), replace=True)
        %timeit choose
        10000000 loops, best of 3: 24.4 ns per loop

        """
        if type(data) == list:
            return(np.random.choice(len(data[0]), 
                                    size=(n_samples, len(data[0])), 
                                    replace=True, p=weights))
        else: #assuming data is an array or dataframe with rows as different variables
            return(np.random.choice(data.shape[0],
                                    size=(n_samples, data.shape[1]), 
                                    replace=True, p=weights))
except ImportError:
    numba = None
    
    def bootstrap_indexes(data, n_samples=10000, weights=None):
        """
        """
        if type(data) == list:
            return(np.random.choice(len(data[0]), 
                                    size=(len(data), n_samples, len(data[0])), 
                                    replace=True, p=weights))
        else: #assuming data is an array or dataframe with rows as different variables
            return(np.random.choice(data.shape[0],
                                    size=(data.shape[0], n_samples, data.shape[1]), 
                                    replace=True, p=weights))


# BCES fitting
# ===============
def bces(y1, y1err, y2, y2err, cerr):
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

v1 Mar 2012: ported from bces_regress.f. Added covariance output.
Rodrigo Nemmen, http://goo.gl/8S1Oo

01/07/2016 Updates by Evan Groopman: vectorized operations
    """
    # Arrays holding the code main results for each method:
    # Elements: 0-Y|X, 1-X|Y, 2-bisector, 3-orthogonal
    a, b, avar, bvar, covarxiz, covar_ba = np.zeros((6,4)) #make use of unpacking
    # Lists holding the xi and zeta arrays for each method above
    xi, zeta = np.zeros((2, 4, len(y1)))
    
    # Calculate sigma's for datapoints using length of conf. intervals
    if cerr is None:
        cov = np.cov(y1err, y2err)
        sig11var = cov[0,0]
        sig22var = cov[1,1]
        sig12var = cov[0,1]
    else:
        sig11var = np.mean( y1err**2 )
        sig22var = np.mean( y2err**2 )
        sig12var = np.mean( cerr )
	
    # Covariance of Y1 (X) and Y2 (Y)
    covar_y1y2 = np.mean( (y1 - y1.mean())*(y2 - y2.mean()) )

    # Compute the regression slopes
    a[0] = (covar_y1y2 - sig12var)/(y1.var() - sig11var)	# Y|X
    a[1] = (y2.var() - sig22var)/(covar_y1y2 - sig12var)	# X|Y
    a[2] = ( a[0]*a[1] - 1.0 + np.sqrt((1.0 + a[0]**2)*(1.0 + a[1]**2)) ) / (a[0]+a[1])	# bisector
    if covar_y1y2 < 0:
        sign = -1.
    else:
        sign = 1.
    a[3] = 0.5*((a[1] - (1./a[0])) + sign*np.sqrt(4. + (a[1] - (1./a[0]))**2))	# orthogonal

    # Compute intercepts
    b = y2.mean() - a * y1.mean()
    	
    # Set up variables to calculate standard deviations of slope/intercept
    #empty lists + appending don't work with numba
    #need to make arrays and replace elements
    xi[0] =	( (y1 - y1.mean()) * (y2 - a[0]*y1 - b[0]) + a[0]*y1err**2 ) / (y1.var() - sig11var)# Y|X
    xi[1] =	( (y2 - y2.mean()) * (y2 - a[1]*y1 - b[1]) - y2err**2 ) / covar_y1y2		# X|Y
    xi[2] =	xi[0] * (1.+a[1]**2)*a[2] / ((a[0] + a[1])*np.sqrt((1. + a[0]**2)*(1. + a[1]**2))) + xi[1] * (1. + a[0]**2)*a[2] / ((a[0] + a[1])*np.sqrt((1. + a[0]**2)*(1. + a[1]**2)))	# bisector
    xi[3] =	xi[0] * a[3]/(a[0]**2*np.sqrt(4. + (a[1] - 1./a[0])**2)) + xi[1]*a[3]/np.sqrt(4. + (a[1] - 1./a[0])**2)	# orthogonal
    
#    zeta = y2 - a*y1 - y1.mean(axis=1, keepdims=True)*xi
    for i in range(4):
        zeta[i] = y2 - a[i]*y1 - y1.mean()*xi[i]
    
    #making use of vectorization
    #small arrays, so these probably don't take much time in for loops,
    #but over 10,000 simulations can add up
    avar = xi.var(axis=1) / xi.shape[1]
    bvar = zeta.var(axis=1) / zeta.shape[1]
    
    # Sample covariance obtained from xi and zeta (paragraph after equation 15 in AB96)
    covarxiz = np.mean((xi - xi.mean(axis=1, keepdims=True)) * (zeta - zeta.mean(axis=1, keepdims=True)))

    # Covariance between a and b (equation after eq. 15 in AB96)
    covar_ab = covarxiz / y1.size

    return a, b, np.sqrt(avar), np.sqrt(bvar), covar_ab
	
def bces_v(y1, y1err, y2, y2err, cerr):
    """
An attempt to make the bces function more highly vectorized for efficiency purposes,
especially in bootstrap.
Expect each row to be a simulation. Add flag to row or column later.

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

v1 Mar 2012: ported from bces_regress.f. Added covariance output.
Rodrigo Nemmen, http://goo.gl/8S1Oo

01/07/2016 Updates by Evan Groopman
- fully vectorized to include bootstrap simulations
    """
    nsim = y1.shape[0] #how many rows (simulations)
    
    # Arrays holding the code main results for each method:
    # Elements: 0-Y|X, 1-X|Y, 2-bisector, 3-orthogonal
    a, b, avar, bvar, covarxiz, covar_ba = np.zeros((6, nsim, 4)) #make use of unpacking
    # Lists holding the xi and zeta arrays for each method above
    xi, zeta = np.zeros((2, nsim, 4, y1.shape[1]))
    
    # Calculate sigma's for datapoints using length of conf. intervals
    sig11var = np.mean(y1err**2, axis=1)
    sig22var = np.mean(y2err**2, axis=1)
    sig12var = np.mean(cerr, axis=1)
	
    # Covariance of Y1 (X) and Y2 (Y)
    covar_y1y2 = np.mean( (y1 - y1.mean(axis=1, keepdims=True))*(y2 - y2.mean(axis=1, keepdims=True)), axis=1)

    # Compute the regression slopes
    a[:,0] = (covar_y1y2 - sig12var)/(y1.var(axis=1) - sig11var)	 # Y|X
    a[:,1] = (y2.var(axis=1) - sig22var)/(covar_y1y2 - sig12var)	 # X|Y
    a[:,2] = ( a[:, 0]*a[:,1] - 1.0 + np.sqrt((1.0 + a[:,0]**2)* \
                (1.0 + a[:,1]**2)) ) / (a[:,0] + a[:,1])	# bisector

    sign = np.sign(covar_y1y2) # numpy has a built-in sign detection function

    a[:, 3] = 0.5*((a[:, 1] - (1./a[:, 0])) + sign*np.sqrt(4. + (a[:,1] - (1./a[:,0]))**2))	# orthogonal

    # Compute intercepts
    b = y2.mean(axis=1, keepdims=True) - a * y1.mean(axis=1, keepdims=True)
    	
    # Set up variables to calculate standard deviations of slope/intercept
    #empty lists + appending don't work with numba
    #need to make arrays and replace elements
    xi[:,0] =	  ( (y1 - y1.mean(axis=1, keepdims=True)) * (y2 - np.atleast_2d(a[:,0]).T*y1 - np.atleast_2d(b[:,0]).T) + np.atleast_2d(a[:,0]).T*y1err**2 ) / (y1.var(axis=1, keepdims=True) - np.atleast_2d(sig11var).T)# Y|X
    xi[:,1] =	  ( (y2 - y2.mean(axis=1, keepdims=True)) * (y2 - np.atleast_2d(a[:,1]).T*y1 - np.atleast_2d(b[:,1]).T) - y2err**2 ) / np.atleast_2d(covar_y1y2).T		# X|Y
    xi[:,2] =	  xi[:,0] * (1. + np.atleast_2d(a[:,1]**2).T)*np.atleast_2d(a[:,2]).T / np.atleast_2d((a[:,0] + a[:,1])* \
                np.sqrt((1. + a[:,0]**2)*(1. + a[:,1]**2))).T + xi[:,1] * \
                np.atleast_2d((1. + a[:,0]**2)*a[:,2] / ((a[:,0] + a[:,1])* \
                np.sqrt((1. + a[:,0]**2)*(1. + a[:,1]**2)))).T	# bisector
    xi[:,3] =	xi[:,0] * np.atleast_2d(a[:,3]/(a[:,0]**2*np.sqrt(4. + (a[:,1] - 1./a[:,0])**2))).T + xi[:,1] * np.atleast_2d(a[:,3]/np.sqrt(4. + (a[:,1] - 1./a[:,0])**2)).T	# orthogonal
    
#    zeta = y2 - a*y1 - y1.mean(axis=1, keepdims=True)*xi
    for i in range(4):
        zeta[:,i] = y2 - np.atleast_2d(a[:,i]).T*y1 - y1.mean(axis=1, keepdims=True)*xi[:,i]
        
    #making use of vectorization
    #small arrays, so these probably don't take much time in for loops,
    #but over 10,000 simulations can add up
    avar = xi.var(axis=2) / xi.shape[2]
    bvar = zeta.var(axis=2) / zeta.shape[2]
    
    # Sample covariance obtained from xi and zeta (paragraph after equation 15 in AB96)
    covarxiz = np.mean((xi - xi.mean(axis=2, keepdims=True)) * (zeta - zeta.mean(axis=2, keepdims=True)))

    # Covariance between a and b (equation after eq. 15 in AB96)
    covar_ab = covarxiz / y1.shape[1]

    return a, b, np.sqrt(avar), np.sqrt(bvar), covar_ab

def jackknife_indexes(data):
    """
Given data points data, where axis 0 is considered to delineate points, return
a list of arrays where each array is a set of jackknife indexes.

For a given set of data Y, the jackknife sample J[i] is defined as the data set
Y with the ith data point deleted.

References
----------
Efron, An Introduction to the Bootstrap. Chapman & Hall 1993

Adapted from Scikits.bootstrap by Constantine Evans
https://scikits.appspot.com/bootstrap
    """
    base = np.arange(0, len(data))
    return (np.delete(base,i) for i in base)
    
    
def bcesboot(y1, y1err, y2, y2err, cerr, n_samples=10000, alpha=0.05):
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

v1 Mar 2012: ported from bces_regress.f. Added covariance output.
Rodrigo Nemmen, http://goo.gl/8S1Oo


TODO: #Evan Groopman
- add bias correction to bootstrap, probably BCA. See Efron, An Introduction to the Bootstrap. Chapman & Hall 1993
- add 5 - 95% confidence intervals to bootstrap paramters

Speed examples (random noise): Time increases linearly with size of arrays.
x, y, xerr, yerr, cerr = np.random.rand(5,1000)
%timeit bcesboot(x,xerr,y,yerr,cerr)
1 loops, best of 3: 4.89 s per loop

x, y, xerr, yerr, cerr = np.random.rand(5,100)
%timeit bcesboot(x,xerr,y,yerr,cerr)
1 loops, best of 3: 493 ms per loop

x, y, xerr, yerr, cerr = np.random.rand(5,10)
%timeit bcesboot(x,xerr,y,yerr,cerr)
10 loops, best of 3: 40.5 ms per loop

BCA method: Bias-Corrected Accelerated Non-Parametric (Efron 14.3)
alpha is confidence interal. default 5 - 95%
"""
    alphas = np.array([alpha/2, 1-alpha/2])
    
    data = [y1, y1err, y2, y2err, cerr]
    indexes = bootstrap_indexes(data, n_samples=n_samples, weights=None)
    [y1sim, y1errsim, y2sim, y2errsim, cerrsim] = [data[i][indexes] for i in range(len(data))]
    
    asim, bsim, errasim, errbsim, covabsim = bces_v(y1sim, y1errsim, 
                                                  y2sim, y2errsim, cerrsim)	

    # Bootstrapping results
    a = asim.mean(axis=0)
    b = bsim.mean(axis=0)
    
    ### Confidence Intervals ###
    #Do bias correction, BCa method
    cis = np.zeros((2,2,4))
    
    for i, (sim, stat) in enumerate(zip((asim, bsim), (a, b))):
        sim.sort(axis=0)
        # The bias correction value.
        z0 = norm.ppf( ( 1.0*np.sum(sim < stat, axis=0)  ) / n_samples )
    
        # Statistics of the jackknife distribution
        jackindexes = jackknife_indexes(sim[:,0])
        jstat = np.array([[np.mean(s[ind]) for s in sim.T] for ind in jackindexes])
        jmean = jstat.mean(axis=0)
    
        # Acceleration value
        accel = np.sum( (jmean - jstat)**3, axis=0 ) / ( 6.0 * np.sum( (jmean - jstat)**2, axis=0)**1.5 )
        zs = z0 + norm.ppf(alphas).reshape(alphas.shape+(1,)*z0.ndim)
        avals = norm.cdf(z0 + zs/(1 - accel*zs))
        nvals = np.round((n_samples-1)*avals).astype('int')
#        cis[i] = sim[nvals]
    
    a_cis = cis[0]
    b_cis = cis[1]
    
    # Error from unbiased sample variances
    erra, errb, covab = np.zeros((3,4))
    for i in range(4):
#        erra[i] = np.sqrt( 1./(n_samples-1) * ( np.sum(asim[:,i]**2)-n_samples*(asim[:,i].mean())**2 ))
#        errb[i] = np.sqrt( 1./(n_samples-1) * ( np.sum(bsim[:,i]**2)-n_samples*(bsim[:,i].mean())**2 ))
        covab[i] = 1./(n_samples-1) * ( np.sum(asim[:,i]*bsim[:,i])-n_samples*asim[:,i].mean()*bsim[:,i].mean() )
    
    erra = asim.std(axis=0, ddof=1)
    errb = bsim.std(axis=0, ddof=1)
    return(a, b, erra, errb, covab, a_cis, b_cis)
