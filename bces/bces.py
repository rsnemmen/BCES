
import numpy as np
import scipy
from scipy.stats import norm

#import astropy functions for use in bootstrap (optional)
try:
    import astropy
    from astropy.stats import biweight_location
except ImportError:
    astropy = None


# BCES fitting
# ===============

def bces(y1, y1err, y2, y2err, cerr=None):
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

Updated by Evan Groopman, US NRL, 20200425
"""
    
    if cerr is None:
        cerr = np.zeros_like(y1)
    
    # Arrays holding the code main results for each method:
    # Elements: 0-Y|X, 1-X|Y, 2-bisector, 3-orthogonal
    a, b, avar, bvar, covarxiz = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)
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
    # for i in range(4):
    b = y2.mean() - a*y1.mean()
	
    # Set up variables to calculate standard deviations of slope/intercept 
    xi.append(	( (y1-y1.mean()) * (y2-a[0]*y1-b[0]) + a[0]*y1err**2 ) / (y1.var()-sig11var)	)	# Y|X
    xi.append(	( (y2-y2.mean()) * (y2-a[1]*y1-b[1]) - y2err**2 ) / covar_y1y2	)	# X|Y
    xi.append(	xi[0] * (1.+a[1]**2)*a[2] / ((a[0]+a[1])*np.sqrt((1.+a[0]**2)*(1.+a[1]**2))) + xi[1] * (1.+a[0]**2)*a[2] / ((a[0]+a[1])*np.sqrt((1.+a[0]**2)*(1.+a[1]**2)))	)	# bisector
    xi.append(	xi[0] * a[3]/(a[0]**2*np.sqrt(4.+(a[1]-1./a[0])**2)) + xi[1]*a[3]/np.sqrt(4.+(a[1]-1./a[0])**2)	)	# orthogonal
    for i in range(4):
        zeta.append( y2 - a[i]*y1 - y1.mean()*xi[i]	)

    for i in range(4):
        # Calculate variance for all a and b
        avar[i] = xi[i].var()/xi[i].size
        bvar[i] = zeta[i].var()/zeta[i].size
		
        # Sample covariance obtained from xi and zeta (paragraph after equation 15 in AB96)
        covarxiz[i] = np.mean( (xi[i]-xi[i].mean()) * (zeta[i]-zeta[i].mean()) )

    # Covariance between a and b (equation after eq. 15 in AB96)
    covar_ab = covarxiz/y1.size
	
    return(a, b, np.sqrt(avar), np.sqrt(bvar), covar_ab)
	
	
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

Rodrigo Nemmen, http://goo.gl/8S1Oo
	"""
	if type(v)==list:
		vboot=[]	# list of boostrapped arrays
		n=v[0].size
		iran=scipy.random.randint(0,n,n)	# Array of random indexes
		for x in v:	vboot.append(x[iran])
	else:	# if v is an array, not a list of arrays
		n=v.size
		iran=scipy.random.randint(0,n,n)	# Array of random indexes
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

v1 Mar 2012: ported from bces_regress.f. Added covariance output.
Rodrigo Nemmen, http://goo.gl/8S1Oo
"""
    import fish
 	
    # Progress bar initialization
    peixe = fish.ProgressFish(total=nsim)
    print("Bootstrapping progress:")
	
    """
    My convention for storing the results of the bces code below as 
	matrixes for processing later are as follow:
	
	      simulation\method  y|x x|y bisector orthogonal
	          sim0           ...
	Am =      sim1           ...
	          sim2           ...
	          sim3           ...
"""

    for i in range(nsim): 
        [y1sim,y1errsim,y2sim,y2errsim,cerrsim]=bootstrap([y1,y1err,y2,y2err,cerr])
		
        asim,bsim,errasim,errbsim,covabsim=bces(y1sim,y1errsim,y2sim,y2errsim,cerrsim)	
		
        if i==0:
            # Initialize the matrixes
            am,bm=asim.copy(),bsim.copy()
        else: 
            am=np.vstack((am,asim))
            bm=np.vstack((bm,bsim))
				
        # Progress bar
        peixe.animate(amount=i)
	
    # Bootstrapping results
    a=np.array([ am[:,0].mean(), am[:,1].mean(), am[:,2].mean(), am[:,3].mean() ])
    b=np.array([ bm[:,0].mean(), bm[:,1].mean(), bm[:,2].mean(), bm[:,3].mean() ])

	# Error from unbiased sample variances
    erra, errb, covab = np.zeros(4), np.zeros(4), np.zeros(4)
    for i in range(4):
        erra[i]=np.sqrt( 1./(nsim-1) * ( np.sum(am[:,i]**2)-nsim*(am[:,i].mean())**2 ))
        errb[i]=np.sqrt( 1./(nsim-1) * ( np.sum(bm[:,i]**2)-nsim*(bm[:,i].mean())**2 ))
        
        covab[i]=1./(nsim-1) * ( np.sum(am[:,i]*bm[:,i])-nsim*am[:,i].mean()*bm[:,i].mean() )
	
    return a,b,erra,errb,covab

# BCES bootstrap fitting
# faster using vectorization
# ===============

def bootstrap_indexes(data_length, n_samples=10000, weights= None):
    """
Given data points data, where axis 0 is considered to delineate points, return
an array where each row is a set of bootstrap indexes. This can be used as a list
of bootstrap indexes as well.
    """
    d0 = np.int64(data_length)
    if weights is not None:
        assert(len(weights) == d0)
    n = np.int64(n_samples)
    arr = np.arange(d0)
    #Note: numba does not support p= weights argument of np.random.choice
    #but, this is pretty fast already
    return(np.random.choice(arr, size=(n, d0), replace=True, p=weights))

def jackknife_indexes(data_length):
    """
Given data points data, where axis 0 is considered to delineate points, return
a list of arrays where each array is a set of jackknife indexes.

For a given set of data Y, the jackknife sample J[i] is defined as the data set
Y with the ith data point deleted.
    """
    base = np.arange(0, data_length)
    return(np.vstack([np.delete(base, i) for i in base]))


def normalize(array, axis=None):
    """If using weights in bootstrap resample,
    numpy.random.choice requires probabilities to sum to 1.0."""
    return(array/np.sum(array, axis=axis))


def bces_bootstrap(y1, y1err, y2, y2err, cerr=None, n_samples=10000, 
              weights=None, bias_correct= False, symmetric_errors= True,
              statistic='mean', smooth= False):
    """
    Does the BCES with bootstrap resampling. 
    Original version by Rodrigo Nemmen, http://goo.gl/8S1Oo.
    Fast, vectorized version with (optional) BCA bias correction by Evan Groopman. 05/13/2020
    (evan.groopman@nrl.navy.mil, eegroopm@gmail.com)
    For BCES method see: Akritas & Bershady (1996) The Astrophysical Journal 470, 706
    For BCA method see: Efron, An Introduction to the Bootstrap. Chapman & Hall 1993
    
    Example runtimes: 8-core Intel i7-7700HQ @ 2.8Ghz
    example runtime, len(data) = 50:    %timeit bces_bootstrap(y1, y1err, y2, y2err, n_samples=10000, bias_correct=True) 
                                        50.5 ms ± 619 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    example runtime, len(data) = 100:   %timeit bces_bootstrap(y1, y1err, y2, y2err, n_samples=10000, bias_correct=True)
                                        91.4 ms ± 1.12 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    example runtime, len(data) = 100:   %timeit bces_bootstrap(y1, y1err, y2, y2err, n_samples=100000, bias_correct=True)
                                        937 ms ± 6.37 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    example runtime, len(data) = 1000:  %timeit bces_bootstrap(y1, y1err, y2, y2err, n_samples=10000, bias_correct=True)
                                        889 ms ± 9.08 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
                                    
    Parameters
    ----------
    y1 : array, 
            "x data".
    y1err : array, 
            "x errors".
    y2 : array, 
            "y data".
    y2err : array, 
            "y errros".
    cerr : array, optional
        correlated error (rho) values. The default is None, which generates an array of 0s
    n_samples : int, optional
        number of bootstrap samples to draw. The default is 10000.
    weights : array, optional
        weights for biasing the bootstrap resampling (probabilities for redrawing each data point).
        if the sum of weights does not equal 1, the weights are normalized.
        The default is None.
    bias_correct : bool, optional
        calculate the bias-corrected and accelerated (BCA) confidence intervals for the 
        mean slope and intercept values. The default is False.
    symmetric_errors : bool, optional
        Only used if bias_correct=True. Averages the potentially asymmetric error
        values to return a single uncertainty for the a and b parameters.
        This makes the outputs compatible with the standard bces function.
        The default is True (but be cautious).
    statistic : str, optional
        Choose the method to calculate values of a and b from their bootstrap distribution.
        Values can be 'mean', 'biweight', 'median'. 
        'mean' (default) takes the arithmetic mean using numpy. This can be influenced heavily by outliers.
        'biweight' uses the astropy.stats.biweight_location function to calculate the robust
        mean statistic.
        'median' calculates the distribution center using the median. However, this
        can result in a sparse discretized distribution, which may require smoothing.
    smooth : bool, optional
        Adds a small amount of random noise to each sample (N(0, sigma**2) where sigma = 1/sqrt(n_samples)).
        This prevents an overly discretized statistic distribution for better
        maximum-likelihood estimation, especially if using the median statistic.

    Returns
    -------
    a: (1x4) array of slopes for the different regressions: Y|X, X|Y, bisector, orthogonal
    b: (1x4) array of intercepts
    aerr: (1x4) or (2x4) array of slope uncertainties. (1x4) array for no bias_correction=False and/or symmetric_errors=True.
          (2x4) for bias_correction=True and symmestric_errors=False
    berr: (1x4) or (2x4) array of intercept uncertainties, as with aerr.
    covab: (1x4) array of a,b covariances for each regression method.
    """

    ######################
        
    if cerr is None:
        cerr = np.zeros_like(y1)
        
    if type(weights) in [list, np.ndarray]:
            weights = normalize(weights, axis=0) #change this if shape of input is different than 1
    
    boot_ind = bootstrap_indexes(len(y1), n_samples, weights=weights)
    y1sim = y1[boot_ind]
    y1errsim = y1err[boot_ind]
    y2sim = y2[boot_ind]
    y2errsim = y2err[boot_ind]
    cerrsim = cerr[boot_ind]
    
    # arrays are now resampled (n_samples x data_length) arrays, so calculate
    # bces fit parameters by axis=1, then take mean of remaining n_samples
    # then (optional) bias correction
    
    # Arrays holding the code main results for each method:
    # Elements: 0-Y|X, 1-X|Y, 2-bisector, 3-orthogonal
    a = np.empty((n_samples, 4))
    b = np.empty_like(a)
    
    #don't need these parameters if we're taking the variance of the bootstrapped a & b distributions
    #the calculations below have been vectorized, though. if confidence intervals for each
    #parameter are required.
    
    # avar = np.empty_like(a)
    # bvar = np.empty_like(a)
    # covarxiz = np.empty_like(a)
    
    # xi = np.empty((4, n_samples, len(y1)))
    # zeta = np.empty_like(xi)
	
    # Calculate sigma's for datapoints using length of conf. intervals
    sig11var = np.mean(y1errsim**2, axis=1)
    sig22var = np.mean(y2errsim**2, axis=1)
    sig12var = np.mean(cerrsim, axis=1)
	
    # Covariance of Y1 (X) and Y2 (Y)
    covar_y1y2 = np.mean((y1sim - y1sim.mean(axis=1)[np.newaxis].T)*(y2sim - y2sim.mean(axis=1)[np.newaxis].T), axis=1)

    # Compute the regression slopes
    a[:,0] = (covar_y1y2 - sig12var)/(y1sim.var(axis=1) - sig11var)	# Y|X
    a[:,1] = (y2sim.var(axis=1) - sig22var)/(covar_y1y2 - sig12var)	# X|Y
    a[:,2] = ( a[:,0]*a[:,1] - 1.0 + np.sqrt((1.0 + a[:,0,]**2)*(1.0 + a[:,1,]**2)) ) / (a[:,0] + a[:,1])	# bisector
    sign = np.ones_like(covar_y1y2, dtype=float)
    sign[covar_y1y2 < 0] = -1.0

    a[:,3] = 0.5*((a[:,1,] - (1./a[:,0,])) + sign*np.sqrt(4. + (a[:,1] - (1./a[:,0]))**2))	# orthogonal
	
    # Compute intercepts
    b = (y2sim.mean(axis=1) - a.T*y1sim.mean(axis=1)).T
	
    # # Set up variables to calculate standard deviations of slope/intercept 
    # # this may not be necessary since we're using the variance of the simulated distributions of a and b
    
    # xi[0,:,:] = ( (y1sim - y1sim.mean(axis=1)[np.newaxis].T) * (y2sim - a[:,0][np.newaxis].T*y1sim - b[:,0][np.newaxis].T) + a[:,0][np.newaxis].T*y1errsim**2 ) / ((y1sim.var(axis=1) - sig11var)[np.newaxis].T)	# Y|X
    # xi[1,:,:] = ( (y2sim - y2sim.mean(axis=1)[np.newaxis].T) * (y2sim - a[:,1][np.newaxis].T*y1sim - b[:,1][np.newaxis].T) - y2errsim**2 ) / (covar_y1y2[np.newaxis].T)		# X|Y
    # #not yet here
    # xi[2,:,:] = xi[0,:,:] * (1. + a[:,1][np.newaxis].T**2) * a[:,2][np.newaxis].T / ((a[:,0][np.newaxis].T + a[:,1][np.newaxis].T) * np.sqrt((1. + a[:,0][np.newaxis].T**2)*(1. + a[:,1][np.newaxis].T**2))) + xi[1,:,:] * (1. + a[:,0][np.newaxis].T**2)*a[:,2][np.newaxis].T / ((a[:,0][np.newaxis].T + a[:,1][np.newaxis].T)*np.sqrt((1. + a[:,0][np.newaxis].T**2)*(1. + a[:,1][np.newaxis].T**2)))		# bisector
    # xi[3,:,:] = xi[0,:,:] * a[:,3][np.newaxis].T/(a[:,0][np.newaxis].T**2 * np.sqrt(4. + (a[:,1][np.newaxis].T - 1./a[:,0][np.newaxis].T)**2)) + xi[1,:,:] * a[:,3][np.newaxis].T/np.sqrt(4. + (a[:,1][np.newaxis].T - 1./a[:,0][np.newaxis].T)**2)		# orthogonal
    
    # for i in range(4):
    #     zeta[i,:,:] = y2sim - a[:,i][np.newaxis].T*y1sim - y1sim.mean(axis=1)[np.newaxis].T*xi[i,:,:]
    #     #calculate variance for a and b
    #     avar[:,i] = xi[i,:,:].var(axis=1)/y1.size
    #     bvar[:,i] = zeta[i,:,:].var(axis=1)/y1.size
    #     # Sample covariance obtained from xi and zeta (paragraph after equation 15 in AB96)
    #     covarxiz[:,i] = np.mean( (xi[i,:,:] - xi[i,:,:].mean(axis=1)[np.newaxis].T) * (zeta[i,:,:] - zeta[i,:,:].mean(axis=1)[np.newaxis].T), axis=1)

    # # Covariance between a and b (equation after eq. 15 in AB96)
    # covar_ab = covarxiz/y1.size

    if smooth:
        #if smoothing, add a small amount of random noise to the bootstrap samples
        #loc specifies the mean of the distribution, scale specifies the standard deviation
        a += norm.rvs(loc=0, scale = 1/np.sqrt(n_samples), size= (n_samples, 4))
        b += norm.rvs(loc=0, scale = 1/np.sqrt(n_samples), size= (n_samples, 4))

    #reduce bootstrapped results by taking the mean along the n_samples axis
    if statistic not in ['biweight', 'median']: #aka, is mean or any other value
        aval = a.mean(axis=0)
        bval = b.mean(axis=0)
    elif statistic == 'biweight' and astropy is not None:
        aval = biweight_location(a, axis=0)
        bval = biweight_location(b, axis=0)
    else:
        aval = np.median(a, axis=0)
        bval = np.median(b, axis=0)
    
    if bias_correct:
        #bca bias correction on the uncertainties on a & b (aerr, berr)
        #this seems to result in tighter uncertainties than the original calculation
        
        # confidence interval levels
        alpha = 0.67 #1-sigma by default
        alphas = np.array([1-alpha, alpha])
        
        #need to sort each set of a and b values. sort along the n_samples dimension (axis=0)
        a.sort(axis=0)
        b.sort(axis=0)
    
        # The bias correction value.
        z0a = norm.ppf( ( 1.0*np.sum(a < aval, axis=0)  ) / n_samples )
        z0b = norm.ppf( ( 1.0*np.sum(b < bval, axis=0)  ) / n_samples )
    
        # Statistics of the jackknife distribution
        jackindexes = jackknife_indexes(len(y1))
        
        #jacknife statistic on each set of data, y1 & y2
        jstat = np.column_stack((y1[jackindexes].mean(axis=1), y2[jackindexes].mean(axis=1)))
        jmean = np.mean(jstat, axis=0)
    
        # Acceleration value
        accel = np.sum( (jmean - jstat)**3, axis=0 ) / ( 6.0 * np.sum( (jmean - jstat)**2, axis=0)**1.5 )
    
        zsa = z0a + norm.ppf(alphas).reshape(alphas.shape+(1,)*z0a.ndim)
        zsb = z0b + norm.ppf(alphas).reshape(alphas.shape+(1,)*z0b.ndim)
    
        dist_cutoffs_a = norm.cdf(z0a + zsa/(1 - accel[np.newaxis].T*zsa))
        dist_cutoffs_b = norm.cdf(z0b + zsb/(1 - accel[np.newaxis].T*zsb))
        
        #calculate the indices for the alpha-level values in the statistic distributions (a & b)
        nvals_a = np.round((n_samples - 1) * dist_cutoffs_a).astype(int)
        nvals_b = np.round((n_samples - 1) * dist_cutoffs_b).astype(int)
        
        #initialize error arrays
        erra = np.empty((2,4))
        errb = np.empty_like(erra)
        cia = np.empty_like(erra)
        cib = np.empty_like(erra)
        
        #pull out real values of upper and lower confidence intervals (may be asymmetric)
        for i in range(4):
            cia[:,i] = a[:,i][nvals_a[:,i]]
            cib[:,i] = b[:,i][nvals_b[:,i]]
        
        #calculate the errors by removing the mean value. top row = -error, bottom row = +error
        # Note: this returns values from both sides of the distribution, which
        # may not be symmetric.
        # if symmetric_errors = True, average the error values to make (1x4) arrays
        erra = np.abs(aval - cia)
        errb = np.abs(bval - cib)
        if symmetric_errors:
            erra = erra.mean(axis=0)
            errb = errb.mean(axis=0)
        
    else:
        #no bias correction.
        erra = np.sqrt( 1./(n_samples-1) * ( np.sum(a**2, axis=0) - n_samples*(aval)**2 ))
        errb = np.sqrt( 1./(n_samples-1) * ( np.sum(b**2, axis=0) - n_samples*(bval)**2 ))
        
    covab = 1./(n_samples-1) * ( np.sum(a*b, axis=0) - n_samples * aval * bval )
    
    return(aval, bval, erra, errb, covab)



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
        [y1sim,y1errsim,y2sim,y2errsim,cerrsim]=bootstrap([y1,y1err,y2,y2err,cerr])
        
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

.. codeauthor: Rodrigo Nemmen, http://goo.gl/8S1Oo
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
