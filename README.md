Linear regression for data with measurement errors and intrinsic scatter
==========

Python module for performing robust linear regression on (X,Y) data points where both X and Y have measurement errors. 

The fitting method is the *bivariate correlated errors and intrinsic scatter* (BCES) and follows the description given in [Akritas, M. G., & Bershady, M. A. Astrophysical Journal, 1996, 470, 706](http://labs.adsabs.harvard.edu/adsabs/abs/1996ApJ...470..706A/). Some of the advantages of BCES regression compared to ordinary least squares fitting (quoted from Akritas & Bershady 1996):

* it allows for measurement errors on both variables
* it permits the measurement errors for the two variables to be dependent
* it permits the magnitudes of the measurement errors to depend on the measurements
* other "symmetric" lines such as the bisector and the orthogonal regression can be constructed.

In order to understand how to perform and interpret the regression results, I suggest you to read the paper. 

## Usage 

	import bces
	a,b,aerr,berr,covab=bces.bces(x,xerr,y,yerr,cov)

Output:

- *a,b* : best-fit parameters a,b of the linear regression 
- *aerr,berr* : the standard deviations in a,b
- *covab* : the covariance between a and b (e.g. for plotting confidence bands)

Arguments:

- *x,y* : data
- *xerr,yerr*: measurement errors affecting x and y
- *cov* : covariance between the measurement errors
(all are arrays)


## Requirements

Numpy, Scipy, [fish](https://pypi.python.org/pypi/fish/). 

## More

This python module is inspired on the (much faster) fortran routine [originally written Akritas et al](http://www.astro.wisc.edu/%7Emab/archive/stats/stats.html). I wrote it because I wanted something more portable and easier to use, trading off speed. If you speed this module up, please submit a pull request. :)

If you have suggestions of improvements, by all means please contribute! If you end up using this code in your work and it gets published (yay), you could cite one of my papers which made use of BCES fitting as an example of a nice application of the method: [Nemmen, R. et al. *Science*, 2012, 338, 1445](http://labs.adsabs.harvard.edu/adsabs/abs/2012Sci...338.1445N/). :)

For a general tutorial on how to (and how not to) perform linear regression, [please read this paper: Hogg, D. et al. 2010, arXiv:1008.4686](http://labs.adsabs.harvard.edu/adsabs/abs/2010arXiv1008.4686H/). In particular, *please refrain from using the bisector method*.

For the Bayesian way of performing linear regression similar to BCES (and even more powerful), I suggest having a look at [Kelly, B. 2007, ApJ, 665, 1489](http://labs.adsabs.harvard.edu/adsabs/abs/2007ApJ...665.1489K/).


## Todo

* implement weighted least squares (WLS)
* port the [bayesian regression code linminmax from IDL](https://github.com/wlandsman/IDLAstro/blob/master/pro/math/linmix_err.pro) to python

[Visit the author's web page](http://www.astro.iag.usp.br/~nemmen/) and/or follow him on twitter ([@astrorho](https://twitter.com/astrorho)).

---

Copyright (c) 2012, [Rodrigo Nemmen](http://asd.gsfc.nasa.gov/Rodrigo.Nemmen/Rodrigo_Nemmens_Homepage/Home.html).
[All rights reserved](http://opensource.org/licenses/BSD-2-Clause).

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.