Linear regression for data with measurement errors and intrinsic scatter (BCES)
==========

Python module for performing robust linear regression on (X,Y) data points where both X and Y have measurement errors. 

The fitting method is the *bivariate correlated errors and intrinsic scatter* (BCES) and follows the description given in [Akritas & Bershady. 1996, ApJ](http://labs.adsabs.harvard.edu/adsabs/abs/1996ApJ...470..706A/). Some of the advantages of BCES regression compared to ordinary least squares fitting (quoted from Akritas & Bershady 1996):

* it allows for measurement errors on both variables
* it permits the measurement errors for the two variables to be dependent
* it permits the magnitudes of the measurement errors to depend on the measurements
* other "symmetric" lines such as the bisector and the orthogonal regression can be constructed.

In order to understand how to perform and interpret the regression results, please read the paper. 

# Installation

The command line script can be installed via

    python setup.py install

You may need to run the last command with `sudo`.

Install the package with a symlink, so that changes to the source files will be immediately available:

    python setup.py develop




# Usage 

	import bces
	a,b,aerr,berr,covab=bces.bces(x,xerr,y,yerr,cov)

Arguments:

- *x,y* : 1D data arrays
- *xerr,yerr*: measurement errors affecting x and y, 1D arrays
- *cov* : covariance between the measurement errors, 1D array

If you have no reason to believe that your measurement errors are correlated (which is usual the case), you can provide an  array of zeroes as input for *cov*:

    cov = numpy.zeros_like(x)

Output:

- *a,b* : best-fit parameters a,b of the linear regression such that *y = Ax + B*. 
- *aerr,berr* : the standard deviations in a,b
- *covab* : the covariance between a and b (e.g. for plotting confidence bands)

Each element of the arrays *a*, *b*, *aerr*, *berr* and *covab* correspond to the result of one of the different BCES lines: *y|x*, *x|y*, bissector and orthogonal, as detailed in the table below. Please read the [original BCES paper](http://labs.adsabs.harvard.edu/adsabs/abs/1996ApJ...470..706A/) to understand what these different lines mean.


| Element  | Method  |  Description |
|---|---| --- |
| 0  | *y\|x*  | Assumes *x* as the independent variable |
| 1  |  *x\|y* | Assumes *y* as the independent variable |
| 2  | bissector  | Line that bisects the *y\|x* and *x\|y*. This approach is self-inconsistent, *do not use this method*, cf. [Hogg, D. et al. 2010, arXiv:1008.4686](http://labs.adsabs.harvard.edu/adsabs/abs/2010arXiv1008.4686H/). |
| 3  | orthogonal  | Orthogonal least squares: line that minimizes orthogonal distances. Should be used when it is not clear which variable should be treated as the independent one |

## Parallel code

There is a faster, parallel version of the code, *bcesp*, which runs in the same way as bces and is considerably faster in multicore machines.

# Examples of how to use the code

Check out this [jupyter notebook](https://github.com/rsnemmen/BCES/blob/master/misc%20howto%20bces.ipynb). Want do download the notebook and run it locally? [Try this.](https://github.com/takluyver/nbopen)

If you have suggestions of more examples, feel free to add them.

# Requirements

See `requirements.txt`.


# Citation

If you end up using this code in your work and it gets published, you are morally obliged to cite the original BCES paper: [Akritas, M. G., & Bershady, M. A. Astrophysical Journal, 1996, 470, 706](http://labs.adsabs.harvard.edu/adsabs/abs/1996ApJ...470..706A/). I also ask you to cite [Nemmen, R. et al. *Science*, 2012, 338, 1445](http://labs.adsabs.harvard.edu/adsabs/abs/2012Sci...338.1445N/) ([bibtex citation info](http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2012Sci...338.1445N&data_type=BIBTEX&db_key=AST&nocookieset=1)) as one of the examples of application of the BCES method. I spent time writing this code, making sure it works and is user-friendly, so I would appreciate your citation of [my paper](http://labs.adsabs.harvard.edu/adsabs/abs/2012Sci...338.1445N/) as a token of gratitute. Thanks!



# Misc.

This python module is inspired on the (much faster) fortran routine [originally written Akritas et al](http://www.astro.wisc.edu/%7Emab/archive/stats/stats.html). I wrote it because I wanted something more portable and easier to use, trading off speed. 

For a general tutorial on how to (and how not to) perform linear regression, [please read this paper: Hogg, D. et al. 2010, arXiv:1008.4686](http://labs.adsabs.harvard.edu/adsabs/abs/2010arXiv1008.4686H/). In particular, *please refrain from using the bisector method*.

If you want to plot confidence bands for your fits, have a look at [`nmmn` package](https://github.com/rsnemmen/nmmn) (in particular, modules `nmmn.plots.fitconf` and `stats`).


## Bayesian linear regression

There are a couple of Bayesian approaches to perform linear regression which can be more powerful than BCES, some of which are described below.

**A Gibbs Sampler for Multivariate Linear Regression:** 
[R code](https://github.com/abmantz/lrgs), [arXiv:1509.00908](http://arxiv.org/abs/1509.00908).
Linear regression in the fairly general case with errors in X and Y, errors may be correlated, intrinsic scatter. The prior distribution of covariates is modeled by a flexible mixture of Gaussians. This is an extension of the very nice work by Brandon Kelly [(Kelly, B. 2007, ApJ)](http://labs.adsabs.harvard.edu/adsabs/abs/2007ApJ...665.1489K/).

**LIRA: A Bayesian approach to linear regression in astronomy:** [R code](https://github.com/msereno/lira), [arXiv:1509.05778](http://arxiv.org/abs/1509.05778)
Bayesian hierarchical modelling of data with heteroscedastic and possibly correlated measurement errors and intrinsic scatter. The method fully accounts for time evolution. The slope, the normalization, and the intrinsic scatter of the relation can evolve with the redshift. The intrinsic distribution of the independent variable is approximated using a mixture of Gaussian distributions whose means and standard deviations depend on time. The method can address scatter in the measured independent variable (a kind of Eddington bias), selection effects in the response variable (Malmquist bias), and departure from linearity in form of a knee. 

**AstroML: Machine Learning and Data Mining for Astronomy.**
[Python example](http://www.astroml.org/book_figures/chapter8/fig_total_least_squares.html) of a linear fit to data with correlated errors in x and y using AstroML. In the literature, this is often referred to as total least squares or errors-in-variables fitting.




# Todo

If you have improvements to the code, suggestions of examples,speeding up the code etc, feel free to [submit a pull request](https://guides.github.com/activities/contributing-to-open-source/).

* [x] add practical example of using the code with data
* [ ] speed up the code (`numba`? `f2py`?). The big bottleneck is the data bootstrapping
* [ ] implement weighted least squares (WLS)
* [ ] merge with astropy?
* [x] install script

[Visit the author's web page](http://rodrigonemmen.com/) and/or follow him on twitter ([@nemmen](https://twitter.com/nemmen)).


---


Copyright (c) 2018, [Rodrigo Nemmen](http://rodrigonemmen.com).
[All rights reserved](http://opensource.org/licenses/BSD-2-Clause).


Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
