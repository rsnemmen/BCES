BCES and WLS: Linear regression for data with measurement errors and intrinsic scatter
=========================================

Python module for performing robust linear regression on (X,Y) data points with measurement errors.

The **BCES** fitting method is the *bivariate correlated errors and intrinsic scatter* (BCES) and follows the description given in [Akritas & Bershady. 1996, ApJ](http://labs.adsabs.harvard.edu/adsabs/abs/1996ApJ...470..706A/). Some of the advantages of BCES regression compared to ordinary least squares (OLS) fitting:

* it allows for measurement errors on both variables
* it permits the measurement errors for the two variables to be dependent
* it permits the magnitudes of the measurement errors to depend on the measurements
* other "symmetric" lines such as the bisector and the orthogonal regression can be constructed.

The **WLS** (weighted least squares) method handles the case where only Y has measurement errors and X is treated as error-free. It accounts for intrinsic scatter in the data and follows Akritas & Bershady 1996, §2.3. 

## Installation

    pip install bces

Alternatively, if you plan to modify the source then install the package with a symlink, so that changes to the source files will be immediately available:

    pip install -e .







## Usage 

### BCES

	import bces.bces as BCES
	a,b,aerr,berr,covab=BCES.bcesp(x,xerr,y,yerr,cov)

Arguments:

- *x,y* : 1D data arrays
- *xerr,yerr*: measurement errors affecting x and y, 1D arrays
- *cov* : covariance between the measurement errors, 1D array

If you have no reason to believe that your measurement errors are correlated (which is usually the case), you can provide an  array of zeroes as input for *cov*:

    cov = numpy.zeros_like(x)

Output:

- *a,b* : best-fit parameters a,b of the linear regression such that *y = Ax + B*. 
- *aerr,berr* : the standard deviations in a,b
- *covab* : the covariance between a and b (e.g. for plotting confidence bands)

Each element of the arrays `a`, `b`, `aerr`, `berr` and `covab` correspond to the result of one of the different BCES lines: $y|x$, $x|y$, bissector and orthogonal, as detailed in the table below. Please read the [original BCES paper](http://labs.adsabs.harvard.edu/adsabs/abs/1996ApJ...470..706A/) to understand what these different lines mean.


| Element  | Method  |  Description |
|---|---| --- |
| 0  | *y\|x*  | Assumes *x* as the independent variable |
| 1  |  *x\|y* | Assumes *y* as the independent variable |
| 2  | bissector  | Line that bisects the *y\|x* and *x\|y*. This approach is self-inconsistent, [*do not use this method*](https://arxiv.org/abs/1008.4686). |
| 3  | orthogonal  | Orthogonal least squares: line that minimizes orthogonal distances. Should be used when it is not clear which variable should be treated as the independent one |

By default, `bcesp` runs the bootstrapping in parallel.



### WLS

	import bces.bces as BCES
	a,b,aerr,berr,covab=BCES.wls(x,y,yerr)

Arguments:

- *x,y*: 1D data arrays
- *yerr*: measurement errors affecting y, 1D array

Output:

- *a,b*: best-fit slope and intercept of the linear regression such that *y = Ax + B* (scalars)
- *aerr,berr*: the standard deviations in a,b
- *covab*: the covariance between a and b

Note that unlike BCES, WLS returns scalar values (a single regression line) rather than 4-element arrays.

The `wlsp` method performs bootstrapping in parallel, if you need that.

### When to use BCES or WLS?

Both methods return unbiased estimates of the slope and intercept, but they suit different statistical situations:

- **Use BCES** when both X and Y have measurement errors, or when measurement errors on X and Y may be correlated.
- **Use WLS** when only Y has measurement errors (X is error-free or its errors are negligible). 

Both methods account for intrinsic scatter.

**Why choose WLS over OLS?**
When only Y has measurement errors, prefer WLS over OLS. OLS assigns equal weight to every data point regardless of measurement uncertainty, while WLS weights each point by the inverse of its error variance so more precisely measured points have greater influence on the fit. This produces more accurate and statistically efficient estimates when data points have heteroscedastic (unequal) errors.



## Examples

[`bces-examples.ipynb` is a jupyter notebook](https://github.com/rsnemmen/BCES/blob/master/doc/bces-examples.ipynb) including a practical, step-by-step example of how to use BCES to perform regression on data with uncertainties on x and y. It also illustrates how to plot the confidence band for a fit.

[`wls.ipynb` is a jupyter notebook](https://github.com/rsnemmen/BCES/blob/master/doc/wls.ipynb) with examples of WLS regression, including fits with intrinsic scatter.

![](./doc/fit.png)



## Running Tests

```bash
pytest -v -s
```


## Citation

If you end up using this code in your paper, you are morally obliged to cite the following works 

- [Nemmen, R. et al. *Science*, 2012, 338, 1445](http://labs.adsabs.harvard.edu/adsabs/abs/2012Sci...338.1445N/) ([bibtex citation info](http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2012Sci...338.1445N&data_type=BIBTEX&db_key=AST&nocookieset=1)) 
- The original BCES paper: [Akritas, M. G., & Bershady, M. A. Astrophysical Journal, 1996, 470, 706](http://labs.adsabs.harvard.edu/adsabs/abs/1996ApJ...470..706A/) 

I spent considerable time writing this code, making sure it is correct and *user-friendly*, so I would appreciate your citation of the first paper in the above list as a token of gratitude.

[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=&slug=nemmen&button_colour=FFDD00&font_colour=000000&font_family=Cookie&outline_colour=000000&coffee_colour=ffffff)](https://www.buymeacoffee.com/nemmen)

