Other information
==================

The BCES python package is inspired on the much faster Fortran routine which was [originally written by Akritas et al](http://www.astro.wisc.edu/%7Emab/archive/stats/stats.html). I wrote it because I wanted something more portable and easier to use, trading off speed. 

For a general tutorial on how to—and how not to—perform linear regression, [please read this paper: Hogg, D. et al. 2010, arXiv:1008.4686](http://labs.adsabs.harvard.edu/adsabs/abs/2010arXiv1008.4686H/). In particular, *please refrain from using the bisector method*.

If you want to plot confidence bands for your fits, have a look at [`nmmn` package](https://github.com/rsnemmen/nmmn) (in particular, modules `nmmn.plots.fitconf` and `stats`).


# Bayesian linear regression

There are a couple of Bayesian approaches to perform linear regression which can be more powerful than BCES, some of which are described below. The advantage of these methods is that they give you posterior distributions on the parameters. Also, some of them handle censored data (i.e. upper limits) while BCES does not.

**A Gibbs Sampler for Multivariate Linear Regression:** 
[R code](https://github.com/abmantz/lrgs), [arXiv:1509.00908](http://arxiv.org/abs/1509.00908).
Linear regression in the fairly general case with errors in X and Y, errors may be correlated, intrinsic scatter. The prior distribution of covariates is modeled by a flexible mixture of Gaussians. This is an extension of the very nice work by Brandon Kelly [(Kelly, B. 2007, ApJ)](http://labs.adsabs.harvard.edu/adsabs/abs/2007ApJ...665.1489K/).

**LIRA: A Bayesian approach to linear regression in astronomy:** [R code](https://github.com/msereno/lira), [arXiv:1509.05778](http://arxiv.org/abs/1509.05778)
Bayesian hierarchical modelling of data with heteroscedastic and possibly correlated measurement errors and intrinsic scatter. The method fully accounts for time evolution. The slope, the normalization, and the intrinsic scatter of the relation can evolve with the redshift. The intrinsic distribution of the independent variable is approximated using a mixture of Gaussian distributions whose means and standard deviations depend on time. The method can address scatter in the measured independent variable (a kind of Eddington bias), selection effects in the response variable (Malmquist bias), and departure from linearity in form of a knee. 

**AstroML: Machine Learning and Data Mining for Astronomy.**
[Python example](http://www.astroml.org/book_figures/chapter8/fig_total_least_squares.html) of a linear fit to data with correlated errors in x and y using AstroML. In the literature, this is often referred to as total least squares or errors-in-variables fitting.

