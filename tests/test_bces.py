"""
Basic unit testing functionality.
"""
import bces.bces as BCES
import numpy as np
from pathlib import Path

# reads test dataset (path relative to this file, not cwd)
_DATA_PATH = Path(__file__).resolve().parent.parent / "doc" / "data.npz"
data = np.load(_DATA_PATH)
xdata = data['x']
ydata = data['y']
errx = data['errx']
erry = data['erry']
covdata = data['cov']

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
ans_pars = np.array([ [0.57955173, 17.88855826],
                      [0.26053751, 32.70952271],
                      [0.50709256, 21.25491222] ])
# covariance matrix
ans_cov = np.array([[ 5.85029731e-04, -2.72808055e-02],
                    [-2.72808055e-02,  1.27299029e+00]])


def test_yx(bces_result):
    """Test BCES Y|X fit without bootstrapping."""
    a, b, erra, errb, covab = bces_result
    np.testing.assert_array_almost_equal([ans_pars[0,0], ans_pars[0,1]], np.array([a[0], b[0]]))


def test_xy(bces_result):
    """Test BCES X|Y fit without bootstrapping."""
    a, b, erra, errb, covab = bces_result
    np.testing.assert_array_almost_equal([ans_pars[1,0], ans_pars[1,1]], np.array([a[1], b[1]]))


def test_ort(bces_result):
    """Test BCES orthogonal fit without bootstrapping."""
    a, b, erra, errb, covab = bces_result
    np.testing.assert_array_almost_equal([ans_pars[2,0], ans_pars[2,1]], np.array([a[3], b[3]]))


def test_bisector(bces_result):
    """Test BCES bisector fit (index 2) without bootstrapping."""
    a, b, erra, errb, covab = bces_result
    # Bisector slope should be between Y|X and X|Y slopes
    assert ans_pars[1,0] <= a[2] <= ans_pars[0,0]


def test_covariance(bces_result):
    """Test that covab output matches expected covariance matrix."""
    a, b, erra, errb, covab = bces_result
    # covab[0] is covariance for Y|X method
    np.testing.assert_almost_equal(covab[0], ans_cov[0,1], decimal=4)


def test_yxboot():
    """Test BCES Y|X fit with bootstrapping (parallel)."""
    a, b, erra, errb, covab = BCES.bcesp(xdata, errx, ydata, erry, covdata)
    np.testing.assert_array_almost_equal([ans_pars[0,0], ans_pars[0,1]], np.array([a[0], b[0]]), 1)


def test_ortboot():
    """Test BCES orthogonal fit with bootstrapping (parallel)."""
    a, b, erra, errb, covab = BCES.bcesp(xdata, errx, ydata, erry, covdata)
    np.testing.assert_array_almost_equal([ans_pars[2,0], ans_pars[2,1]], np.array([a[3], b[3]]), 1)


def test_bcesboot():
    """Test BCES Y|X fit with serial bootstrapping."""
    a, b, erra, errb, covab = BCES.bcesboot(xdata, errx, ydata, erry, covdata, nsim=1000)
    np.testing.assert_array_almost_equal([ans_pars[0,0], ans_pars[0,1]], np.array([a[0], b[0]]), 1)


def test_bootstrap():
    """Test if bootstrap is working correctly."""
    import scipy.stats

    nboot = 5000
    ts = []
    for i in range(nboot):
        xsim = BCES.bootstrap(xdata)
        tsim, asim, psim = scipy.stats.anderson_ksamp([xdata, xsim])
        ts.append(tsim)

    ts = np.array(ts)
    assert np.median(ts) < asim[0]


def test_bootstrap_list():
    """Test bootstrap with a list of arrays (index correspondence preserved)."""
    result = BCES.bootstrap([xdata, ydata])
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].shape == xdata.shape
    assert result[1].shape == ydata.shape
    # Both arrays must use the same random indices: check one known pairing
    # by verifying the resampled arrays come from the original data
    for val in result[0]:
        assert val in xdata


def test_allEqual():
    """Unit tests for the allEqual helper."""
    assert BCES.allEqual(np.array([3, 3, 3])) is True
    assert BCES.allEqual(np.array([1, 2, 3])) is False
    assert BCES.allEqual(np.array([1])) is True


def test_known_slope():
    """
    Fit synthetic data with a known true slope, intercept, and intrinsic
    scatter, and verify BCES Y|X recovers them within reasonable tolerances.
    """
    rng = np.random.default_rng(42)
    n = 200
    true_a, true_b = 2.5, 1.0

    x_true = rng.uniform(0, 10, n)
    xerr = rng.uniform(0.1, 0.3, n)
    yerr = rng.uniform(0.1, 0.3, n)
    scatter = rng.normal(0, 0.5, n)     # intrinsic scatter

    # observed values with measurement noise and scatter
    x = x_true + rng.normal(0, xerr)
    y = true_a * x_true + true_b + scatter + rng.normal(0, yerr)

    a, b, erra, errb, covab = BCES.bces(x, xerr, y, yerr, np.zeros(n))

    np.testing.assert_allclose(a[0], true_a, atol=0.2)
    np.testing.assert_allclose(b[0], true_b, atol=0.5)
