# BCES Repository

## Commands

```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/

# Run a specific test
pytest tests/test_bces.py::test_yx
```

## Architecture

Single-file library: all regression logic lives in `bces/bces.py`. The package `__init__.py` exposes `__version__`, `__all__`, and a `__getattr__` lazy-import mechanism for the public API. `__all__` and `__getattr__` cover: `bces`, `bcesboot`, `bcesp`, `bootstrap`, `wls`, `wlsboot`, `wlsp`.

### Key functions in `bces/bces.py`

**BCES regression:**
- `bces(y1, y1err, y2, y2err, cerr)` — core BCES regression, returns 4 slopes simultaneously: Y|X, X|Y, bisector, orthogonal. Output arrays are indexed `[0,1,2,3]` in that order.
- `bcesboot(...)` — BCES with bootstrapping (serial, uses `tqdm`).
- `bcesp(...)` — parallel bootstrap using `multiprocessing`, auto-detects CPU count.

**WLS regression** (use when X is measured without error):
- `wls(x, y, yerr)` — Weighted Least Squares, implements Section 2.3 of Akritas & Bershady (1996).
- `wlsboot(x, y, yerr, nsim)` — serial bootstrap WLS (uses `tqdm`).
- `wlsp(x, y, yerr, nsim)` — parallel bootstrap WLS using `multiprocessing`.

**Helpers:**
- `bootstrap(v)` — accepts a single array or a list of arrays (preserves index correspondence).
- `_bootstrap_results(am, bm, nsim)` — computes mean, std, and covariance from bootstrap slope/intercept matrices; shared by `bcesboot` and `bcesp`.
- `checkNan(am, bm)` — removes NaN rows from bootstrap result matrices (degenerate samples).
- `allEqual(x)` — returns True if all elements of an array are equal; used to skip degenerate bootstrap samples.

### Usage pattern

```python
import bces.bces as BCES

# BCES (both X and Y have measurement errors)
a, b, aerr, berr, covab = BCES.bces(x, xerr, y, yerr, cov)
# a[0], b[0] → Y|X slope and intercept
# a[3], b[3] → orthogonal slope and intercept

# With bootstrapped errors (parallel):
a, b, aerr, berr, covab = BCES.bcesp(x, xerr, y, yerr, cov, nsim=10000)

# WLS (X is error-free)
a, b, aerr, berr, covab = BCES.wls(x, y, yerr)

# WLS with bootstrapped errors (parallel):
a, b, aerr, berr, covab = BCES.wlsp(x, y, yerr, nsim=10000)
```

`cov` is the covariance between measurement errors on x and y (set to zeros array if uncorrelated).

## Dependencies

- `numpy`, `scipy` — core computation
- `tqdm` — progress bar in `bcesboot` and `wlsboot`

## Testing notes

- Tests use `Path(__file__).resolve()` to locate `doc/data.npz`, so `pytest` can be run from any directory.
- `conftest.py` provides session-scoped fixtures `test_data` and `bces_result` used by most BCES tests.
- `test_yxboot`, `test_ortboot`, and `test_wlsboot` use parallel bootstrap and check results to 1 decimal place only — these are stochastic and slower.
- The 4-element output arrays use index `3` for orthogonal (not `2`); the bisector is index `2`.
- Full test list: `test_yx`, `test_xy`, `test_ort`, `test_bisector`, `test_covariance`, `test_yxboot`, `test_ortboot`, `test_bcesboot`, `test_bootstrap`, `test_bootstrap_list`, `test_allEqual`, `test_known_slope`, `test_wls_known_slope`, `test_wls_vs_ols`, `test_wlsboot`.
