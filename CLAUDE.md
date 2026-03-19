# BCES Repository

## Commands

```bash
# Install in development mode
pip install -e .

# Run tests (must be run from repo root — tests load doc/data.npz by relative path)
pytest tests/

# Run a specific test
pytest tests/test_bces.py::test_yx
```

## Architecture

Single-file library: all regression logic lives in `bces/bces.py`. The package `__init__.py` is empty (just a docstring).

### Key functions in `bces/bces.py`

- `bces(y1, y1err, y2, y2err, cerr)` — core BCES regression, returns 4 slopes simultaneously: Y|X, X|Y, bisector, orthogonal. Output arrays are indexed `[0,1,2,3]` in that order.
- `bcesboot(...)` — BCES with bootstrapping (serial, uses `tqdm`).
- `bcesp(...)` — parallel bootstrap using `multiprocessing`, auto-detects CPU count.
- `bootstrap(v)` — helper; accepts a single array or a list of arrays (preserves index correspondence).

### Usage pattern

```python
import bces.bces as BCES

a, b, aerr, berr, covab = BCES.bces(x, xerr, y, yerr, cov)
# a[0], b[0] → Y|X slope and intercept
# a[3], b[3] → orthogonal slope and intercept

# With bootstrapped errors (parallel):
a, b, aerr, berr, covab = BCES.bcesp(x, xerr, y, yerr, cov, nsim=10000)
```

`cov` is the covariance between measurement errors on x and y (set to zeros array if uncorrelated).

## Dependencies

- `numpy`, `scipy` — core computation
- `tqdm` — progress bar in `bcesboot`
- `nmmn` — only used inside `checkNan` (optional, only triggered when NaNs appear in bootstrap results for very small datasets)

## Testing notes

- Tests use a fixed dataset at `doc/data.npz` (loaded by relative path), so `pytest` must be run from the repo root.
- `test_yxboot` and `test_ortboot` use `bcesp` (parallel bootstrap) and check results to 1 decimal place only — these are stochastic and slower.
- The 4-element output arrays use index `3` for orthogonal (not `2`); the bisector is index `2` but has no dedicated test.
