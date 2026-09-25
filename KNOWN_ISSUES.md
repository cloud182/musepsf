# Known issues — tracked separately, not fixed during the structural refactor

This list is intentionally kept apart from `REFACTORING_PLAN.md`. The refactor in progress is
scoped to be behavior-preserving; every bug below changes observable output or crashes in some
case, so none of them are touched until we decide to tackle this list on its own.

Status legend: 🔴 open (not started). ✅ fixed.

## 1. ✅ `Image.resample(pixscale=...)` ignores the requested pixel scale

**Where:** `src/musepsf/image.py:176-190`

`newstep=0.2` is hardcoded in the call to MPDAF's `image.resample(...)`, regardless of the
`pixscale` argument passed in. The target shape (`newdim_y`/`newdim_x`) is computed from
`pixscale` correctly, but the actual resampling step size is not — so calling
`resample(pixscale=X)` for any `X != 0.2` produces a geometrically inconsistent result (wrong
plate scale baked into the resampled data).

**Fix applied:** pass `newstep=pixscale` instead of the literal `0.2`. Regression test:
`tests/test_image.py::test_resample_to_pixscale_uses_requested_scale`.

## 2. ✅ `Image.__init__` leaves `self.units` undefined if `BUNIT` is missing

**Where:** `src/musepsf/image.py:115-122`

```python
try:
    self.units = u.Unit(self.header['BUNIT'])
    self.units *= u.pix
except KeyError:
    print('The card BUNIT does not exist.')
```

The `except` branch only prints a message; it never sets `self.units`. Any later code path that
reads `self.units` (e.g. `measure_psf`'s unit-equality assert) raises `AttributeError` instead of
a clear error about the missing header card.

**Fix applied:** raise a clear `ValueError` pointing at the missing `BUNIT` card (chained with
`from None` to hide the internal `KeyError`). Regression test:
`tests/test_image.py::TestInit::test_raises_value_error_when_bunit_missing_and_no_units_given`.

## 3. ✅ `utils.bin_image()` mutates the caller's array in place

**Where:** `src/musepsf/utils.py:129-130`

```python
if zeromask is not None:
    image[zeromask] = 0
```

`bin_image` writes into the array it was given rather than a copy. Since
`MUSEImage.measure_psf()` calls `check_flux_calibration(self.data, ...)`, which in turn calls
`bin_image(muse, ...)` with `muse` aliased to `self.data`, `self.data` gets zeroed at the
`zeromask` positions as a side effect of a binning/statistics helper — before the "official"
reassignment (`self.data = check_flux_calibration(...)`) even happens. Surprising action-at-a-
distance for anyone reading `measure_psf` top to bottom.

**Fix applied:** `bin_image` now copies `image` before zeroing the `zeromask` positions, so the
caller's array is untouched. Verified this doesn't change any return value anywhere in the call
chain (`check_flux_calibration` already overwrites the corresponding output positions with `nan`
right after, independent of this internal side effect). Regression test:
`tests/test_utils.py::TestBinImage::test_does_not_mutate_the_caller_array`.

## 4. ✅ `assert`-based input validation throughout `musepsf.py` / `utils.py`

**Where:** e.g. `src/musepsf/musepsf.py:146-148`, `src/musepsf/utils.py:252`, `:337-338`, `:556`

Shape/unit/consistency checks are written as bare `assert` statements. Two problems: (1) they are
silently stripped when Python runs with the `-O` optimization flag, so the checks disappear
exactly when something is misconfigured in a "production" run; (2) they raise `AssertionError`,
which is not a very descriptive exception type for API misuse.

**Fix applied:** replaced all 8 occurrences (`apply_mask`, `to_minimize` x2, `run_measure_psf`,
`rmse_test`, `MUSEImage.measure_psf` x3) with explicit `if ...: raise ValueError(...)`, same
messages. Regression tests: `tests/test_utils.py::TestApplyMask::test_shape_mismatch_raises`,
`tests/test_musepsf.py::TestMeasurePsfValidation`.

## 5. ✅ `fit_2D_psf(data, mode='gaussian')` always raises `TypeError`

**Where:** `src/musepsf/utils.py:770-816` (found while writing `tests/test_utils.py`, Phase 2)

```python
model = models.Gaussian2D(amplitude=1, x_0=data.shape[1]/2, y_0=data.shape[0]/2,
                           stddev_x=1, stddev_y=1)
```

Astropy's `Gaussian2D` model does not have parameters named `x_0`/`y_0`/`stddev_x`/`stddev_y` — it
uses `x_mean`/`y_mean`/`x_stddev`/`y_stddev`. Calling `fit_2D_psf(..., mode='gaussian')` raises
`TypeError: Gaussian2D.__init__() got an unrecognized parameter 'x_0'` unconditionally. This mode
is not reachable from the normal package workflow (`build_psf`/`open_psf` default to
`mode='empirical'`), which is presumably why it went unnoticed.

**Fix applied:** renamed the constructor kwargs to `x_mean`/`y_mean`/`x_stddev`/`y_stddev`.
`fit.x_fwhm`/`fit.y_fwhm` were already the correct attribute names (verified against the installed
astropy version), so no further change was needed there. Regression test:
`tests/test_utils.py::TestFit2DPsf::test_gaussian_mode_recovers_injected_fwhm`.

## 6. 🔴 `reconstruct_psf()` mis-indexes the SDSS `psField` coefficient table

**Where:** `src/musepsf/utils.py:611-641` (found while writing `tests/test_utils.py`, Phase 2;
third sub-issue found during the KNOWN_ISSUES fix pass, see below)

Three separate problems, all in the coefficient-combination logic:

- `nb = nrow_b * ncol_b` is a NumPy scalar; `coeffs = np.zeros(nb.size, float)` uses `.size`,
  which is always `1` for a scalar regardless of the actual value of `nrow_b * ncol_b` — so the
  polynomial coefficient loop (`for ii in range(nb.size)`) only ever runs once, silently ignoring
  all but the first basis term whenever the real PSF model uses more than one.
- `cmat = ps['c']` reads the *whole* FITS table column (shape `(nrows, ...)`), but the indexing
  a few lines down (`cmat[int(ii / nrow_b), ii % nrow_b, jj]`) assumes no leading row axis. Tested
  this against a synthetic `psField`-like fixture built to match the real TDIM convention for a
  3D vector column (`float c[nrow_b][ncol_b][3]`, i.e. `TDIM='(3,ncol_b,nrow_b)'`): reading it back
  with `astropy.io.fits` does give an extra leading axis of size `nrows=1` on `ps['c']` — the code
  looks like it's missing a `[0]` (`cmat = ps['c'][0]`) to select that single row.
- `coeffs[ii] = (row * rcs) ** (ii % nrow_b) * (col * rcs) ** (ii / nrow_b)` uses a *fractional*
  exponent (`ii / nrow_b`, true division), while the `ecoeff` loop right below indexes with
  `int(ii / nrow_b)` (integer). For a 2D polynomial basis these should almost certainly use the
  same integer power, but this is inferred from how the surrounding code is structured, not
  confirmed against a real file.

**Not fixed**: left exactly as-is, by deliberate choice, pending validation against a real SDSS
`psField` file (no such file is available and none is fetched over the network for this repo's
test suite). Affects `create_sdss_psf()` and `Image.recover_SDSS_PSF()`.

## 7. ✅ `Image.get_gaia_catalog()` / `build_startable()` crash if `mask_galaxy()` was never called

**Where:** `src/musepsf/image.py:254-257` (`get_gaia_catalog`), `:330` (`build_startable`)
(found while writing `tests/test_image.py`, Phase 3)

```python
if self.galaxy is not None:
    mask2 = self.galaxy.contains(coords, wcs=self.wcs)

gaia_cat = gaia_cat[inside*(~mask2)*valid_mask].copy()
```

`mask2` is only assigned inside the `if self.galaxy is not None` branch, but used
unconditionally right after. If `mask_galaxy()` was never called (`self.galaxy` is still `None`,
its default from `__init__`), this raises `UnboundLocalError: cannot access local variable
'mask2'`. Separately, `build_startable()` unconditionally calls
`self.galaxy.to_pixel(self.wcs)` for its diagnostic plot, which raises `AttributeError: 'NoneType'
object has no attribute 'to_pixel'` under the same condition. In practice both methods have an
undocumented hard precondition — `mask_galaxy()` must be called first — that isn't mentioned in
either docstring.

**Fix applied:** in `get_gaia_catalog`, when `self.galaxy is None`, `mask2` now defaults to
all-`False` (so `~mask2` excludes nothing) and a `UserWarning` is raised pointing at the missing
`mask_galaxy()` call. In `build_startable`, the galaxy overlay on the diagnostic plot is now
skipped (no warning there, purely cosmetic). Regression tests:
`tests/test_image.py::TestGetGaiaCatalog::test_without_mask_galaxy_keeps_stars_and_warns`,
`tests/test_image.py::test_build_startable_works_without_mask_galaxy`.

## 8. ✅ `Image.recover_SDSS_PSF()` can silently store a `memoryview` instead of the PSF array

**Where:** `src/musepsf/image.py:463-465` (found while writing `tests/test_image.py`, Phase 3)

```python
if np.abs(1-psf_flux) < 0.0001:
    hdu = fits.PrimaryHDU(new_psf)
    self.psf = new_psf.data
```

`new_psf` here is already a plain `numpy.ndarray` (the return value of `create_sdss_psf`), not a
photutils fit-result object. `ndarray.data` is *not* "the array" — it is the raw buffer/
`memoryview` behind it. Whenever the reconstructed PSF already sums to ~1 (the `if` branch),
`self.psf` ends up holding a `memoryview`, not an array — confirmed with a synthetic PSF that
already summed to 1 (`type(img.psf)` was `<class 'memoryview'>` after the call). Anything
downstream expecting `self.psf` to be array-like (shape checks in `measure_psf`, `moffat_kernel`,
`rotate`, ...) would break. This line looks like it was copy-pasted from `build_psf`, where
`new_psf` genuinely is an object with a meaningful `.data` attribute (an `EPSFModel` fit result).

**Fix applied:** drop `.data` in that branch — just `self.psf = new_psf`. Regression test:
`tests/test_image.py::TestRecoverSdssPsf::test_stores_array_when_psf_already_normalized`.

## 9. ✅ `Image.recover_SDSS_PSF()` never sets `self.psf_fwhm`

**Where:** `src/musepsf/image.py:447-475`

Unlike `build_psf()` and `open_psf()`, `recover_SDSS_PSF()` does not call `fit_2D_psf(...)` at the
end, so `self.psf_fwhm` is left unset for any `Image` whose PSF was obtained through this method.
Not currently consumed anywhere inside `measure_psf()`, so it doesn't break the main workflow, but
it's an inconsistency between the three ways of populating `self.psf` and would surprise anyone
relying on `reference.psf_fwhm` being available after any of them.

**Fix applied:** not the originally suggested `fit_2D_psf(...)` call — this PSF comes directly
from the SDSS calibration file rather than being fit to a model, so a model-based FWHM isn't a
meaningful thing to compute here. Instead `self.psf_fwhm` is now explicitly set to `None`, so the
attribute always exists (consistent with `build_psf`/`open_psf`) but clearly signals "not
available" rather than silently being absent. Regression tests:
`tests/test_image.py::TestRecoverSdssPsf` (both cases assert `psf_fwhm is None`).

## 10. ✅ (minor, undocumented precondition) `Image.build_startable()` requires a masked `data` array

**Where:** `src/musepsf/image.py:295` (found while writing `tests/test_image.py`, Phase 3)

`cutout.data.mask.sum()` assumes the `data` array passed in is a `numpy.ma.MaskedArray` (true for
the real caller, `build_psf()`, when `pixscale` is not `None`: the data then comes from MPDAF's
`resample()`, which returns a masked array). Called with a plain `numpy.ndarray` — e.g. if
`build_psf(..., pixscale=None, stars_file=None)` is used, since then `data = self.data` — it
raises `AttributeError: 'numpy.ndarray' object has no attribute 'mask'`. Not documented anywhere
as a requirement of `build_startable`.

**Fix applied:** `build_startable` now reads the cutout's mask via
`getattr(cutout.data, 'mask', None)` and only checks/skips on masked-pixel count when a mask is
actually present, so a plain `ndarray` cutout is treated as fully unmasked instead of crashing.
Docstring updated to document the accepted input types. Regression test:
`tests/test_image.py::test_build_startable_works_with_plain_ndarray_data`.

## 11. ✅ (minor, efficiency-adjacent) `locate_stars()` always runs `DAOStarFinder` even when a manual region file is supplied

**Where:** `src/musepsf/utils.py:180-181` (calls `starfinder(image)` unconditionally, before checking `filename`)

When a manual star list (`filename=...`) is given, its result overrides the DAOStarFinder output,
but the (potentially expensive) star search still runs and its result is discarded. Not a
correctness bug, just wasted work; listed here so it isn't forgotten, but lower priority than the
rest of this list.

**Fix applied:** the `filename is not None` branch now short-circuits before the
`sigma_clipped_stats`/`DAOStarFinder` setup, so the star search only runs when no manual file is
supplied. Regression test:
`tests/test_utils.py::TestLocateStars::test_manual_file_skips_starfinder_call` (asserts
`DAOStarFinder` is never constructed in that path).

## 12. ✅ `to_minimize()` masking never actually excluded pixels from the fit

**Where:** `src/musepsf/utils.py:359-364` (`to_minimize`), `:594` (`leastsq` call in
`run_measure_psf`), `:931-934` (`rmse_test`). Found while implementing
`IMPLEMENTATION_PLAN.md` Phase 1 (the masked-residual handling needed for the `least_squares`
migration).

`apply_mask()` correctly builds a `numpy.ma.MaskedArray` with `starmask`/`zeromask` (star
positions, dilated edges, zero/NaN pixels) masked out. `to_minimize()` returned that masked array
via `function.ravel()`. But nothing downstream is mask-aware:

- `scipy.optimize.leastsq` coerces its objective function's return value with `numpy.asarray(...)`
  internally, which drops `.mask` and keeps the raw (masked) data values. Verified black-box:
  ```python
  a = np.ma.masked_array([1., 2., 3., 100.], mask=[False, False, False, True])
  def resid(p):
      return np.ma.masked_array(np.array([1.,2.,3.,100.]) - p[0], mask=a.mask).ravel()
  leastsq(resid, x0=[0.])   # -> [26.5], i.e. (1+2+3+100)/4 -- the masked 100 was never excluded
  ```
- `rmse_test()` does `np.nanmean(out**2)` on the same masked return value; `np.nanmean` is not
  mask-aware either, so the RMSE grid used the same unmasked-in-practice residuals.

In effect, `starmask`/`zeromask` (including the border always dilated by `edge` pixels in
`_prepare_fit_arrays`) were built correctly but never excluded anything from the actual PSF fit or
from `rmse_test`'s sensitivity map -- every pixel counted, star mask included.

**Fix applied:** `to_minimize()` now returns `function.compressed()` (drops masked entries
entirely) instead of `function.ravel()`. Both `leastsq`/`least_squares` and `rmse_test`'s
`np.nanmean` consume this same return value, so the fix applies to both call sites without further
changes there. Regression tests: `tests/test_utils.py::TestToMinimizeMasking` (asserts the return
value is a plain, non-masked array; that corrupting pixels *inside* the star mask does not change
the residual; and that corrupting pixels *outside* it does, as a sanity check against a vacuously
passing test).

**Note:** this changes actual fit output for any run that previously had pixels excluded by
`starmask`/`zeromask` with non-negligible residuals under the mask (i.e. essentially all real runs
with a star mask). The Phase 0 golden-master case in `tests/test_utils.py` happened not to change
value (its injected point sources were kept ≥`edge` pixels from the border by construction, so the
masked border pixels had negligible signal either way) -- this is a coincidence of that specific
synthetic case, not evidence the fix is a no-op in general.

## 13. ✅ `rmse_test()` ignored its own `scale` argument, always using 0.2 arcsec/pixel internally

**Where:** `src/musepsf/utils.py:996` (the `to_minimize()` call inside `rmse_test()`'s grid-scan
loop). Found while refactoring `rmse_test()` to use `FitContext` (`IMPLEMENTATION_PLAN.md` Phase
3); fixed while implementing Phase 2 (`rmse_test` as quality control), since Phase 2's confidence
intervals are only meaningful if the grid is built in the correct physical units.

`rmse_test(..., scale=0.2, ...)` accepts a `scale` argument (pixel scale in arcsec/pixel) and
passes it into `_prepare_fit_arrays()`, but that value only affected a throwaway kernel used
internally to compute the FFT frequency grid shape (a shape that doesn't actually depend on the
numeric value of `scale` at all). The grid-scan loop's own `to_minimize(...)` calls hardcoded
`scale=0.2` directly, regardless of what `scale` the function was actually called with. Any caller
using `rmse_test()` on data with a different pixel scale -- e.g. oversampled data, where
`scale = original_scale / oversample` -- got a sensitivity/RMSE map computed with the wrong
physical FWHM/offset units on its axes.

**Fix applied:** removed the hardcoded `scale=0.2` from the `to_minimize()` call; it now uses the
`FitContext` built from the function's own `scale` argument, same as `run_measure_psf()`.
Regression test: `tests/test_utils.py::TestRmseTest::test_uses_the_passed_scale_not_a_hardcoded_default`.

## 14. ✅ Internal Moffat model kernel used an even `img_size`, biasing the fitted `dx`/`dy` by ~0.5 pixel

**Where:** `src/musepsf/utils.py` -- the `moffat_kernel(..., img_size=50*ctx.oversample)` calls in
`to_minimize()`, `plot_results()`, and `_prepare_fit_arrays()` (three occurrences, all had to
change together since the shapes they produce must match). Found in discussion with the user about
what the fitted `dx`/`dy` parameters actually represent -- the offsets did not look like genuine
astrometric registration residuals.

`astropy.convolution.convolve_fft()` centers an even-sized kernel array at a spurious half-pixel
offset from the true geometric center (there is no single center pixel for an even-length axis).
Verified directly: convolving a point source injected at the exact center of a pixel with an
odd-sized Moffat kernel array leaves the result exactly centered (0.000 px centroid shift); the
same test with an even-sized array (`img_size=50`, the value these three call sites used) shifts
the centroid by ~0.4-0.6 px (the exact amount depends on the Moffat shape, not a clean 0.5).

`to_minimize()`'s model of the MUSE PSF (`ker_MUSE`) was built with this even `img_size`, but the
MUSE-side convolution in `run_measure_psf()` (`convolve_fft(data, psf)`, using the externally
provided/measured `reference.psf`, typically odd-sized -- e.g. `build_psf()`'s default `npix=35`)
was not. This kernel-parity mismatch between the two sides of the fit meant the optimizer's
`dx`/`dy` parameters partly (or, in the synthetic case below, almost entirely) absorbed this
bookkeeping artifact rather than a genuine registration offset between the images. Confirmed
end-to-end: a synthetic MUSE/reference pair built from the exact same source pixel grid (true
offset = 0) consistently fit `dx = dy ≈ 0.500` before this fix. It also visibly distorted `fwhm`/
`alpha` in fits that don't include an offset term (`fit_alpha=True, offset=False` on that same
synthetic case: `fwhm` went from 0.838 to 0.800, `alpha` from 2.897 to 2.797 -- much closer to the
true injected values of 0.8 and 2.8 -- once the kernel-parity bug was fixed), since the optimizer
was otherwise trying to compensate for the phantom shift through the PSF shape instead.

**Fix applied:** all three call sites now use `img_size=50*ctx.oversample + 1` (or
`50*oversample + 1` in `_prepare_fit_arrays()`), always odd. Regression tests:
`tests/test_utils.py::test_even_sized_kernel_shifts_convolution_centroid` (isolates the root cause
at the `convolve_fft()` level, independent of the fit), `TestRunMeasurePsfGoldenMaster::
test_offset_fit_recovers_true_zero_offset` (end-to-end: the zero-true-offset synthetic case now
recovers `dx = dy ≈ 0`), and the `TestRunMeasurePsfGoldenMaster` golden-master values (re-frozen a
second time, see that class's docstring).

**Note:** this changes real fit output for essentially every run with `offset=True` (dx/dy no
longer carry the ~0.5 px bias), and to a smaller degree for runs with `offset=False` (fwhm/alpha
no longer partially compensate for it either).

## 15. ✅ `rmse_test()`'s QC grid was corrupted by `to_minimize()`'s (now-vestigial) bounds-penalty hack

**Where:** `src/musepsf/utils.py:918` (`rmse_test()`'s `_prepare_fit_arrays()` call, before the
fix left `fwhm_bound`/`alpha_bound`/`dd_bound` at their `_prepare_fit_arrays()` defaults), consumed
by `to_minimize()`'s bound check at `:318-332`. Found while investigating a user report that
`plot_qc_summary()`'s output "looked weird" (RMSE heatmaps mostly flat/dark except a few cells at
~1e11, degenerate `[0.000, 0.000]` confidence intervals).

`to_minimize()` has a bounds-penalty branch (`git blame`: 2023-05-29, commit `d18c707`, i.e.
written for the original `scipy.optimize.leastsq`, which has no native bounds support): if a
parameter falls outside `ctx.fwhm_bound`/`ctx.alpha_bound`/`ctx.dd_bound`, it clips the parameter
to a placeholder and multiplies the whole residual by `1e10`. `run_measure_psf()` has since
migrated to `scipy.optimize.least_squares(..., bounds=bounds)` (`:610`), which enforces `bounds=`
natively and never calls `to_minimize()` outside them -- so this branch is dead code for the
actual fit.

`rmse_test()`'s diagnostic grid scan calls `to_minimize()` directly, though, with points that are
meant to go beyond the best fit by construction (`dx0 +/- delta_off`, etc. -- that is the point of
a sensitivity scan). Its `FitContext` was still being built with `_prepare_fit_arrays()`'s
optimizer-oriented bound defaults (`fwhm_bound=[0.4, 2]`, `dd_bound=[-2, 2]`, independent of
whatever bound the actual fit used), so any grid point where `dx0`/`dy0`/`fwhm0` + the tested
offset crossed that hardcoded default got hit with the `1e10` penalty -- not a measure of fit
quality at that point. Reproduced with the synthetic fixture and a plausible `offset=True` best
fit (`dx0=1.7, dy0=-1.8`): 971/1331 grid cells (73%) contaminated, max value `7.9e8`, QC verdict
flipped to a spurious FAIL ("minimum not at center"), and `confidence_from_rmse_cube()`'s
confidence intervals collapsed to degenerate `(0, 0)`/edge values -- matching the user's screenshot
exactly. Existing tests never caught this because they all use `dx0=dy0=0` with a small
`delta_off=0.5`, always safely inside the default bound.

**Fix applied:** `rmse_test()` now builds its `FitContext` with `fwhm_bound=None,
alpha_bound=None, dd_bound=None`, disabling `to_minimize()`'s bounds-penalty branches entirely for
the grid scan. The one genuine physical constraint left -- `moffat_kernel()` needs `fwhm > 0`
(astropy's `Moffat2DKernel` requires `gamma > 0`) -- is now enforced directly in the grid loop via
`new_fhwm = max(fwhm0 + df, 1e-3)`, without the `1e10` residual inflation. Regression tests:
`tests/test_utils.py::TestRmseTest::
test_grid_scan_is_not_corrupted_by_bounds_penalty_when_offset_exceeds_default_bound` and
`::test_grid_scan_handles_fwhm_going_non_positive_at_grid_edge`.

**Note:** this changes `rmse_test()`/`confidence_from_rmse_cube()`/`qc_from_rmse_cube()` output
for any call where the tested grid extends past the old hardcoded `[-2, 2]`/`[0.4, 2]` bound
relative to `dx0`/`dy0`/`fwhm0` -- in practice, any real `qc=True` run with a nonzero best-fit
offset and the default `delta_off=2`. Does not affect `run_measure_psf()`'s actual fit output
(that path never triggered the penalty branch in the first place, see above).

## 16. ✅ `plot_qc_summary()`'s FWHM-axis panels had unreadable, overlapping tick labels

**Where:** `src/musepsf/plotting.py:129,140` (`_draw_rmse_grid()`'s `ax.matshow(...)` calls for
the `(y, FWHM)` and `(x, FWHM)` panels). Found alongside #15, in the same user-reported screenshot.

`matshow()` defaults to `aspect='equal'` (locks 1 data-unit-x to the same rendered length as 1
data-unit-y). The `(y, x)` panel's extent is symmetric (`+-delta_off` on both axes) so this looks
fine, but the `(y, FWHM)`/`(x, FWHM)` panels have a 4:1 extent (`+-delta_off=2` vs
`+-delta_fwhm=0.5` by default): with `aspect='equal'`, matplotlib squeezed the actual plotted
heatmap into a narrow sliver of the panel, cramming all 11 tick labels into that sliver as
unreadable overlapping text.

**Fix applied:** both `matshow()` calls now pass `aspect='auto'`, letting the heatmap fill the
panel regardless of its data extent's aspect ratio. Regression test:
`tests/test_plotting.py::TestDrawRmseGrid::test_fwhm_axis_panels_use_auto_aspect_not_equal`.
