import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve_fft
from astropy.modeling import models
from astropy.table import Table
from numpy.fft import fftfreq
from scipy.optimize import OptimizeResult

from musepsf.utils import (
    apply_mask,
    apply_offset_fourier,
    bin_image,
    check_flux_calibration,
    download_ps_file,
    find_peaks_2d,
    fit_2D_psf,
    linear_function,
    locate_stars,
    moffat_kernel,
    query_gaia,
    rebin,
    remove_close_stars,
    run_measure_psf,
    to_minimize,
)
from musepsf.utils import _covariance_from_least_squares, _pack_bounds, _prepare_fit_arrays
from musepsf.plotting import get_norm


def test_linear_function():
    assert linear_function([2.0, 3.0], np.array([0.0, 1.0, 2.0])) == pytest.approx([3.0, 5.0, 7.0])


def test_get_norm_returns_sane_limits():
    image = np.linspace(0, 100, 400).reshape(20, 20)
    norm = get_norm(image, perc=99.9)
    assert norm.vmin < norm.vmax


class TestBinImage:
    def test_recovers_known_block_medians(self):
        bin_size = 4
        blocks = np.arange(9).reshape(3, 3).astype(float)
        image = np.kron(blocks, np.ones((bin_size, bin_size)))

        median, std = bin_image(image.copy(), bin_size=bin_size)

        assert median == pytest.approx(blocks.ravel())
        assert std == pytest.approx(np.zeros(9))

    def test_output_is_raveled_to_expected_length(self):
        image = np.zeros((16, 24))
        median, std = bin_image(image, bin_size=4)
        assert median.shape == std.shape == (4 * 6,)

    def test_does_not_mutate_the_caller_array(self):
        # Regression test for KNOWN_ISSUES.md #3: bin_image used to zero the input array
        # in-place wherever zeromask is True, as a side effect on the caller's array.
        image = np.ones((8, 8))
        zeromask = np.zeros((8, 8), dtype=bool)
        zeromask[0, 0] = True

        bin_image(image, bin_size=4, zeromask=zeromask)

        assert image[0, 0] == 1.0

    @staticmethod
    def _bin_image_loop_reference(image, bin_size=15, zeromask=None):
        # Pre-Phase-4 implementation of bin_image(), kept here only as a
        # reference to check the vectorized version against on random input.
        image = image.copy()
        if zeromask is not None:
            image[zeromask] = 0
        n_x = image.shape[1] // bin_size
        n_y = image.shape[0] // bin_size
        median = np.zeros((n_y, n_x))
        std = np.zeros_like(median)
        for i in range(n_x):
            for j in range(n_y):
                median[j, i] = np.median(image[j * bin_size:j * bin_size + bin_size,
                                                i * bin_size:i * bin_size + bin_size])
                std[j, i] = np.std(image[j * bin_size:j * bin_size + bin_size,
                                          i * bin_size:i * bin_size + bin_size])
        return np.ravel(median), np.ravel(std)

    @pytest.mark.parametrize('shape,bin_size', [
        ((45, 45), 4), ((45, 45), 15), ((100, 73), 5), ((16, 24), 4), ((203, 151), 7),
    ])
    def test_vectorized_implementation_matches_loop_reference(self, shape, bin_size):
        rng = np.random.default_rng(0)
        image = rng.normal(size=shape)
        zeromask = rng.random(shape) < 0.05

        expected_median, expected_std = self._bin_image_loop_reference(image.copy(), bin_size, zeromask.copy())
        median, std = bin_image(image.copy(), bin_size, zeromask.copy())

        assert median == pytest.approx(expected_median)
        assert std == pytest.approx(expected_std)


def test_moffat_kernel_is_normalized_and_centered():
    kernel = moffat_kernel(fwhm=0.8, alpha=2.8, scale=0.2, img_size=41)
    assert kernel.array.shape == (41, 41)
    assert kernel.array.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.unravel_index(np.argmax(kernel.array), kernel.array.shape) == (20, 20)


@pytest.mark.parametrize('img_size,expected_shift', [(49, 0.0), (50, None), (51, 0.0)])
def test_even_sized_kernel_shifts_convolution_centroid(img_size, expected_shift):
    # Regression/documentation test for KNOWN_ISSUES.md #14: convolve_fft() centers an
    # even-sized kernel array at a spurious half-pixel offset. A point source injected at the
    # exact center of a pixel stays exactly centered after convolving with an odd-sized kernel
    # (img_size=49 or 51 here), but its centroid shifts noticeably (by roughly, though not
    # exactly, half a pixel) when convolved with an even-sized one (img_size=50, the value
    # to_minimize()/plot_results()/_prepare_fit_arrays() used before the fix -- they now all use
    # an odd img_size to avoid this).
    size = 81
    img = np.zeros((size, size))
    img[40, 40] = 1.0

    kernel = moffat_kernel(fwhm=0.8, alpha=2.8, scale=0.2, img_size=img_size)
    out = convolve_fft(img, kernel)
    peak = np.unravel_index(np.argmax(out), out.shape)

    from scipy.ndimage import center_of_mass
    w = 5
    y0, x0 = peak
    sub = out[y0 - w:y0 + w + 1, x0 - w:x0 + w + 1]
    cy, cx = center_of_mass(sub)
    dy = peak[0] + cy - w - 40
    dx = peak[1] + cx - w - 40

    if expected_shift is None:
        assert abs(dy) > 0.1 and abs(dx) > 0.1
    else:
        assert dy == pytest.approx(expected_shift, abs=1e-6)
        assert dx == pytest.approx(expected_shift, abs=1e-6)


class TestApplyMask:
    def test_without_starmask_only_nanmask_applies(self):
        image = np.arange(9.0).reshape(3, 3)
        nanmask = np.zeros((3, 3), dtype=bool)
        nanmask[1, 1] = True

        masked = apply_mask(image, None, nanmask)

        assert masked.mask[1, 1]
        assert not masked.mask[0, 0]

    def test_with_starmask_combines_both_masks(self):
        image = np.zeros((3, 3))
        nanmask = np.zeros((3, 3), dtype=bool)
        starmask = np.zeros((3, 3), dtype=bool)
        starmask[0, 0] = True

        masked = apply_mask(image, starmask, nanmask)

        assert masked.mask[0, 0]

    def test_shape_mismatch_raises(self):
        image = np.zeros((3, 3))
        nanmask = np.zeros((3, 3), dtype=bool)
        starmask = np.zeros((2, 2), dtype=bool)

        with pytest.raises(ValueError):
            apply_mask(image, starmask, nanmask)


class TestToMinimizeMasking:
    """
    Regression tests for KNOWN_ISSUES.md #12: to_minimize() used to return a MaskedArray
    (function.ravel()), whose mask was silently discarded by whatever consumed it downstream
    (leastsq/least_squares coerce with np.asarray(), np.nanmean() in rmse_test isn't mask-aware
    either) -- so masked pixels (star mask, edges, zeros) always leaked into the fit residual
    with their real value instead of being excluded. Fixed by returning function.compressed().
    """

    @staticmethod
    def _synthetic_context(shape=(40, 40), edge=2, scale=0.2):
        rng = np.random.default_rng(3)
        reference = rng.normal(10, 1, size=shape)
        convolved = reference.copy()

        starmask = np.zeros(shape, dtype=bool)
        starmask[10:15, 10:15] = True
        zeromask = np.zeros(shape, dtype=bool)

        ctx = _prepare_fit_arrays(reference, zeromask, starmask, edge=edge, scale=scale, oversample=1)

        return convolved, reference, ctx

    def test_return_value_is_plain_array_not_masked(self):
        convolved, reference, ctx = self._synthetic_context()

        out = to_minimize([0.8], convolved, reference, ctx, alpha0=2.8)

        assert not np.ma.is_masked(out)
        assert out.size < convolved.size  # edge + star mask really dropped some pixels

    def test_corrupting_masked_pixels_does_not_change_the_residual(self):
        convolved, reference, ctx = self._synthetic_context()
        pars = [0.8]

        out_before = to_minimize(pars, convolved, reference, ctx, alpha0=2.8)

        corrupted = convolved.copy()
        corrupted[ctx.starmask] = 1e6  # only touches pixels inside the (always-masked) star region

        out_after = to_minimize(pars, corrupted, reference, ctx, alpha0=2.8)

        assert out_after == pytest.approx(out_before)

    def test_corrupting_unmasked_pixels_does_change_the_residual(self):
        # sanity check that the test above isn't vacuously true (e.g. to_minimize ignoring
        # `convolved` altogether): corrupting an unmasked pixel must move the residual.
        convolved, reference, ctx = self._synthetic_context()
        pars = [0.8]

        out_before = to_minimize(pars, convolved, reference, ctx, alpha0=2.8)

        corrupted = convolved.copy()
        corrupted[20, 20] = 1e6  # well outside the star mask and the dilated edge

        out_after = to_minimize(pars, corrupted, reference, ctx, alpha0=2.8)

        assert not np.allclose(out_after, out_before)


class TestApplyOffsetFourier:
    @staticmethod
    def _grids(shape):
        fx = fftfreq(shape[1])
        fy = fftfreq(shape[0])
        fxx, fyy = np.meshgrid(fx, fy)
        arrayslices = [0, shape[0], 0, shape[1]]
        return fxx, fyy, arrayslices

    def test_zero_shift_is_identity(self):
        img = np.random.default_rng(0).normal(size=(12, 12))
        spectrum = np.fft.fftn(img)
        fxx, fyy, arrayslices = self._grids(img.shape)

        out = apply_offset_fourier(spectrum.copy(), 0, 0, fxx, fyy, arrayslices)

        assert out == pytest.approx(img, abs=1e-8)

    def test_integer_shift_matches_np_roll(self):
        img = np.zeros((12, 12))
        img[3, 4] = 1.0
        spectrum = np.fft.fftn(img)
        fxx, fyy, arrayslices = self._grids(img.shape)

        dx, dy = 1, 2
        out = apply_offset_fourier(spectrum.copy(), dx, dy, fxx, fyy, arrayslices)
        expected = np.roll(img, shift=(dy, dx), axis=(0, 1))

        assert out == pytest.approx(expected, abs=1e-8)


def test_rebin_conserves_flux_for_constant_image():
    factor = 2
    image = np.full((40, 40), 3.0)

    result = rebin(image, factor)

    # away from the edges, smoothing+resampling a constant image stays constant,
    # and multiplying by factor**2 turns pixel intensity into the "flux" of the
    # larger rebinned pixel
    assert result[5:-5, 5:-5] == pytest.approx(3.0 * factor**2, rel=0.02)


def test_find_peaks_2d_detects_injected_peaks():
    data = np.zeros((20, 20))
    data[5, 5] = 10
    data[15, 12] = 8

    peaks = {tuple(p) for p in find_peaks_2d(data, threshold=1)}

    assert peaks == {(5, 5), (15, 12)}


def test_remove_close_stars_removes_pair_below_min_distance():
    # first two stars are ~1 arcsec apart (below the 3.5 arcsec threshold),
    # the other two are far from everything
    coords = SkyCoord(
        ra=[10.0, 10.0 + 1.0 / 3600, 20.0, 30.0] * u.deg,
        dec=[0.0, 0.0, 0.0, 0.0] * u.deg,
    )

    mask = remove_close_stars(coords)

    assert list(mask) == [False, False, True, True]


class TestFit2DPsf:
    def test_moffat_mode_recovers_injected_fwhm(self):
        size = 41
        model = models.Moffat2D(amplitude=1.0, x_0=size // 2, y_0=size // 2, gamma=5.0, alpha=2.5)
        yy, xx = np.indices((size, size))
        data = model(xx, yy)

        recovered = fit_2D_psf(data, mode='moffat')

        assert recovered == pytest.approx(model.fwhm, rel=1e-3)

    def test_empirical_mode_recovers_injected_fwhm(self):
        size = 41
        model = models.Moffat2D(amplitude=1.0, x_0=size // 2, y_0=size // 2, gamma=5.0, alpha=2.5)
        yy, xx = np.indices((size, size))
        data = model(xx, yy)

        recovered = fit_2D_psf(data, mode='empirical')

        assert recovered == pytest.approx(model.fwhm, rel=0.02)

    def test_gaussian_mode_recovers_injected_fwhm(self):
        # Regression test for KNOWN_ISSUES.md #5: fit_2D_psf(mode='gaussian') used to
        # build models.Gaussian2D with parameter names ('x_0', 'stddev_x', ...) that
        # astropy's Gaussian2D does not accept ('x_mean', 'x_stddev', ...), so this mode
        # always raised TypeError.
        size = 41
        stddev = 3.0
        model = models.Gaussian2D(amplitude=1.0, x_mean=size // 2, y_mean=size // 2,
                                   x_stddev=stddev, y_stddev=stddev)
        yy, xx = np.indices((size, size))
        data = model(xx, yy)

        recovered = fit_2D_psf(data, mode='gaussian')

        expected = stddev * 2 * np.sqrt(2 * np.log(2))
        assert recovered == pytest.approx(expected, rel=1e-6)


def test_check_flux_calibration_recovers_linear_relation():
    rng = np.random.default_rng(2)
    muse = rng.uniform(1, 10, size=(45, 45))
    true_gamma, true_b = 2.0, 0.5
    reference = true_gamma * muse + true_b
    zeromask = np.zeros(muse.shape, dtype=bool)

    rescaled = check_flux_calibration(muse.copy(), reference, zeromask, bin_size=5, plot=False)

    assert rescaled == pytest.approx(true_gamma * muse + true_b, abs=1e-6)


class TestLocateStars:
    def test_detects_injected_star(self):
        rng = np.random.default_rng(1)
        shape = (60, 60)
        image = rng.normal(scale=0.1, size=shape)
        yy, xx = np.indices(shape)
        star = models.Gaussian2D(amplitude=50, x_mean=20, y_mean=30, x_stddev=1.5, y_stddev=1.5)
        image += star(xx, yy)

        stars, mask = locate_stars(image, filename=None, fwhm=3, brightest=5, sigma=3.0, radius=5)

        assert len(stars) == 1
        assert stars['xcentroid'][0] == pytest.approx(20, abs=0.5)
        assert stars['ycentroid'][0] == pytest.approx(30, abs=0.5)
        assert mask[30, 20]
        assert not mask[0, 0]

    def test_manual_file_overrides_detection(self, tmp_path):
        manual_file = tmp_path / 'manual.reg'
        manual_file.write_text('12.5 30.2\n40.0 10.0\n')
        image = np.zeros((60, 60))

        stars, mask = locate_stars(image, filename=str(manual_file), radius=4)

        assert list(stars['xcentroid']) == pytest.approx([12.5, 40.0])
        assert list(stars['ycentroid']) == pytest.approx([30.2, 10.0])
        assert mask.sum() > 0

    def test_no_sources_and_no_filename_returns_none(self):
        image = np.zeros((30, 30))

        stars, mask = locate_stars(image, filename=None)

        assert stars is None
        assert mask is None

    def test_manual_file_skips_starfinder_call(self, tmp_path, monkeypatch):
        # Regression test for KNOWN_ISSUES.md #11: locate_stars() used to run
        # DAOStarFinder unconditionally and discard the result whenever a manual
        # region file was supplied. It should now be skipped entirely.
        manual_file = tmp_path / 'manual.reg'
        manual_file.write_text('12.5 30.2\n')
        image = np.zeros((30, 30))

        def _boom(*args, **kwargs):
            raise AssertionError('DAOStarFinder should not be constructed when filename is given')

        monkeypatch.setattr('musepsf.utils.DAOStarFinder', _boom)

        stars, mask = locate_stars(image, filename=str(manual_file), radius=4)

        assert list(stars['xcentroid']) == pytest.approx([12.5])


def test_query_gaia_filters_by_classprob(mock_gaia_query):
    table = Table({
        'ra': [10.0, 11.0, 12.0],
        'dec': [0.0, 0.0, 0.0],
        'parallax': [1.0, 1.0, 1.0],
        'phot_g_mean_mag': [15.0, 16.0, 17.0],
        'classprob_dsc_combmod_star': [0.999, 0.5, 0.995],
    })
    mock_gaia_query(table)

    result = query_gaia(SkyCoord(10 * u.deg, 0 * u.deg), 5 * u.arcmin)

    assert list(result['phot_g_mean_mag']) == pytest.approx([15.0, 17.0])
    assert set(result.colnames) == {'ra', 'dec', 'parallax', 'phot_g_mean_mag',
                                     'classprob_dsc_combmod_star'}


class TestPackBounds:
    @pytest.mark.parametrize('fit_alpha,offset,expected_len', [
        (False, False, 1),
        (True, False, 2),
        (False, True, 3),
        (True, True, 4),
    ])
    def test_bounds_length_matches_p0_length(self, fit_alpha, offset, expected_len):
        # must match the p0 length run_measure_psf() builds for the same fit_alpha/offset
        lower, upper = _pack_bounds(fit_alpha, offset, fwhm_bound=[0.2, 2], alpha_bound=[1, 10],
                                    dd_bound=[-2, 2])
        assert len(lower) == len(upper) == expected_len

    def test_bound_ordering_matches_unpack_fit_params(self):
        # offset=True, fit_alpha=True -> [fwhm, dx, dy, alpha], same order _unpack_fit_params() expects
        lower, upper = _pack_bounds(fit_alpha=True, offset=True, fwhm_bound=[0.2, 2],
                                    alpha_bound=[1, 10], dd_bound=[-3, 3])
        assert lower == [0.2, -3, -3, 1]
        assert upper == [2, 3, 3, 10]


class TestCovarianceFromLeastSquares:
    def test_matches_manual_formula_for_well_posed_fit(self):
        jac = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        fun = np.array([0.1, -0.2, 0.05])
        res = OptimizeResult(x=np.zeros(2), jac=jac, fun=fun, cost=0.5 * np.sum(fun**2))

        cov = _covariance_from_least_squares(res)

        dof = fun.size - 2
        expected = np.linalg.inv(jac.T @ jac) * (2 * res.cost / dof)
        assert cov == pytest.approx(expected)

    def test_returns_nan_matrix_when_dof_is_not_positive(self):
        jac = np.eye(2)
        fun = np.array([0.1, -0.2])  # as many data points as parameters -> dof == 0
        res = OptimizeResult(x=np.zeros(2), jac=jac, fun=fun, cost=0.5 * np.sum(fun**2))

        cov = _covariance_from_least_squares(res)

        assert cov.shape == (2, 2)
        assert np.all(np.isnan(cov))

    def test_returns_nan_matrix_when_jtj_is_singular(self):
        jac = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])  # rank-deficient columns
        fun = np.array([0.1, -0.2, 0.05])
        res = OptimizeResult(x=np.zeros(2), jac=jac, fun=fun, cost=0.5 * np.sum(fun**2))

        cov = _covariance_from_least_squares(res)

        assert cov.shape == (2, 2)
        assert np.all(np.isnan(cov))


def _build_synthetic_fit_case():
    """
    Deterministic synthetic MUSE/reference pair for the run_measure_psf/rmse_test golden-master
    tests below (Phase 0 of IMPLEMENTATION_PLAN.md). This is NOT an injection-recovery accuracy
    test: it exists only to freeze today's numeric output so later refactors (least_squares
    migration, FitContext) can be checked for non-regression against it. Not meant to assert that
    the pipeline recovers the injected FWHM accurately.
    """
    rng = np.random.default_rng(42)
    size = 121
    scale = 0.2

    true_sky = np.zeros((size, size))
    for _ in range(15):
        y, x = rng.integers(10, size - 10, size=2)
        true_sky[y, x] += rng.uniform(5, 20)

    ref_kernel = moffat_kernel(fwhm=0.3, alpha=2.5, scale=scale, img_size=31)
    psf_ref = ref_kernel.array / ref_kernel.array.sum()
    reference_data = convolve_fft(true_sky, ref_kernel)

    true_alpha = 2.8
    muse_kernel = moffat_kernel(fwhm=0.8, alpha=true_alpha, scale=scale, img_size=61)
    data = convolve_fft(true_sky, muse_kernel)

    zeromask = np.zeros((size, size), dtype=bool)
    starmask = np.zeros((size, size), dtype=bool)

    return data, reference_data, psf_ref, zeromask, starmask, scale, true_alpha


class TestRunMeasurePsfGoldenMaster:
    """
    Regression/characterization tests (Phase 0): freeze today's run_measure_psf output on a fixed
    synthetic case for each fit_alpha/offset combination, so Phase 3 (FitContext refactor) can be
    checked against these values. `res` is a scipy.optimize.OptimizeResult since Phase 1
    (leastsq -> least_squares migration).

    The expected values below were re-frozen a second time after KNOWN_ISSUES.md #14 (odd
    img_size fix for the internal Moffat kernel): before that fix, every case here converged to a
    spurious ~0.5 pixel dx/dy (even with a zero true offset -- see
    test_offset_fit_recovers_true_zero_offset below) which also visibly distorted fwhm/alpha in
    the cases that don't fit for offset (e.g. fit_alpha=True, offset=False used to converge to
    fwhm=0.838 -- now 0.800, much closer to the true 0.8 injected FWHM).
    """

    @pytest.mark.parametrize('fit_alpha,offset,expected', [
        (False, False, [0.80019739]),
        (False, True, [0.80019736, 0.00002430, -0.00000026]),
        (True, False, [0.79986968, 2.79658490]),
        (True, True, [0.79986977, 0.00000432, -0.00000110, 2.79658588]),
    ])
    def test_best_fit_matches_frozen_reference(self, fit_alpha, offset, expected):
        data, reference_data, psf_ref, zeromask, starmask, scale, true_alpha = _build_synthetic_fit_case()

        res, star_pos, starmask_out = run_measure_psf(
            data, reference_data, psf_ref, None, starmask.copy(), zeromask.copy(),
            oversample=1, figname=None, alpha=true_alpha, edge=10, fwhm0=0.6,
            fit_alpha=fit_alpha, plot=False, save=False, show=False, scale=scale, offset=offset,
        )

        assert res.x == pytest.approx(expected, abs=1e-3)
        assert star_pos is None

    def test_offset_fit_recovers_true_zero_offset(self):
        # Regression test for KNOWN_ISSUES.md #14: the internal Moffat model kernel used to be
        # built with an even img_size (50*oversample), which astropy's convolve_fft centers at a
        # spurious half-pixel offset from the true center (verified directly against a single
        # injected point source: convolving with an odd-sized kernel array leaves it exactly
        # centered, an even-sized one shifts the centroid by ~0.4-0.6 px). Since this synthetic
        # case's "MUSE" and "reference" images are built from the exact same source pixel grid,
        # the true dx/dy is exactly 0 -- before the fix, the fit consistently converged to
        # dx=dy=~0.5 instead (see the parametrized case above, pre-fix values in the class
        # docstring), a registration offset that was purely a kernel-centering artifact.
        data, reference_data, psf_ref, zeromask, starmask, scale, true_alpha = _build_synthetic_fit_case()

        res, *_ = run_measure_psf(
            data, reference_data, psf_ref, None, starmask.copy(), zeromask.copy(),
            oversample=1, figname=None, alpha=true_alpha, edge=10, fwhm0=0.6,
            fit_alpha=False, plot=False, save=False, show=False, scale=scale, offset=True,
        )

        assert res.x[1] == pytest.approx(0.0, abs=1e-3)
        assert res.x[2] == pytest.approx(0.0, abs=1e-3)

    def test_covariance_is_attached_and_well_formed(self):
        data, reference_data, psf_ref, zeromask, starmask, scale, true_alpha = _build_synthetic_fit_case()

        res, *_ = run_measure_psf(
            data, reference_data, psf_ref, None, starmask.copy(), zeromask.copy(),
            oversample=1, figname=None, alpha=true_alpha, edge=10, fwhm0=0.6,
            fit_alpha=False, plot=False, save=False, show=False, scale=scale, offset=False,
        )

        assert res.cov.shape == (1, 1)
        assert res.cov[0, 0] > 0

    def test_converges_cleanly_to_the_bound_when_the_true_optimum_lies_outside_it(self):
        # Phase 1 (IMPLEMENTATION_PLAN.md): with real bounds (least_squares), a true optimum
        # outside fwhm_bound should make the fit converge smoothly to the bound itself, rather
        # than relying on to_minimize()'s internal 1e10 penalty hack (still there for rmse_test's
        # manual grid scan, but no longer load-bearing for the optimizer itself).
        rng = np.random.default_rng(42)
        size = 121
        scale = 0.2
        true_sky = np.zeros((size, size))
        for _ in range(15):
            y, x = rng.integers(10, size - 10, size=2)
            true_sky[y, x] += rng.uniform(5, 20)

        ref_kernel = moffat_kernel(fwhm=0.3, alpha=2.5, scale=scale, img_size=31)
        psf_ref = ref_kernel.array / ref_kernel.array.sum()
        reference_data = convolve_fft(true_sky, ref_kernel)

        # true FWHM well above the default fwhm_bound upper limit of 2.0
        muse_kernel = moffat_kernel(fwhm=3.0, alpha=2.8, scale=scale, img_size=61)
        data = convolve_fft(true_sky, muse_kernel)

        zeromask = np.zeros((size, size), dtype=bool)
        starmask = np.zeros((size, size), dtype=bool)

        res, *_ = run_measure_psf(
            data, reference_data, psf_ref, None, starmask, zeromask,
            oversample=1, figname=None, alpha=2.8, edge=10, fwhm0=0.6,
            fit_alpha=False, plot=False, save=False, show=False, scale=scale, offset=False,
        )

        assert res.success
        assert res.x[0] == pytest.approx(2.0)


def test_download_ps_file_builds_expected_url_and_calls_wget(mock_wget_download, tmp_path):
    out_dir = str(tmp_path)

    result = download_ps_file(run=94, camcol=1, frame=11, rerun=301, out_dir=out_dir)

    assert result is True
    assert len(mock_wget_download) == 1
    url, out = mock_wget_download[0]
    assert url.endswith('/301/94/objcs/1/psField-000094-1-0011.fit')
    assert out == out_dir
