import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.modeling import models
from astropy.table import Table
from numpy.fft import fftfreq

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
)
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


def test_download_ps_file_builds_expected_url_and_calls_wget(mock_wget_download, tmp_path):
    out_dir = str(tmp_path)

    result = download_ps_file(run=94, camcol=1, frame=11, rerun=301, out_dir=out_dir)

    assert result is True
    assert len(mock_wget_download) == 1
    url, out = mock_wget_download[0]
    assert url.endswith('/301/94/objcs/1/psField-000094-1-0011.fit')
    assert out == out_dir
