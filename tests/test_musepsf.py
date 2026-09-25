import os

import numpy as np
import pytest

import astropy.units as u
from astropy.convolution import convolve_fft
from astropy.io import fits

from musepsf.image import Image
from musepsf.musepsf import MUSEImage
from musepsf.utils import moffat_kernel

MUSE_UNITS = u.erg / (u.cm * u.cm * u.second * u.AA) * 1.e-20


def _write_muse_file(write_fits, make_wcs_header, filename, data, shape, pixscale,
                      ins_mode='WFM-NOAO-N'):
    header = make_wcs_header(shape=shape, pixscale=pixscale)
    main_header = fits.Header()
    main_header['HIERARCH ESO INS MODE'] = ins_mode
    input_dir, fname = write_fits(filename, data, header, main_header=main_header, datahdu=1)
    return input_dir, fname


class TestInit:
    def test_sets_scale_region_dir_and_zeromask(self, write_fits, make_wcs_header, tmp_path):
        shape = (20, 20)
        data = np.ones(shape)
        data[0, 0] = 0.0
        data[1, 1] = np.nan
        input_dir, fname = _write_muse_file(write_fits, make_wcs_header, 'muse.fits', data,
                                             shape, pixscale=0.2)

        img = MUSEImage(fname, input_dir=input_dir, output_dir=str(tmp_path),
                         datahdu=1, headerhdu=0, units=MUSE_UNITS)

        assert img.scale == pytest.approx(0.2)
        assert os.path.isdir(os.path.join(str(tmp_path), 'regions'))
        assert img.flux_calibration_plot_dir == os.path.join(str(tmp_path), 'plots', 'flux_calibration')
        assert img.resampled_plot_dir == os.path.join(str(tmp_path), 'plots', 'resampled')
        assert img.fit_plot_dir == os.path.join(str(tmp_path), 'plots', 'fit')
        assert img.qc_plot_dir == os.path.join(str(tmp_path), 'plots', 'qc')
        for directory in (img.flux_calibration_plot_dir, img.resampled_plot_dir,
                          img.fit_plot_dir, img.qc_plot_dir):
            assert os.path.isdir(directory)
        assert img.zeromask[0, 0]
        assert img.zeromask[1, 1]  # NaN -> 0 during Image.__init__, then flagged by zeromask
        assert not img.zeromask[5, 5]


class TestGetRot:
    @pytest.mark.parametrize('rotation', [0.0, 30.0, 90.0, -45.0])
    def test_recovers_known_rotation(self, write_fits, make_wcs_header, tmp_path, rotation):
        shape = (20, 20)
        header = make_wcs_header(shape=shape, pixscale=0.2, rotation=rotation)
        main_header = fits.Header()
        main_header['HIERARCH ESO INS MODE'] = 'WFM-NOAO-N'
        input_dir, fname = write_fits('rot.fits', np.zeros(shape), header,
                                       main_header=main_header, datahdu=1)

        img = MUSEImage(fname, input_dir=input_dir, output_dir=str(tmp_path),
                         datahdu=1, headerhdu=0, units=MUSE_UNITS)

        # make_wcs_header builds the CD matrix with the opposite sign convention to the
        # one get_rot() reports the angle in -- characterized empirically, not assumed
        assert img.get_rot() == pytest.approx(-rotation, abs=1e-6)


class TestMeasurePsfValidation:
    # Regression tests for KNOWN_ISSUES.md #4: these were bare `assert` statements
    # (stripped under `python -O`, and raising the not-very-descriptive AssertionError).

    @staticmethod
    def _make_pair(write_fits, make_wcs_header, tmp_path, shape=(20, 20), pixscale=0.2):
        input_dir, fname = _write_muse_file(write_fits, make_wcs_header, 'muse.fits',
                                             np.zeros(shape), shape, pixscale)
        muse_img = MUSEImage(fname, input_dir=input_dir, output_dir=str(tmp_path),
                              datahdu=1, headerhdu=0, units=MUSE_UNITS)

        ref_header = make_wcs_header(shape=shape, pixscale=pixscale)
        ref_input_dir, ref_fname = write_fits('ref.fits', np.zeros(shape), ref_header, datahdu=0)
        ref_img = Image(ref_fname, input_dir=ref_input_dir, output_dir=str(tmp_path),
                        datahdu=0, units=MUSE_UNITS)
        return muse_img, ref_img

    def test_raises_on_unit_mismatch(self, write_fits, make_wcs_header, tmp_path):
        muse_img, ref_img = self._make_pair(write_fits, make_wcs_header, tmp_path)
        ref_img.units = u.Jy
        ref_img.psf = np.ones((5, 5))
        ref_img.psfscale = 0.2

        with pytest.raises(ValueError, match='not in the same units'):
            muse_img.measure_psf(ref_img)

    def test_raises_when_reference_psf_missing(self, write_fits, make_wcs_header, tmp_path):
        muse_img, ref_img = self._make_pair(write_fits, make_wcs_header, tmp_path)

        with pytest.raises(ValueError, match='reference PSF is missing'):
            muse_img.measure_psf(ref_img)

    def test_raises_on_psfscale_mismatch(self, write_fits, make_wcs_header, tmp_path):
        muse_img, ref_img = self._make_pair(write_fits, make_wcs_header, tmp_path)
        ref_img.psf = np.ones((5, 5))
        ref_img.psfscale = 0.4

        with pytest.raises(ValueError, match='different pixel scale'):
            muse_img.measure_psf(ref_img)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_measure_psf_recovers_injected_fwhm(write_fits, make_wcs_header, tmp_path):
    """
    End-to-end regression test for the core cross-convolution algorithm.

    Builds a synthetic "true sky" of point sources, derives a sharp "reference" image and
    a blurrier "MUSE" image by convolving that same sky with two different Moffat PSFs, and
    checks that measure_psf() recovers the FWHM used to blur the MUSE image. Star masking is
    disabled (mask_stars=False) and resampling is skipped (resample=False, both images already
    share the same synthetic WCS) to isolate the fitting algorithm itself; this is also the
    regression safety net for the Phase 4 deduplication work.
    """
    shape = (121, 121)
    pixscale = 0.2
    true_alpha = 2.8
    true_fwhm = 0.9

    rng = np.random.default_rng(1)
    true_sky = np.zeros(shape)
    n_sources = 6
    ys = rng.integers(30, shape[0] - 30, n_sources)
    xs = rng.integers(30, shape[1] - 30, n_sources)
    true_sky[ys, xs] = rng.uniform(5, 20, n_sources)

    psf_ref_kernel = moffat_kernel(0.4, true_alpha, scale=pixscale, img_size=41)
    psf_muse_kernel = moffat_kernel(true_fwhm, true_alpha, scale=pixscale, img_size=41)

    reference_data = convolve_fft(true_sky, psf_ref_kernel)
    muse_data = convolve_fft(true_sky, psf_muse_kernel)

    muse_input_dir, muse_fname = _write_muse_file(write_fits, make_wcs_header, 'muse.fits',
                                                   muse_data, shape, pixscale)
    ref_header = make_wcs_header(shape=shape, pixscale=pixscale)
    ref_input_dir, ref_fname = write_fits('ref.fits', reference_data, ref_header, datahdu=0)

    muse_img = MUSEImage(muse_fname, input_dir=muse_input_dir, output_dir=str(tmp_path),
                          datahdu=1, headerhdu=0, units=MUSE_UNITS)
    ref_img = Image(ref_fname, input_dir=ref_input_dir, output_dir=str(tmp_path),
                     datahdu=0, units=MUSE_UNITS)
    ref_img.psf = psf_ref_kernel.array
    ref_img.psfscale = pixscale

    muse_img.measure_psf(ref_img, fit_alpha=False, plot=False, save=False, show=False,
                          mask_stars=False, resample=False, edge=10, fwhm0=0.5)

    assert muse_img.best_fit[0] == pytest.approx(true_fwhm, rel=0.05)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_measure_psf_with_qc_populates_confidence_and_qc_flag(write_fits, make_wcs_header, tmp_path):
    """
    End-to-end test for measure_psf(qc=True) (Phase 2 of IMPLEMENTATION_PLAN.md): same synthetic
    setup as test_measure_psf_recovers_injected_fwhm, checking that the rmse_test()-based QC
    machinery runs on a real fit and produces sensible output ("clean pass" case).

    Before KNOWN_ISSUES.md #14 was fixed, this exact synthetic pair needed offset=True here: with
    offset=False, the fit (via the then-even-sized internal Moffat kernel) picked up a spurious
    ~0.5 pixel registration offset that rmse_test()'s grid correctly flagged as "minimum not at
    the best fit". That was itself a genuine demonstration the QC check works, just not what this
    test is checking, so offset=True was used to remove the confound. #14 fixes the root cause
    (verified: offset=False now also passes QC cleanly on this setup), but offset=True is kept
    here anyway for the extra coverage of the offset-fitting + 3-parameter covariance path.
    """
    shape = (121, 121)
    pixscale = 0.2
    true_alpha = 2.8
    true_fwhm = 0.9

    rng = np.random.default_rng(1)
    true_sky = np.zeros(shape)
    n_sources = 6
    ys = rng.integers(30, shape[0] - 30, n_sources)
    xs = rng.integers(30, shape[1] - 30, n_sources)
    true_sky[ys, xs] = rng.uniform(5, 20, n_sources)

    psf_ref_kernel = moffat_kernel(0.4, true_alpha, scale=pixscale, img_size=41)
    psf_muse_kernel = moffat_kernel(true_fwhm, true_alpha, scale=pixscale, img_size=41)

    reference_data = convolve_fft(true_sky, psf_ref_kernel)
    muse_data = convolve_fft(true_sky, psf_muse_kernel)

    muse_input_dir, muse_fname = _write_muse_file(write_fits, make_wcs_header, 'muse.fits',
                                                   muse_data, shape, pixscale)
    ref_header = make_wcs_header(shape=shape, pixscale=pixscale)
    ref_input_dir, ref_fname = write_fits('ref.fits', reference_data, ref_header, datahdu=0)

    muse_img = MUSEImage(muse_fname, input_dir=muse_input_dir, output_dir=str(tmp_path),
                          datahdu=1, headerhdu=0, units=MUSE_UNITS)
    ref_img = Image(ref_fname, input_dir=ref_input_dir, output_dir=str(tmp_path),
                     datahdu=0, units=MUSE_UNITS)
    ref_img.psf = psf_ref_kernel.array
    ref_img.psfscale = pixscale

    muse_img.measure_psf(ref_img, fit_alpha=False, plot=False, save=False, show=False,
                          mask_stars=False, resample=False, offset=True, edge=10, fwhm0=0.5,
                          qc=True, num_points=5, delta_off=1, delta_fwhm=0.3)

    lo, hi = muse_img.qc_confidence['fwhm']
    assert lo <= 0 <= hi
    assert muse_img.qc.passed
    assert muse_img.qc.reasons == []


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_measure_psf_qc_saves_qc_summary_plot_when_requested(write_fits, make_wcs_header, tmp_path):
    # measure_psf(qc=True, plot=True, save=True) should save the dedicated QC plot
    # (utils.plot_qc_summary(), see MUSEImage.measure_psf's qc docstring) alongside the numeric
    # qc/qc_confidence attributes -- same synthetic setup as the other qc=True tests above.
    shape = (121, 121)
    pixscale = 0.2
    true_alpha = 2.8
    true_fwhm = 0.9

    rng = np.random.default_rng(1)
    true_sky = np.zeros(shape)
    n_sources = 6
    ys = rng.integers(30, shape[0] - 30, n_sources)
    xs = rng.integers(30, shape[1] - 30, n_sources)
    true_sky[ys, xs] = rng.uniform(5, 20, n_sources)

    psf_ref_kernel = moffat_kernel(0.4, true_alpha, scale=pixscale, img_size=41)
    psf_muse_kernel = moffat_kernel(true_fwhm, true_alpha, scale=pixscale, img_size=41)

    reference_data = convolve_fft(true_sky, psf_ref_kernel)
    muse_data = convolve_fft(true_sky, psf_muse_kernel)

    muse_input_dir, muse_fname = _write_muse_file(write_fits, make_wcs_header, 'muse.fits',
                                                   muse_data, shape, pixscale)
    ref_header = make_wcs_header(shape=shape, pixscale=pixscale)
    ref_input_dir, ref_fname = write_fits('ref.fits', reference_data, ref_header, datahdu=0)

    muse_img = MUSEImage(muse_fname, input_dir=muse_input_dir, output_dir=str(tmp_path),
                          datahdu=1, headerhdu=0, units=MUSE_UNITS)
    ref_img = Image(ref_fname, input_dir=ref_input_dir, output_dir=str(tmp_path),
                     datahdu=0, units=MUSE_UNITS)
    ref_img.psf = psf_ref_kernel.array
    ref_img.psfscale = pixscale

    muse_img.measure_psf(ref_img, fit_alpha=False, plot=True, save=True, show=False,
                          mask_stars=False, resample=False, offset=True, edge=10, fwhm0=0.5,
                          qc=True, num_points=5, delta_off=1, delta_fwhm=0.3)

    assert os.path.isfile(os.path.join(str(tmp_path), 'plots', 'qc', 'muse_qc.png'))


def test_measure_psf_qc_raises_with_oversample(write_fits, make_wcs_header, tmp_path):
    # rmse_test() has no oversampling support, so qc=True + oversample>1 is not supported
    # (see NotImplementedError in MUSEImage.measure_psf) rather than silently computing a QC
    # grid in the wrong units.
    shape = (41, 41)
    pixscale = 0.2
    input_dir, fname = _write_muse_file(write_fits, make_wcs_header, 'muse.fits',
                                         np.zeros(shape), shape, pixscale)
    muse_img = MUSEImage(fname, input_dir=input_dir, output_dir=str(tmp_path),
                          datahdu=1, headerhdu=0, units=MUSE_UNITS)

    ref_header = make_wcs_header(shape=shape, pixscale=pixscale)
    ref_input_dir, ref_fname = write_fits('ref.fits', np.zeros(shape), ref_header, datahdu=0)
    ref_img = Image(ref_fname, input_dir=ref_input_dir, output_dir=str(tmp_path),
                    datahdu=0, units=MUSE_UNITS)
    ref_img.psf = np.ones((5, 5)) / 25
    ref_img.psfscale = pixscale

    with pytest.raises(NotImplementedError, match='oversample'):
        muse_img.measure_psf(ref_img, mask_stars=False, resample=False, qc=True, oversample=2)
