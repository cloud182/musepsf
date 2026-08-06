import os

import numpy as np
import pytest

import astropy.units as u
from astropy.io import fits
from astropy.modeling import models
from astropy.table import Table
from astropy.wcs import WCS

import musepsf.image as image_mod
from musepsf.image import Image


def _make_image(write_fits, make_wcs_header, tmp_path, filename, data, shape=None,
                 pixscale=0.2, units=None, **header_kwargs):
    shape = shape or data.shape
    header = make_wcs_header(shape=shape, pixscale=pixscale, **header_kwargs)
    input_dir, fname = write_fits(filename, data, header, datahdu=0)
    return Image(fname, input_dir=input_dir, output_dir=str(tmp_path), datahdu=0, units=units)


class TestInit:
    def test_parses_data_header_wcs_units_and_cleans_non_finite(self, write_fits, make_wcs_header, tmp_path):
        shape = (30, 40)
        data = np.arange(30 * 40, dtype=float).reshape(shape)
        data[0, 0] = np.nan
        data[1, 1] = np.inf

        img = _make_image(write_fits, make_wcs_header, tmp_path, 'basic.fits', data, shape=shape)

        assert img.data.shape == shape
        assert img.data[0, 0] == 0.0
        assert np.isfinite(img.data).all()
        assert img.units == u.Unit('erg cm-2 s-1 Angstrom-1') * u.pix
        assert isinstance(img.wcs, WCS)
        assert img.galaxy is None
        assert img.psf is None

    def test_raises_value_error_when_bunit_missing_and_no_units_given(self, write_fits,
                                                                        make_wcs_header, tmp_path):
        # Regression test for KNOWN_ISSUES.md #2: a missing BUNIT card used to be silently
        # swallowed (just a print()), leaving self.units undefined until first use.
        shape = (10, 10)
        header = make_wcs_header(shape=shape, pixscale=0.2)
        del header['BUNIT']
        input_dir, fname = write_fits('nobunit.fits', np.zeros(shape), header, datahdu=0)

        with pytest.raises(ValueError, match='BUNIT'):
            Image(fname, input_dir=input_dir, output_dir=str(tmp_path), datahdu=0)


def test_mask_galaxy_sets_expected_region(write_fits, make_wcs_header, tmp_path):
    shape = (60, 60)
    img = _make_image(write_fits, make_wcs_header, tmp_path, 'm.fits', np.zeros(shape))

    center = img.wcs.pixel_to_world(30, 30)
    img.mask_galaxy(center, amax=10 * u.arcsec, amin=5 * u.arcsec, pa=45 * u.deg)

    assert img.galaxy is not None
    assert img.galaxy.center.separation(center) < 0.001 * u.arcsec


class TestGetGaiaCatalog:
    def test_filters_by_magnitude_edge_margin_and_galaxy_region(self, write_fits, make_wcs_header,
                                                                  tmp_path, monkeypatch):
        shape = (100, 100)
        img = _make_image(write_fits, make_wcs_header, tmp_path, 'g.fits', np.zeros(shape))

        # four synthetic Gaia sources, each isolating one filter in get_gaia_catalog:
        # good_px passes everything; edge_px is too close to the border; faint_px fails
        # the G-mag cut; galaxy_px sits inside the masked galaxy region. All four are kept
        # more than 3.5 arcsec apart so remove_close_stars() does not interfere.
        good_px, edge_px, faint_px, galaxy_px = (25, 25), (5, 5), (75, 25), (25, 75)
        xs = [p[0] for p in (good_px, edge_px, faint_px, galaxy_px)]
        ys = [p[1] for p in (good_px, edge_px, faint_px, galaxy_px)]
        sky = img.wcs.pixel_to_world(xs, ys)

        table = Table({
            'ra': sky.ra.deg,
            'dec': sky.dec.deg,
            'parallax': [1.0] * 4,
            'phot_g_mean_mag': [15.0, 15.0, 25.0, 15.0],
            'classprob_dsc_combmod_star': [0.999] * 4,
        })
        monkeypatch.setattr(image_mod, 'query_gaia', lambda center, radius: table)

        galaxy_center = img.wcs.pixel_to_world(*galaxy_px)
        img.mask_galaxy(galaxy_center, amax=3 * u.arcsec, amin=3 * u.arcsec, pa=0 * u.deg)

        field_center = img.wcs.pixel_to_world(50, 50)
        img.get_gaia_catalog(field_center, gmin=10, gmax=20, radius=5 * u.arcmin)

        assert len(img.stars) == 1
        assert img.stars['ra'][0] == pytest.approx(sky.ra.deg[0])

    def test_without_mask_galaxy_keeps_stars_and_warns(self, write_fits, make_wcs_header,
                                                         tmp_path, monkeypatch):
        # Regression test for KNOWN_ISSUES.md #7: get_gaia_catalog() used to crash with
        # UnboundLocalError if mask_galaxy() had never been called. Now the galaxy
        # exclusion is skipped (nothing is excluded) and a warning is raised instead.
        shape = (60, 60)
        img = _make_image(write_fits, make_wcs_header, tmp_path, 'g2.fits', np.zeros(shape))

        table = Table({
            'ra': [img.wcs.pixel_to_world(30, 30).ra.deg],
            'dec': [img.wcs.pixel_to_world(30, 30).dec.deg],
            'parallax': [1.0],
            'phot_g_mean_mag': [15.0],
            'classprob_dsc_combmod_star': [0.999],
        })
        monkeypatch.setattr(image_mod, 'query_gaia', lambda center, radius: table)

        center = img.wcs.pixel_to_world(30, 30)
        with pytest.warns(UserWarning, match='mask_galaxy'):
            img.get_gaia_catalog(center, gmin=10, gmax=20)

        assert len(img.stars) == 1


def test_build_startable_recenters_to_local_peak(write_fits, make_wcs_header, tmp_path):
    shape = (80, 80)
    yy, xx = np.indices(shape)
    star = models.Gaussian2D(amplitude=100, x_mean=42, y_mean=44, x_stddev=1.2, y_stddev=1.2)
    data = star(xx, yy) + np.random.default_rng(0).normal(scale=0.01, size=shape)

    img = _make_image(write_fits, make_wcs_header, tmp_path, 'bs.fits', data)
    center = img.wcs.pixel_to_world(40, 40)
    img.mask_galaxy(center, amax=1 * u.arcsec, amin=1 * u.arcsec, pa=0 * u.deg)

    # build_startable() expects a masked array (its real caller, build_psf(), gets one
    # from MPDAF's resample() -- see KNOWN_ISSUES.md #10)
    masked_data = np.ma.masked_array(img.data, mask=np.zeros_like(img.data, dtype=bool))
    guess_coord = img.wcs.pixel_to_world(40, 40)

    stars_tbl = img.build_startable([guess_coord], masked_data, img.wcs, save=False, show=False)

    assert stars_tbl['x'][0] == pytest.approx(42, abs=0.1)
    assert stars_tbl['y'][0] == pytest.approx(44, abs=0.1)


def test_build_startable_works_without_mask_galaxy(write_fits, make_wcs_header, tmp_path):
    # Regression test for KNOWN_ISSUES.md #7: build_startable() used to crash with
    # AttributeError ('NoneType' has no 'to_pixel') if mask_galaxy() had never been
    # called, because it unconditionally plotted self.galaxy on the diagnostic figure.
    shape = (80, 80)
    yy, xx = np.indices(shape)
    star = models.Gaussian2D(amplitude=100, x_mean=42, y_mean=44, x_stddev=1.2, y_stddev=1.2)
    data = star(xx, yy)

    img = _make_image(write_fits, make_wcs_header, tmp_path, 'bs2.fits', data)
    assert img.galaxy is None

    masked_data = np.ma.masked_array(img.data, mask=np.zeros_like(img.data, dtype=bool))
    guess_coord = img.wcs.pixel_to_world(40, 40)

    stars_tbl = img.build_startable([guess_coord], masked_data, img.wcs, save=False, show=False)

    assert stars_tbl['x'][0] == pytest.approx(42, abs=0.1)


def test_build_startable_works_with_plain_ndarray_data(write_fits, make_wcs_header, tmp_path):
    # Regression test for KNOWN_ISSUES.md #10: build_startable() used to crash with
    # AttributeError ('numpy.ndarray' object has no attribute 'mask') when handed a
    # plain array instead of a numpy.ma.MaskedArray (e.g. build_psf(pixscale=None)).
    shape = (80, 80)
    yy, xx = np.indices(shape)
    star = models.Gaussian2D(amplitude=100, x_mean=42, y_mean=44, x_stddev=1.2, y_stddev=1.2)
    data = star(xx, yy)

    img = _make_image(write_fits, make_wcs_header, tmp_path, 'bs3.fits', data)
    guess_coord = img.wcs.pixel_to_world(40, 40)

    assert not isinstance(img.data, np.ma.MaskedArray)
    stars_tbl = img.build_startable([guess_coord], img.data, img.wcs, save=False, show=False)

    assert stars_tbl['x'][0] == pytest.approx(42, abs=0.1)
    assert stars_tbl['y'][0] == pytest.approx(44, abs=0.1)


def test_resample_to_header_rescales_flux_and_shape(write_fits, make_wcs_header, tmp_path):
    shape = (50, 50)
    img = _make_image(write_fits, make_wcs_header, tmp_path, 'src.fits', np.full(shape, 3.0),
                       pixscale=0.2)

    coarser_header = make_wcs_header(shape=(25, 25), pixscale=0.4)
    img.resample(header=coarser_header)

    assert img.data.shape == (25, 25)
    # pixel area quadruples (0.4/0.2)**2, and resample() rescales flux by area_new/area_old
    assert np.nanmean(img.data[5:-5, 5:-5]) == pytest.approx(3.0 * 4, rel=1e-6)


def test_resample_to_pixscale_uses_requested_scale(write_fits, make_wcs_header, tmp_path):
    # Regression test for KNOWN_ISSUES.md #1: resample(pixscale=...) used to hardcode
    # newstep=0.2 regardless of the requested pixscale. Using 0.1 here (not the old
    # hardcoded 0.2) would have failed before the fix.
    shape = (60, 60)
    img = _make_image(write_fits, make_wcs_header, tmp_path, 'mp.fits',
                       np.random.default_rng(0).normal(loc=5, size=shape), pixscale=0.2)

    img.resample(pixscale=0.1)

    new_scale = img.wcs.proj_plane_pixel_scales()[0].to(u.arcsec).value
    assert new_scale == pytest.approx(0.1, abs=1e-6)


def test_convert_units_scales_data(write_fits, make_wcs_header, tmp_path):
    data = np.full((10, 10), 2.0)
    img = _make_image(write_fits, make_wcs_header, tmp_path, 'u.fits', data, units=u.Jy)

    img.convert_units(u.mJy)

    assert img.data == pytest.approx(data * 1000)
    assert img.units == u.mJy


def test_open_psf_reads_back_psf_scale_and_fwhm(write_fits, make_wcs_header, write_psf_fits, tmp_path):
    img = _make_image(write_fits, make_wcs_header, tmp_path, 'holder.fits', np.zeros((10, 10)))

    size = 31
    model = models.Moffat2D(amplitude=1, x_0=size // 2, y_0=size // 2, gamma=4, alpha=2.5)
    yy, xx = np.indices((size, size))
    psf_data = model(xx, yy)
    psf_data /= psf_data.sum()
    psf_path = write_psf_fits('ref.psf.fits', psf_data, psfscale=0.4)

    img.open_psf(psf_path, mode='moffat')

    assert img.psf == pytest.approx(psf_data)
    assert img.psfscale == 0.4
    assert img.psf_fwhm == pytest.approx(model.fwhm, rel=1e-6)


def test_build_psf_with_stars_file_end_to_end(write_fits, make_wcs_header, tmp_path):
    # Bypassing Gaia (stars_file=...) and resampling (pixscale=None) exercises the full
    # extract_stars -> EPSFBuilder -> normalization -> fit_2D_psf pipeline offline.
    shape = (100, 100)
    yy, xx = np.indices(shape)
    star_positions = [(25, 25), (70, 30), (50, 75)]
    data = np.zeros(shape)
    for x0, y0 in star_positions:
        star = models.Gaussian2D(amplitude=200, x_mean=x0, y_mean=y0, x_stddev=1.5, y_stddev=1.5)
        data += star(xx, yy)
    data += np.random.default_rng(3).normal(scale=0.05, size=shape)

    img = _make_image(write_fits, make_wcs_header, tmp_path, 'bp.fits', data)

    sky = [img.wcs.pixel_to_world(x, y) for x, y in star_positions]
    stars_file = os.path.join(str(tmp_path), 'stars.dat')
    with open(stars_file, 'w') as f:
        for s in sky:
            f.write(f'{s.ra.deg} {s.dec.deg}\n')

    center = img.wcs.pixel_to_world(50, 50)
    img.build_psf(center, gmin=10, gmax=20, stars_file=stars_file, pixscale=None,
                  npix=15, oversampling=2, save=False, show=False)

    assert img.psf.sum() == pytest.approx(1.0, abs=1e-6)
    assert img.psf_fwhm > 0
    assert os.path.exists(os.path.join(str(tmp_path), 'bp.psf.fits'))


class TestRecoverSdssPsf:
    def test_wires_normalized_psf_and_writes_file(self, write_fits, make_wcs_header, tmp_path,
                                                    monkeypatch):
        img = _make_image(write_fits, make_wcs_header, tmp_path, 'sdss.fits', np.zeros((40, 40)))

        fake_psf = np.zeros((21, 21))
        fake_psf[10, 10] = 2.0  # deliberately not normalized to 1
        monkeypatch.setattr(image_mod, 'create_sdss_psf',
                             lambda data, hdr, out_dir, pixscale=0.2: fake_psf.copy())

        img.recover_SDSS_PSF(save=False, show=False, pixscale=None)

        assert img.psf.sum() == pytest.approx(1.0)
        assert img.psfscale == pytest.approx(0.2)
        assert img.psf_fwhm is None  # KNOWN_ISSUES.md #9: no model fit for this PSF source
        assert os.path.exists(os.path.join(str(tmp_path), 'sdss.psf.fits'))
        with fits.open(os.path.join(str(tmp_path), 'sdss.psf.fits')) as hdu:
            assert hdu[0].header['PSFSCALE'] == pytest.approx(0.2)

    def test_stores_array_when_psf_already_normalized(self, write_fits, make_wcs_header,
                                                         tmp_path, monkeypatch):
        # Regression test for KNOWN_ISSUES.md #8: `self.psf = new_psf.data` used to store a
        # raw memoryview (ndarray.data is not "the array") whenever the psf_flux branch took
        # the "already normalized" path.
        img = _make_image(write_fits, make_wcs_header, tmp_path, 'sdss2.fits', np.zeros((40, 40)))

        fake_psf = np.zeros((21, 21))
        fake_psf[10, 10] = 1.0  # sums to exactly 1 -> takes the previously-buggy branch
        monkeypatch.setattr(image_mod, 'create_sdss_psf',
                             lambda data, hdr, out_dir, pixscale=0.2: fake_psf.copy())

        img.recover_SDSS_PSF(save=False, show=False, pixscale=None)

        assert isinstance(img.psf, np.ndarray)
        assert img.psf == pytest.approx(fake_psf)
        assert img.psf_fwhm is None  # KNOWN_ISSUES.md #9: no model fit for this PSF source
