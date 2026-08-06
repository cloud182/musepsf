import matplotlib
matplotlib.use('Agg')  # no display available / needed while running the test suite

import numpy as np
import pytest

from astropy.io import fits
from astropy.wcs import WCS


@pytest.fixture
def make_wcs_header():
    """Factory returning a synthetic FITS header with a simple tangent-plane WCS."""

    def _make(shape=(50, 50), pixscale=0.2, crval=(150.0, 2.0), rotation=0.0,
              bunit='erg cm-2 s-1 Angstrom-1'):
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        wcs.wcs.crpix = [shape[1] / 2, shape[0] / 2]
        wcs.wcs.crval = list(crval)

        scale = pixscale / 3600.0
        theta = np.radians(rotation)
        wcs.wcs.cd = np.array([
            [-scale * np.cos(theta), scale * np.sin(theta)],
            [scale * np.sin(theta), scale * np.cos(theta)],
        ])

        header = wcs.to_header()
        header['BUNIT'] = bunit
        # reproject needs an explicit shape when reprojecting onto this header
        header['NAXIS'] = 2
        header['NAXIS1'] = shape[1]
        header['NAXIS2'] = shape[0]
        return header

    return _make


@pytest.fixture
def write_fits(tmp_path):
    """Factory writing a synthetic FITS file, returning (input_dir, filename)."""

    def _write(filename, data, header, main_header=None, datahdu=0):
        if datahdu == 0:
            hdul = fits.HDUList([fits.PrimaryHDU(data=data, header=header)])
        else:
            hdus = [fits.PrimaryHDU(header=main_header)]
            hdus += [fits.ImageHDU() for _ in range(datahdu - 1)]
            hdus.append(fits.ImageHDU(data=data, header=header))
            hdul = fits.HDUList(hdus)
        hdul.writeto(tmp_path / filename, overwrite=True)
        return str(tmp_path), filename

    return _write


@pytest.fixture
def write_psf_fits(tmp_path):
    """Factory writing a synthetic PSF FITS file (with PSFSCALE header) and returning its path."""

    def _write(filename, psf_data, psfscale=0.2):
        hdu = fits.PrimaryHDU(psf_data)
        hdu.header['PSFSCALE'] = psfscale
        path = tmp_path / filename
        hdu.writeto(path, overwrite=True)
        return str(path)

    return _write


@pytest.fixture
def mock_gaia_query(monkeypatch):
    """Install a fake Gaia.query_object_async returning a canned table.

    query_gaia() imports astroquery.gaia.Gaia lazily (importing it touches the
    Gaia archive as a side effect), so the fake is installed on the real
    astroquery class rather than on a module-level name in musepsf.utils.
    """

    def _install(table):
        def _fake_query_object_async(coordinate, radius):
            return table
        monkeypatch.setattr('astroquery.gaia.Gaia.query_object_async', _fake_query_object_async)

    return _install


@pytest.fixture
def mock_wget_download(monkeypatch):
    """Install a fake wget.download that records calls instead of hitting the network."""

    calls = []

    def _fake_download(url, out=None):
        calls.append((url, out))
        return None

    monkeypatch.setattr('musepsf.utils.wget.download', _fake_download)
    return calls
