from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from astropy.nddata import NDData, Cutout2D
from astropy.coordinates import SkyCoord
from astropy.modeling import models, fitting
from astroquery.gaia import Gaia
from photutils.psf import extract_stars
from reproject import reproject_interp
from regions import EllipseSkyRegion
from photutils import EPSFBuilder

import sys

# configure astroquery gaia
Gaia.ROW_LIMIT = 10000
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

def query_gaia(center, radius):

    r = Gaia.query_object_async(coordinate=center, radius=radius)
    r = r['ra', 'dec', 'parallax', 'phot_g_mean_mag', 'classprob_dsc_combmod_star'].copy()
    mask = np.abs(r['classprob_dsc_combmod_star']) > 0.99
    r = r[mask].copy()
    return r


class Image:

    def __init__(self, filename, datahdu=0, headerhdu=None, debug=False, units=None):

        self.filename = filename
        self.datahdu = datahdu

        with fits.open(self.filename) as hdu:
            self.data = hdu[self.datahdu].data
            self.header = hdu[self.datahdu].header
            if headerhdu is not None:
                self.main_header = hdu[headerhdu].header
            else:
                self.main_header = None

        self.wcs = WCS(self.header)
        if units is None:
            try:
                self.units = u.Unit(self.header['BUNIT'])
                self.units *= u.pix
            except ValueError:
                print('The card BUNIT does not exist.')
        else:
            self.units=units
        self.galaxy = None
        self.debug = debug
        self.psf = None

    def resample(self, header=None, pixscale=None):

        if header is None and pixscale is None:
            raise ValueError(f'One between header and pixscale must be defined')
        elif header is not None and pixscale is not None:
            raise ValueError(f'Only one between header and pixscale can be defined')

        if header is not None:
            self.data = reproject_interp((self.data, self.header), header, return_footprint=False)
            area_old = self.wcs.proj_plane_pixel_area()
            area_new = WCS(header).proj_plane_pixel_area()
            factor = area_new/area_old
            self.data *= factor
            self.wcs = WCS(header)
            self.header = header

        if pixscale is not None:
            print('Using MPDAF to resample the image')
            image = Image(filename=self.filename)
            scale = image.get_step() * 3600
            newdim_y = int(image.shape[0] * scale[0]//pixscale)
            newdim_x = int(image.shape[1] * scale[1]//pixscale)
            image = image.resample(newdim=(newdim_y, newdim_x), newstart=None,
                                   newstep=0.2, flux=True, order=3)
            self.data = image.data
            self.header = image.data_header
            self.wcs = image.wcs.wcs

    def mask_galaxy(self, center, amax, amin, pa):
        self.galaxy = EllipseSkyRegion(center, width=amin, height=amax, angle=pa)

        if self.debug:
            plt.imshow(self.data, origin='lower', vmin=0, vmax=5)
            pixel_region = self.galaxy.to_pixel(self.wcs)
            pixel_region.plot()
            plt.show()

    def get_gaia_catalog(self, center, gmin, gmax, radius=10*u.arcmin, save=True):

        gaia_cat = query_gaia(center, radius)
        mask1 = (gaia_cat['phot_g_mean_mag'] >= gmin) * (gaia_cat['phot_g_mean_mag'] <= gmax)
        print(f'Selecting stars between {gmin} and {gmax} G mag')
        gaia_cat = gaia_cat[mask1].copy()

        if self.galaxy is not None:
            coords = SkyCoord(gaia_cat['ra'], gaia_cat['dec'], unit=(u.deg, u.deg))
            mask2 = self.galaxy.contains(coords, wcs=self.wcs)
            gaia_cat = gaia_cat[~mask2].copy()

        self.stars = gaia_cat
        self.stars.write(self.filename.replace('.fits', '.stars.fits'), overwrite=True)

    def build_startable(self, coords):

        x, y = np.zeros(len(coords)), np.zeros(len(coords))

        # fitter = fitting.LevMarLSQFitter()

        #recentering the stars. Weirdly fitting a gaussian does not work. For now,
        #I'll try with identifying the max. Will see
        for i, coord in enumerate(coords):
            zoom = Cutout2D(self.data, coord, 7*u.arcsec, wcs=self.wcs)
            if not np.isfinite(zoom.data).all():
                continue
            guess = np.unravel_index(zoom.data.argmax(), zoom.data.shape)
            # model = models.Gaussian2D(np.nanmax(zoom.data), x_mean=guess[1], y_mean=guess[0],
            #                           x_stddev=5, y_stddev=5)+models.Const2D(np.nanmean(zoom.data))
            # model.theta_0.fixed=True
            # xx, yy = np.mgrid[:zoom.data.shape[1], :zoom.data.shape[0]]
            # fit = fitter(model, xx, yy, zoom.data)
            # newcoord = zoom.wcs.pixel_to_world(fit.x_mean_0, fit.y_mean_0)
            # if self.debug:
            #     plt.imshow(zoom.data, origin='lower')
            #     plt.scatter(fit.x_mean_0, fit.y_mean_0, c='r')
            #     plt.scatter(guess[1], guess[0])
            #     plt.show()
            #     print(fit)
            newcoord = zoom.wcs.pixel_to_world(guess[1], guess[0])
            newpix = self.wcs.world_to_pixel(newcoord)
            x[i] = newpix[0]
            y[i] = newpix[1]

        stars_tbl = Table()
        stars_tbl['x'] = x
        stars_tbl['y'] = y

        return stars_tbl

    def build_psf(self, center, gmin, gmax, radius=10*u.arcmin, npix=35,
                  oversampling=4, save=True, show=True):

        self.get_gaia_catalog(center, gmin, gmax, radius=radius)

        coords = SkyCoord(self.stars['ra'], self.stars['dec'], unit=(u.deg, u.deg))

        stars_tbl = self.build_startable(coords)

        nddata = NDData(data=self.data)
        stars = extract_stars(nddata, stars_tbl, size=npix)

        # removing the local background from the selected stars
        for star in stars:
            mean, median, std = sigma_clipped_stats(star.data, sigma=2.0)
            star._data = star._data-median
            if self.debug:
                plt.imshow(star._data, origin='lower')
                plt.show()

        epsf_builder = EPSFBuilder(oversampling=oversampling, progress_bar=True,
                                   center_accuracy=0.1, maxiters=50)

        # rebuilding the ePSF with the proper sampling
        epsf, fitted_stars = epsf_builder(stars)
        yy, xx = np.indices(stars[0].shape, dtype=float)
        xx = xx - stars[0].cutout_center[0]
        yy = yy - stars[0].cutout_center[1]
        new_psf = epsf.evaluate(xx, yy, 1., 0, 0)

        residuals = []
        for star in fitted_stars:
            residuals.append(star.compute_residual_image(epsf))

        residual = np.median(residuals, axis=0)

        if fitted_stars.n_good_stars > 0:
            print('\n{} stars were used to build this PSF' .format(fitted_stars.n_good_stars))
        else:
            sys.exit('No stars were used for the fit.')

        # plotting some diagnostics results
        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(1, 3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        ax3 = fig.add_subplot(gs[0, 2])
        xx, yy = np.indices(new_psf.shape)
        ax1.imshow(new_psf.data, origin='lower')
        ax2.plot_surface(xx, yy, new_psf.data)
        ax3.imshow(residual, origin='lower')
        ax1.set_title('PSF')
        ax2.set_title('PSF - 3D')
        ax3.set_title('Residuals')
        if save:
            plt.savefig(self.filename.replace('.fits', '.psf.png'), dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

        # saving the ePSF as a fits file, making sure it is normalized to 1

        psf_flux = np.sum(new_psf.data)
        if np.abs(1-psf_flux) < 0.0001:
            hdu = fits.PrimaryHDU(new_psf.data)
            self.psf = new_psf.data
        else:
            hdu = fits.PrimaryHDU(new_psf.data / psf_flux)
            self.psf = new_psf.data / psf_flux
        out = self.filename.replace('.fits', '.psf.fits')
        hdu.writeto(out, overwrite=True)

    def convert_units(self, out_units, equivalency=None):

        print(f'Updating the units from {self.units} to {out_units}')
        tmp_image = self.data * self.units
        tmp_image = tmp_image.to(out_units, equivalencies=equivalency)
        self.data = tmp_image.value
        self.units = out_units

    def open_psf(self, filename):

        with fits.open(filename) as hdu:
            psf = hdu[0].data

        self.psf = psf