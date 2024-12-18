from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from astropy.io import fits, ascii
from astropy.wcs import WCS
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from astropy.nddata import NDData, Cutout2D
from astropy.nddata.utils import NoOverlapError
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm
from astroquery.gaia import Gaia
from photutils.psf import extract_stars
from reproject import reproject_interp
from regions import EllipseSkyRegion
from photutils.psf import EPSFBuilder
from mpdaf.obj import Image as MPDAFImage

import sys
import os

from .utils import query_gaia, create_sdss_psf

class Image:
    """
    Basic class to manage images.

    Attributes:
        filename (str):
            name of the input file
        inpit_dir (str):
            path of the input file
        output_dir (str):
            path where to save the output files
        data (np.ndarray):
            data array
        header (astropy.header):
            header associated to the data
        main_header (astropy.header, None):
            main file header if present.
        wcs (astropy.wcs.WCS):
            wcs information associated tot he data
        units (astropy.units):
            units of measurement associated to the image
        galaxy (regions.shapes.ellipse.EllipseSkyRegion):
            elliptical region used to mask the galaxy when recovering the gaia stars
        debug (bool):
            If true, additional plots will be produced.
        psf (np.ndarray):
            array containing the PSF of the image
        stars (astropy.table.Table):
            Table containing the position of the stars used to measure the PSF.

    Methods:
        resample(header=None, pixscale=None):
            Resample the image to match a specific resolution or a specific header.
        mask_galaxy(center, amax, amin, pa):
            Mask the area covered by the galaxy when looking for gaia stars.
        get_gaia_catalog(center, gmin, gmax, radius=10*u.arcmin, save=True):
            Query the Gaia Catalog to identify stars in the field of the galaxy.
        build_startable(coords):
            Refine the position of the stars and build the star table that will be feed to the
            ePSF builder
        build_psf(center, gmin, gmax, radius=10*u.arcmin, npix=35,
                  oversampling=4, save=True, show=True):
            Build the ePSF of the considered image.
        convert_units(out_units, equivalency=None):
            Convert the unit of measurement of the image.
        open_psf(filename):
            Open the file containing the PSF of the image.
    """

    def __init__(self, filename, input_dir='./', output_dir='./',
                 datahdu=0, headerhdu=None, debug=False, units=None):

        """
        Init method of the class

        Args:
            filename (str):
                name of the file containing the image
            input_dir (str, optional):
                Location of the input file. Defaults to './'.
            output_dir (str, optional):
                Location where to save the output files. Defaults to './'.
            datahdu (int, optional):
                HDU containing the data. Defaults to 0.
            headerhdu (int, None, optional):
                HDU containing the main header. Defaults to None.
            debug (bool, optional):
                If True, several diagnostic plots will be produced. Defaults to False.
            units (astropy.units, None, optional):
                Units of the data extension. Defaults to None.
        """


        self.filename = filename
        self.input_dir = input_dir
        self.output_dir = output_dir

        with fits.open(os.path.join(self.input_dir, self.filename)) as hdu:
            self.data = hdu[datahdu].data
            self.header = hdu[datahdu].header
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
        self.stars = None

    def __shape__(self):
        return self.data.shape

    def resample(self, header=None, pixscale=None, inplace=True):
        """
        Resample the image to match a specific resolution or a specific header.

        Args:
            header (astropy.header, optional):
                Reference header to use for the reprojection. Defaults to None.
            pixscale (float, optional):
                Target pixel scale of the resampling. Defaults to None.
            inplace (bool, optional):
                if true, the attributes of the curtrent object are modified. If false,
                the data, wcs and header of the resampled image are returned in output.
                Defaults to True.

        Returns:
            np.ndarray:
                resampled image
            astropy.wcs.WCS:
                WCS of the resampled image
            astropy.io.fits.header:
                header of theresampled image

        Raises:
            ValueError: raised if both parameters or if no parameter is provided
        """

        if header is None and pixscale is None:
            raise ValueError(f'One between header and pixscale must be defined')
        elif header is not None and pixscale is not None:
            raise ValueError(f'Only one between header and pixscale can be defined')

        if header is not None:
            self.data = reproject_interp((self.data, self.header), header, return_footprint=False)
            area_old = self.wcs.proj_plane_pixel_area()
            area_new = WCS(header).proj_plane_pixel_area()
            factor = area_new/area_old

            if inplace:
                self.data *= factor
                self.wcs = WCS(header)
                self.header = header
            else:
                return self.data * factor, WCS(header), header


        if pixscale is not None:
            print('Using MPDAF to resample the image')
            image = MPDAFImage(filename=os.path.join(self.input_dir, self.filename))
            image.crop()
            scale = image.get_step() * 3600
            newdim_y = int(image.shape[0] * scale[0]//pixscale)
            newdim_x = int(image.shape[1] * scale[1]//pixscale)
            image = image.resample(newdim=(newdim_y, newdim_x), newstart=None,
                                   newstep=0.2, flux=True, order=3)
            if inplace:
                self.data = image.data
                self.header = image.data_header
                self.wcs = image.wcs.wcs
            else:
                return image.data, image.wcs.wcs, image.data_header

    def mask_galaxy(self, center, amax, amin, pa):
        """
        Mask the area covered by the galaxy when looking for gaia stars.
        The masked region is an ellipse.

        Args:
            center (SkyXCoord):
                Center of the region to mask
            amax (u.arcmin):
                Major axis of the ellipse.
            amin (u.arcmin):
                Minor axis of the ellipse
            pa (u.deg):
                Position angle of the ellipse. Counted from North, counter-clockwise
        """
        self.galaxy = EllipseSkyRegion(center, width=2*amin, height=2*amax, angle=pa)

        if self.debug:
            plt.imshow(self.data, origin='lower', vmin=0, vmax=5)
            pixel_region = self.galaxy.to_pixel(self.wcs)
            pixel_region.plot()
            plt.show()

    def get_gaia_catalog(self, center, gmin, gmax, radius=10*u.arcmin, save=True, show=False):
        """
        Query the Gaia Catalog to identify stars in the field of the galaxy.

        Args:
            center (SkyCoord):
                Center of the image.
            gmin (float):
                Minimum magnitude to consider.
            gmax (float):
                Maximum magnitude to consider.
            radius (u.arcmin, optional):
                Radius to search for the stars. Defaults to 10*u.arcmin.
            save (bool, optional):
                Save the catalog of stars. Defaults to True.
        """

        gaia_cat = query_gaia(center, radius)
        mask1 = (gaia_cat['phot_g_mean_mag'] >= gmin) * (gaia_cat['phot_g_mean_mag'] <= gmax)
        print(f'Selecting stars between {gmin} and {gmax} G mag')
        gaia_cat = gaia_cat[mask1].copy()

        coords = SkyCoord(gaia_cat['ra'], gaia_cat['dec'], unit=(u.deg, u.deg))
        inside = np.array([True if self.wcs.footprint_contains(coord) else False for coord in coords])

        if self.galaxy is not None:
            mask2 = self.galaxy.contains(coords, wcs=self.wcs)

        gaia_cat = gaia_cat[inside*(~mask2)].copy()
        self.stars = gaia_cat

        # plotting some diagnostics results
        fig, ax = plt.subplots(1, 1, figsize=(14, 14), subplot_kw={'projection': self.wcs})
        norm = simple_norm(self.data, 'log', percent=99.9)
        ax.imshow(self.data, norm=norm)
        ax.scatter(self.stars['ra'], self.stars['dec'], transform=ax.get_transform('world'),
                   s=80, facecolors='none', edgecolors='r')
        pixel_region = self.galaxy.to_pixel(self.wcs)
        pixel_region.plot(ax=ax)
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        if save:
            outname = os.path.join(self.output_dir, self.filename.replace('.fits', '.stars.png'))
            plt.savefig(outname, dpi=300)
        if show:
            plt.show()
        else:
            plt.close()



    def build_startable(self, coords, data, wcs):

        """
        Refine the position of the stars and build the star table that will be feed to the
        ePSF builder

        Args:
            coords (list):
                list of coordinates of the selected stars.
            data (np.ndarray):
                data array.
            wcs (astropy.wcs.WCS):
                wcs associated to the data array.

        Returns:
            astropy.table.Table:
                Astropy table with the x and y coordinates of the selected stars.
        """

        x, y = [], []

        # fitter = fitting.LevMarLSQFitter()

        #recentering the stars. Weirdly fitting a gaussian does not work. For now,
        #I'll try with identifying the max. Will see
        for i, coord in enumerate(coords):
            try:
                zoom = Cutout2D(data, coord, 7*u.arcsec, wcs=wcs)
            except NoOverlapError:
                continue
            if not np.isfinite(zoom.data).all():
                continue
            if zoom.data.mask.sum() >= 5:
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
            newpix = wcs.world_to_pixel(newcoord)
            x.append(newpix[0])
            y.append(newpix[1])

        stars_tbl = Table()
        stars_tbl['x'] = x
        stars_tbl['y'] = y

        return stars_tbl

    def build_psf(self, center, gmin, gmax, stars_file=None, radius=10*u.arcmin, npix=35, pixscale=0.2,
                  oversampling=4, save=True, show=True):
        """
        Build the ePSF of the considered image. Extracted from the EPSFBuilder tutorial

        Args:
            center (SkyCoord):
                Center of the considered field.
            gmin (float):
                Minimum magnitude to consider.
            gmax (float):
                Maximum magnitude to consider.
            radius (u.arcmin, optional):
                Radius to search for the stars. Defaults to 10*u.arcmin.
            npix (int, optional):
                Number of pixels to use to extract the cutouts of the stars. Defaults to 35.
            pixscale (float, optional):
                pixelscale to use to resample the data array before computing the ePSF. If None,
                no resampling is applied. Defaults to 0.2.
            oversampling (int, optional):
                Oversampling factor for the EPSF builder. Defaults to 4.
            save (bool, optional):
                Save the output plots. Defaults to True.
            show (bool, optional):
                Show the output plots. Defaults to True.
        """

        if pixscale is not None:
            data, wcs, _ = self.resample(pixscale=pixscale, inplace=False)
            self.psfscale = pixscale
        else:
            data = self.data
            wcs = self.wcs
            header = self.header
            self.psfscale = np.around(self.wcs.proj_plane_pixel_scales()[0].to(u.arcsec).value, 3)

        if stars_file is None:
            self.get_gaia_catalog(center, gmin, gmax, radius=radius)
            self.stars['ra', 'dec'].write(os.path.join(self.output_dir, self.filename.replace('.fits', '.stars.dat')),
                            format='ascii.no_header', overwrite=True)
        else:
            print(f'Using {stars_file}')
            self.stars = ascii.read(stars_file, format='no_header', names=['ra', 'dec'])

        coords = SkyCoord(self.stars['ra'], self.stars['dec'], unit=(u.deg, u.deg))
        stars_tbl = self.build_startable(coords, data, wcs)

        nddata = NDData(data=data)
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

        self.plot_psf(new_psf.data, residual=residual, save=save, show=show)

        # saving the ePSF as a fits file, making sure it is normalized to 1

        psf_flux = np.sum(new_psf.data)
        if np.abs(1-psf_flux) < 0.0001:
            hdu = fits.PrimaryHDU(new_psf.data)
            self.psf = new_psf.data
        else:
            hdu = fits.PrimaryHDU(new_psf.data / psf_flux)
            self.psf = new_psf.data / psf_flux
        hdu.header['PSFSCALE'] = self.psfscale
        out = os.path.join(self.output_dir, self.filename.replace('.fits', '.psf.fits'))
        hdu.writeto(out, overwrite=True)


    def plot_psf(self, data, residual=None, save=True, show=False):
      # plotting some diagnostics results
        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(1, 3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        ax3 = fig.add_subplot(gs[0, 2])
        xx, yy = np.indices(data.shape)
        ax1.imshow(data, origin='lower')
        ax2.plot_surface(xx, yy, data)

        if residual is None:
            residual = np.zeros_like(data)

        ax3.imshow(residual, origin='lower')
        ax1.set_title('PSF')
        ax2.set_title('PSF - 3D')
        ax3.set_title('Residuals')
        if save:
            outname = os.path.join(self.output_dir, self.filename.replace('.fits', '.psf.png'))
            plt.savefig(outname, dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

    def recover_SDSS_PSF(self, save=True, show=False, pixscale=0.2):

        if pixscale is not None:
            data, wcs, header = self.resample(pixscale=pixscale, inplace=False)
            self.psfscale = pixscale

        else:
            data = self.data
            header = self.header
            wcs = WCS(header)
            self.psfscale = np.around(self.wcs.proj_plane_pixel_scales()[0].to(u.arcsec).value, 3)

        new_psf = create_sdss_psf(data, header, self.output_dir, pixscale=self.psfscale)

        psf_flux = np.sum(new_psf)

        if np.abs(1-psf_flux) < 0.0001:
            hdu = fits.PrimaryHDU(new_psf)
            self.psf = new_psf.data
        else:
            hdu = fits.PrimaryHDU(new_psf / psf_flux)
            self.psf = new_psf / psf_flux

        hdu.header['PSFSCALE'] = self.psfscale

        self.plot_psf(new_psf, residual=None, save=save, show=show)

        out = os.path.join(self.output_dir, self.filename.replace('.fits', '.psf.fits'))
        hdu.writeto(out, overwrite=True)

    def convert_units(self, out_units, equivalency=None):
        """
        Convert the unit of measurement of the image to a different one.

        Args:
            out_units (astropy.units):
                Final units of the image.
            equivalency (astropy.units.equivalencies.Equivalency, optional):
                Equivalency used to transform flux density units to flux units. Defaults to None.
        """

        print(f'Updating the units from {self.units} to {out_units}')
        tmp_image = self.data * self.units
        tmp_image = tmp_image.to(out_units, equivalencies=equivalency)
        self.data = tmp_image.value
        self.units = out_units

    def open_psf(self, filename):
        """
        Open the PSF file.

        Args:
            filename (str):
                full path to the PSF file.
        """

        with fits.open(filename) as hdu:
            psf = hdu[0].data
            head = hdu[0].header

        self.psf = psf
        self.psfscale = head['PSFSCALE']
