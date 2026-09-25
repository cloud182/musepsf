import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt


from scipy.ndimage import rotate, zoom
from .image import Image

import logging
import os

from .utils import check_flux_calibration, locate_stars, run_measure_psf
from .plotting import plot_images, plot_psf

logger = logging.getLogger(__name__)


class MUSEImage(Image):
    """
    Class to deal with MUSE specific images. It inherits some of the properties from the
    Image class.

    Attributes:
        filename (str):
            name of the input file
        inpit_dir (str):
            path of the input file
        output_dir (str):
            path where to save the output files
        plot_dir (str):
            output_dir/plots -- parent of the per-diagnostic-type plot subfolders (see Image).
        stars_plot_dir, psf_plot_dir (str):
            output_dir/plots/{stars,psf} -- see Image.
        flux_calibration_plot_dir (str):
            output_dir/plots/flux_calibration -- flux rescaling before/after and scatter plots.
        resampled_plot_dir (str):
            output_dir/plots/resampled -- MUSE-vs-reference resampled comparison plot.
        fit_plot_dir (str):
            output_dir/plots/fit -- final PSF-fit diagnostic plot (MUSE/WFI/Diff).
        qc_plot_dir (str):
            output_dir/plots/qc -- QC summary plot (utils.plot_qc_summary()).
        psf_dir (str):
            output_dir/psf -- fitted/recovered PSF FITS files (see Image).
        catalog_dir (str):
            output_dir/catalogs -- star tables (see Image).
        region_dir (str):
            output_dir/regions -- manually/automatically selected star region files.
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
        scale (float):
            pixelscale (arcsec/pix) of the image
        starpos (astropy.table.Table):
            table containig the position of the stars identified in the image
        starmask (np.ndarray):
            array containing the area of the image masked because associated to stars
        convolved (np.ndarray):
            image convolved with the PSF of the reference image
        alpha (float):
            first guess for the power index of the Moffat PSF
        res (list):
            full output of the fitting routine
        best_fit (list):
            best fitting parameters

    Methods:
        measure_psf(reference: Image, fit_alpha=False, plot=False, save=False, show=True,
                    offset=False, oversample=None, **kwargs):
            Measure the PSF of the image by using a cross-convolution technique to compare it
            to a reference image with known PSF.
        get_rot():
            Return the rotation angle (in degrees) of the image with respect to celestial north.

    Inherited from Image:
        resample(header=None, pixscale=None):
            Resample the image to match a specific resolution or a specific header.
        mask_galaxy(center, amax, amin, pa):
            Mask the area covered by the galaxy when looking for gaia stars.
        get_gaia_catalog(center, gmin, gmax, radius=10*u.arcmin):
            Query the Gaia Catalog to identify stars in the field of the galaxy.
        build_startable(coords, data, wcs, save=True, show=False):
            Refine the position of the stars and build the star table that will be feed to the
            ePSF builder
        build_psf(center, gmin, gmax, stars_file=None, radius=10*u.arcmin, npix=35,
                  pixscale=0.2, oversampling=4, save=True, show=True):
            Build the ePSF of the considered image.
        convert_units(out_units, equivalency=None):
            Convert the unit of measurement of the image.
        open_psf(filename):
            Open the file containing the PSF of the image.
    """

    def __init__(self, filename, input_dir='./', output_dir='./', datahdu=1, headerhdu=0, debug=False,
                 units=u.erg / (u.cm * u.cm * u.second * u.AA) * 1.e-20):
        """
        Initialization method of the class

        Args:
            filename (str):
                name of the file containing the image
            input_dir (str, optional):
                Location of the input file. Defaults to './'.
            output_dir (str, optional):
                Location where to save the output files. Defaults to './'.
            datahdu (int, optional):
                HDU containing the data. Defaults to 1.
            headerhdu (int, None, optional):
                HDU containing the main header. Defaults to 0.
            debug (bool, optional):
                If True, several diagnostic plots will be produced. Defaults to False.
            units (astropy.units, None, optional):
                Units of the data extension. Defaults to u.erg/(u.cm * u.cm * u.second * u.AA)*1.e-20.
        """

        super().__init__(filename, input_dir, output_dir, datahdu, headerhdu, debug, units)

        self.scale = self.wcs.proj_plane_pixel_scales()[0].to_value(u.arcsec)
        self.region_dir = os.path.join(self.output_dir, 'regions')
        self.flux_calibration_plot_dir = os.path.join(self.plot_dir, 'flux_calibration')
        self.resampled_plot_dir = os.path.join(self.plot_dir, 'resampled')
        self.fit_plot_dir = os.path.join(self.plot_dir, 'fit')
        self.qc_plot_dir = os.path.join(self.plot_dir, 'qc')
        for directory in (self.region_dir, self.flux_calibration_plot_dir, self.resampled_plot_dir,
                          self.fit_plot_dir, self.qc_plot_dir):
            if not os.path.isdir(directory):
                os.makedirs(directory)

        # I need to know where the image is zero to erode it before minimization
        self.zeromask = (self.data == 0) + np.isnan(self.data)


    def measure_psf(self, reference: Image, fit_alpha=False, plot=False, save=False, show=True,
                    offset=False, oversample=None, **kwargs):
        """
        Measure the PSF of the image by using a cross-convolution technique to compare the
        MUSE image to a reference image with known PSF.

        Args:
            reference (Image):
                Reference image. It must have a PSF already measured and the units must be the same
                as the MUSE ones.
            fit_alpha (bool, optional):
                If true, both the FWHM and the power index of the Moffat will be estimated.
                Defaults to False.
            plot (bool, optional):
                If True, several diagnostic plots will be produced. Defaults to False.
            save (bool, optional):
                If True, the plots will be saved. Defaults to False.
            show (bool, optional):
                If True, the plots will be shown. Defaults to True.
            offset (bool, optional):
                If True, fit an additional offset between the two images. Defaults to False.
            oversample (int, optional):
                If not None, the image will be oversampled by this factor. Defaults to None.
        """

        if self.units != reference.units:
            raise ValueError('The two images are not in the same units')
        if reference.psf is None:
            raise ValueError('The reference PSF is missing')
        if reference.psfscale != np.around(self.wcs.proj_plane_pixel_scales()[0].to(u.arcsec).value, 3):
            raise ValueError('The PSF has been created with a different pixel scale')

        self.suffix = kwargs.pop('suffix', '')

        # resampling the reference to the MUSE WCS
        resample = kwargs.pop('resample', True)
        if resample:
            reference.resample(header=self.header)
        if plot:
            outname = os.path.join(self.resampled_plot_dir, self.filename.replace('.fits', f'{self.suffix}_resampled.png'))
            plot_images(self.data, reference.data, 'MUSE', 'Reference', outname,
                        save=save, show=show)

        # rescaling the flux
        rescaled_name = os.path.join(self.flux_calibration_plot_dir,
                               self.filename.replace('.fits', f'{self.suffix}_rescaled.png'))
        scatter_name = os.path.join(self.flux_calibration_plot_dir,
                                       self.filename.replace('.fits', f'{self.suffix}_scatter.png'))

        self.data = check_flux_calibration(self.data, reference.data, self.zeromask, plot=plot, save=save,
                               show=show, outname=rescaled_name, outname_scatter=scatter_name)


        if 'WFM-AO' in self.main_header['HIERARCH ESO INS MODE']:
            alpha = 2.3
        else:
            alpha = 2.8

        edge = kwargs.pop('edge', 50)
        dx0 = kwargs.pop('dx0', 0)
        dy0 = kwargs.pop('dy0', 0)
        fwhm0 = kwargs.pop('fwhm0', 0.8)
        data = self.data

        reg_name = os.path.join(self.region_dir, self.filename.replace('.fits', '_regions.reg'))

        mask_stars = kwargs.pop('mask_stars', True)

        if mask_stars:
            if os.path.isfile(reg_name):
                logger.info('Including manually selected sources')
                star_pos, self.starmask = locate_stars(data, filename=reg_name, **kwargs)
            else:
                star_pos, self.starmask = locate_stars(data, filename=None, **kwargs)
                if star_pos is not None:
                    star_pos.write(reg_name, format='ascii.no_header')
        else:
            star_pos = None
            self.starmask = np.zeros(data.shape, dtype=bool)


        # get the rotation of the image
        img_rot = self.get_rot()

        # apply rotation to the PSF
        if img_rot != 0:
            logger.info(f'A rotation of {img_rot:0.1f} deg has been detected. Applying rotation to PSF')
            psf = rotate(reference.psf, img_rot)
        else:
            psf = reference.psf

        if oversample is not None and oversample > 1:
            data[np.isnan(data)] = 0
            data = zoom(data, zoom=oversample)/oversample**2
            ref_data = zoom(reference.data, zoom=oversample)/oversample**2
            psf = zoom(psf, zoom=oversample)
            psf /= psf.sum()

            plot_psf(psf, self.psf_plot_dir, self.filename, suffix=f'{self.suffix}_oversampled')

            scale = self.scale/oversample
        else:
            ref_data = reference.data
            scale = self.scale
            oversample = 1

        figname = os.path.join(self.fit_plot_dir, self.filename.replace('.fits', f'{self.suffix}_final.png'))

        self.res, self.star_pos, self.starmask = run_measure_psf(data, ref_data,
                                                                 psf, star_pos, self.starmask,
                                                                 self.zeromask,
                                                                 oversample, figname=figname,
                                                                 fit_alpha=fit_alpha,
                                                                 alpha=alpha, fwhm0=fwhm0,
                                                                 offset=offset,
                                                                 scale=scale,
                                                                 plot=plot, save=save,
                                                                 edge=edge, dx0=dx0, dy0=dy0,
                                                                 **kwargs)
        self.best_fit = self.res[0]


    def get_rot(self):

        # Extract the CD matrix
        cd_matrix = self.wcs.pixel_scale_matrix  # This includes scaling if CD elements are used

        # Compute the angle of the y-axis with respect to celestial north
        theta_rad = np.arctan2(-cd_matrix[0, 1], cd_matrix[1, 1])  # Note the negative sign
        theta_deg = theta_rad * 180 / np.pi

        return theta_deg








