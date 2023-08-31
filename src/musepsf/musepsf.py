import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from scipy.odr import ODR, Model, RealData
from .image import Image

import os

from .utils import plot_images, bin_image, linear_function, run_measure_psf


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
        measure_psf(reference: Image, fit_alpha=False, plot=False, save=False, show=True,
                    **kwargs):
            Measure the PSF of the image by using a cross-convolution technique.
        to_minimize(pars, reference, plot=False, save=False, show=False,
                    figname=None, edge=10):
            Function to be minimize to measure the PSF properties
        check_flux_calibration(self, reference, bin_size=15, plot=False, save=False, show=True):
            Check that the flux calibration between the two images is consistent.
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


    def measure_psf(self, reference: Image, fit_alpha=False, plot=False, save=False, show=True,
                    **kwargs):
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
        """

        assert self.units == reference.units, 'The two images are not in the same units'
        assert reference.psf is not None, 'The reference PSF is missing'
        assert reference.psfscale == np.around(self.wcs.proj_plane_pixel_scales()[0].to(u.arcsec).value, 3), 'The PSF has been created with a different pixel scale'

        # resampling the reference to the MUSE WCS
        reference.resample(header=self.header)
        if plot:
            outname = os.path.join(self.output_dir, self.filename.replace('.fits', '_resampled.png'))
            plot_images(self.data, reference.data, 'MUSE', 'Reference', outname,
                        save=save, show=show)

        # rescaling the flux
        self.check_flux_calibration(reference.data, plot=plot, save=save, show=show)

        figname = os.path.join(self.output_dir, self.filename.replace('.fits', '_final.png'))

        if self.main_header['HIERARCH ESO TPL ID'] == 'MUSE_wfm-ao_obs_genericoffsetLGS':
            alpha = 2.3
        else:
            alpha = 2.8

        edge = kwargs.pop('edge', 50)
        self.res, self.star_pos, self.starmask = run_measure_psf(self.data, reference.data,
                                                                 reference.psf, figname,
                                                                 fit_alpha=fit_alpha,
                                                                 alpha=alpha, fwhm0=0.8,
                                                                 scale=self.scale,
                                                                 plot=plot, save=save,
                                                                 edge=edge)
        self.best_fit = self.res[0]


    def check_flux_calibration(self, reference, bin_size=15, plot=False, save=False, show=True):
        """
        Check that the flux calibration between the two images is consistent.

        Args:
            reference (np.ndarray):
                Reference image
            bin_size (int, optional):
                Size of each bin used to compare the flux calibration. Defaults to 15.
            plot (bool, optional):
                If True, some diagnostic plots will be produced. Defaults to False.
            save (bool, optional):
                If True, the plots will be saved. Defaults to False.
            show (bool, optional):
                If True, the plots will be shown. Defaults to True.
        """


        MUSE_median, MUSE_std = bin_image(self.data, bin_size)
        reference_median, reference_std = bin_image(reference, bin_size)

        #removing nans and 0s
        index1 = np.isnan(MUSE_median)
        index2 = np.isnan(reference_median)
        index3 = MUSE_median == 0
        index4 = reference_median == 0
        index = np.any((index1, index2, index3, index4), axis=0)

        MUSE_median = MUSE_median[~index]
        MUSE_std = MUSE_std[~index]
        reference_median = reference_median[~index]
        reference_std = reference_std[~index]

        # using scipy ODR because I have errors on the x and y measuerements.
        linear = Model(linear_function)
        mydata = RealData(MUSE_median, reference_median, sx=MUSE_std, sy=reference_std)
        myodr = ODR(mydata, linear, beta0=[1, 0])
        output = myodr.run()

        if plot:
            plt.scatter(MUSE_median, reference_median)
            plt.plot(MUSE_median, linear_function(output.beta, MUSE_median))
            plt.ylabel('Flux WFI')
            plt.xlabel('Flux MUSE')
            if save:
                outname = os.path.join(self.output_dir, self.filename.replace('.fits', '_scatter.png'))
                plt.savefig(outname)
            if show:
                plt.show()
            else:
                plt.close()

        print('\nResidual flux correction.')
        print('Gamma: {:0.3f}, b: {:0.3f}' .format(output.beta[0], output.beta[1]))

        orig = self.data/reference

        self.data = output.beta[0]*self.data + output.beta[1]
        rescaled = self.data/reference
        print('Correction applied.')
        if plot:
            outname = os.path.join(self.output_dir, self.filename.replace('.fits', '_rescaled.png'))
            plot_images(orig, rescaled, 'Original', 'Rescaled,', outname,
                        save=save, show=show)











