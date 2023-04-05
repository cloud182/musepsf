import argparse
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import astropy.visualization as vis

from astropy.wcs import WCS
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.ndimage import shift, map_coordinates
from scipy.ndimage.interpolation import rotate
from scipy.optimize import leastsq
from astropy.nddata import Cutout2D
from scipy.odr import ODR, Model, RealData
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.convolution import Moffat2DKernel, Gaussian2DKernel, convolve_fft
from reproject import reproject_exact, reproject_interp
from astropy.coordinates import SkyCoord
from astropy.modeling import models, fitting
from copy import copy
from musepsf.image import Image

import sys
import os
import warnings
import time

def get_norm(image, perc=99.9):

    interval = vis.PercentileInterval(perc)
    vmin, vmax = interval.get_limits(image)
    norm = vis.ImageNormalize(vmin=vmin, vmax=vmax,
                            stretch=vis.LogStretch(1000))
    return norm

def plot_images(image1, image2, title1, title2, name, save=False, show=True):

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    norm = get_norm(image1)
    ax[0].imshow(image1, norm=norm, origin='lower')
    ax[1].imshow(image2, norm=norm, origin='lower')
    ax[0].set_title(title1)
    ax[1].set_title(title2)
    if save:
        plt.savefig(name, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

def bin_image(image, bin_size=15):

    n_x = image.shape[1] // bin_size
    n_y = image.shape[0] // bin_size
    median = np.zeros((n_y, n_x))
    std = np.zeros_like(median)
    for i in range(n_x):
        for j in range(n_y):
            median[j, i] = np.median(image[j*bin_size:j*bin_size+bin_size,
                                                i*bin_size:i*bin_size+bin_size])
            std[j, i] = np.std(image[j*bin_size:j*bin_size+bin_size,
                                          i*bin_size:i*bin_size+bin_size])

    median = np.ravel(median)
    std = np.ravel(std)

    return median, std

def linear_function(B, x):
    """ Linear function for the ODR fitting routine"""
    return B[0]*x + B[1]

def locate_stars(image, **kwargs):

    fwhm = kwargs.get('fwhm', 3)
    brightest = kwargs.get('brightest', 5)
    sigma = kwargs.get('sigma', 3.)

    # Define a threshold to look for stars
    mean, median, std = sigma_clipped_stats(image, sigma=sigma)
    thresh = mean + 10. * std

    # Initializing and starting the starfinder
    starfinder = DAOStarFinder(threshold=thresh, fwhm=fwhm, brightest=brightest)
    sources = starfinder(image)

    if sources is not None:
        stars = sources['xcentroid', 'ycentroid']
    else:
        stars = None

    return stars

def moffat_kernel(fwhm, alpha, scale=0.238, img_size=241):
    """
    Moffat kernel for FFT.
    Input:
        - fwhm:
            fwhm of the Moffat kernel, in arcsec.
        - alpha:
            power index of the Moffat
        - scale:
            pixel scale of the image
        - img_size:
            size of the kernel. It should be an odd number.
    """

    fwhm = fwhm / scale
    gamma = fwhm/(2*np.sqrt(2**(1/alpha)-1))

    moffat_k = Moffat2DKernel(gamma, alpha, x_size=img_size, y_size=img_size)

    return moffat_k

def apply_mask(image1, image2, stars, edge=5, radius=20):

    xx, yy = np.mgrid[ :image1.shape[1], :image1.shape[0]]

    if stars is not None:
        for star in stars:
            distance = np.sqrt((xx-star['xcentroid'])**2+(yy-star['ycentroid'])**2)
            mask = distance < radius
            image1[mask] = np.nan
            image2[mask] = np.nan

    if edge != 0:
        return image1[edge:-edge, edge:-edge], image2[edge:-edge, edge:-edge]
    else:
        return image1, image2


class MUSEImage(Image):

    def __init__(self, filename, datahdu=1, headerhdu=0, debug=False,
                 units=u.erg / (u.cm * u.cm * u.second * u.AA) * 1.e-20):

        super().__init__(filename, datahdu, headerhdu, debug, units)

        self.scale = self.wcs.proj_plane_pixel_scales()[0].to_value(u.arcsec)


    def measure_psf(self, reference: Image, plot=False, save=False, show=True, **kwargs):

        assert self.units == reference.units, 'The two images are not in the same units'
        assert reference.psf is not None, 'The reference PSF is missing'

        # resampling the reference to the MUSE WCS
        reference.resample(header=self.header)
        if plot:
            plot_images(self.data, reference.data, 'MUSE', 'Reference', 'Resampled.png',
                        save=False, show=True)

        # rescaling the flux
        self.check_flux_calibration(reference.data, plot=plot, save=save, show=show)

        self.stars = locate_stars(self.data, **kwargs)

        self.convolved = convolve_fft(self.data, reference.psf)

        # determining the alpha to use for the fit
        alpha = kwargs.get('alpha', None)
        if alpha is None:
            if self.main_header['HIERARCH ESO TPL ID'] == 'MUSE_wfm-ao_obs_genericoffsetLGS':
                alpha = 2.3
                print(f'AO data, using alpha = {alpha}')
            else:
                alpha = 2.8
                print(f'NOAO data, using alpha = {alpha}')
        else:
            print(f'Using Custom alpha = {alpha}')

        self.alpha = alpha

        print('Performing the fit')
        edge = kwargs.get('edge', 10)
        self.res = leastsq(self.to_minimize, x0=[0.8],
                           args=(reference.data, False, False, None, edge),
                           maxfev=600, xtol=1e-8, full_output=True)

        self.best_fit = self.res[0]

        print('Fit completed')
        print(f'Measured FWHM = {self.best_fit[0]}')

        function = self.to_minimize(self.best_fit, reference.data, plot=True, save=True,
                                    figname='Final_result.png', edge=edge)

    def to_minimize(self, fwhm, reference, plot=False, save=False,
                    figname=None, edge=10):

        # creating model of MUSE PSF
        size = self.convolved.shape[0]
        ker_MUSE = moffat_kernel(fwhm, self.alpha, scale=self.scale, img_size=size)

        # convolving WFI image for the model of MUSE PSF
        reference_conv = convolve_fft(reference, ker_MUSE)

        MUSE_masked, ref_masked = apply_mask(self.convolved, reference_conv, self.stars,
                                             edge=edge, radius=20)

        # plotting the results of the convolution if required
        if plot:
            fig = plt.figure(figsize=(16, 6))
            gs = fig.add_gridspec(1, 3)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])
            ax1.set_title('MUSE')
            ax2.set_title('WFI')
            ax3.set_title('Diff')

            # MUSE FFT

            # normalization for a better plot
            interval = vis.PercentileInterval(99.9)
            vmin, vmax = interval.get_limits(MUSE_masked)
            norm = vis.ImageNormalize(vmin=vmin, vmax=vmax,
                                    stretch=vis.LogStretch(1000))

            # MUSE image
            img1 = ax1.imshow(MUSE_masked, norm=norm, origin='lower')
            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes('right', size='5%', pad=0.05)

            # WFI image
            img2 = ax2.imshow(ref_masked, norm=norm, origin='lower')
            divider2 = make_axes_locatable(ax2)
            cax2 = divider2.append_axes('right', size='5%', pad=0.05)

            # Difference image
            img3 = ax3.matshow(MUSE_masked/ref_masked, vmin=0.8,
                            vmax=1.2, origin='lower')
            divider3 = make_axes_locatable(ax3)
            cax3 = divider3.append_axes('right', size='5%', pad=0.05)

            axes = [ax1, ax2, ax3]

            for ax in axes:
                ax.axes.get_yaxis().set_visible(False)
                ax.axes.get_xaxis().set_visible(False)

            fig.colorbar(img1, cax=cax1)
            fig.colorbar(img2, cax=cax2)
            fig.colorbar(img3, cax=cax3)
            if save:
                plt.savefig(figname, dpi=150)
            plt.show()

        # leastsq requires the array of residuals to be minimized
        function = (MUSE_masked-ref_masked)

        return function.ravel()

    def check_flux_calibration(self, reference, bin_size=15, plot=False, save=False, show=True):
        MUSE_median, MUSE_std = bin_image(self.data, bin_size)
        reference_median, reference_std = bin_image(reference, bin_size)

        #removing nans
        index1 = np.isnan(MUSE_median)
        index2 = np.isnan(reference_median)
        index = np.any((index1, index2), axis=0)

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
                plt.savefig('Scatter_flux.png')
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
            plot_images(orig, rescaled, 'Original', 'Rescaled,', 'Rescaled.png',
                        save=save, show=show)











