from mpl_toolkits.axes_grid1 import make_axes_locatable

import astropy.visualization as vis
import matplotlib.pyplot as plt
import numpy as np

from astropy.convolution import Moffat2DKernel
from astropy.stats import sigma_clipped_stats
from astroquery.gaia import Gaia
from photutils.detection import DAOStarFinder
from astropy.convolution import convolve_fft
from scipy.optimize import leastsq
from numpy.fft import fftfreq

import spacepylot.alignment as spalign
import spacepylot.plotting as pl
import spacepylot.alignment_utilities as alutils

# configure astroquery gaia
Gaia.ROW_LIMIT = 10000
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

def query_gaia(center, radius):
    """
    Query Gaia catalog

    Args:
        center (SkyCoord):
            Center of the field
        radius (astropy.units):
            Radius of the area to be searched

    Returns:
        astropy.table.Table:
            table with the Gaia stars in the field
    """

    r = Gaia.query_object_async(coordinate=center, radius=radius)
    r = r['ra', 'dec', 'parallax', 'phot_g_mean_mag', 'classprob_dsc_combmod_star'].copy()
    mask = np.abs(r['classprob_dsc_combmod_star']) > 0.99
    r = r[mask].copy()
    return r


def get_norm(image, perc=99.9):
    """
    Normalize colorscale for plotting the images.

    Args:
        image (np.ndarray):
            Image to be normalized
        perc (float, optional):
            Percentage of the points to be used to normalize the colorscale. Defaults to 99.9.

    Returns:
        astropy.visualization.ImageNormalize:
            Normalization to be applied to the image
    """

    interval = vis.PercentileInterval(perc)
    vmin, vmax = interval.get_limits(image)
    norm = vis.ImageNormalize(vmin=vmin, vmax=vmax,
                            stretch=vis.LogStretch(1000))
    return norm


def plot_images(image1, image2, title1, title2, name, save=False, show=True):
    """
    Routine to plot two images side by side with the same color scale

    Args:
        image1 (np.ndarray):
            First image to plot
        image2 (np.ndarray):
            Second image to plot
        title1 (str):
            Title of the first subplot
        title2 (str):
            Title of the second subplot
        name (str):
            Name of the final figure file
        save (bool, optional):
            Save the output plots. Defaults to True.
        show (bool, optional):
            Show the output plots. Defaults to True.
    """

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
    """
    Bin images for flux calibration checks

    Args:
        image (np.ndarray):
            Image to be binned
        bin_size (int, optional):
            size of the bins. Defaults to 15.

    Returns:
        np.ndarray:
            Median of the flux in each bin
        np.ndarray:
            Standard deviation of the flux in each bin
    """

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
    """
    Routine to automatically detect stars in the image using DAOStarFinder.

    Args:
        image (np.ndarray):
            Input image

    Returns:
        (astropy.table.Table, None):
            table containing the position of the stars identified in the image. If no stars are
            present, None is returned.
        (np.ndarray, None):
            circular mask covering the emission of the stars.
    """

    fwhm = kwargs.get('fwhm', 3)
    brightest = kwargs.get('brightest', 5)
    sigma = kwargs.get('sigma', 3.)
    radius = kwargs.get('radius', 20)

    # Define a threshold to look for stars
    mean, median, std = sigma_clipped_stats(image, sigma=sigma)
    thresh = mean + 10. * std

    # Initializing and starting the starfinder
    starfinder = DAOStarFinder(threshold=thresh, fwhm=fwhm, brightest=brightest)
    sources = starfinder(image)

    if sources is not None:
        mask = np.zeros(image.shape, dtype=bool)
        yy, xx = np.mgrid[ :image.shape[0], :image.shape[1]]
        stars = sources['xcentroid', 'ycentroid']
        for star in stars:
            distance = np.sqrt((xx-star['xcentroid'])**2+(yy-star['ycentroid'])**2)
            mask[distance < radius] = True
    else:
        stars = None
        mask = None

    return stars, mask

def moffat_kernel(fwhm, alpha, scale=0.238, img_size=241):
    """
    Create the Moffat kernel representing MUSE PSF

    Args:
        fwhm (float):
            FWHM (in arcsec) of the Moffat kernel
        alpha (float):
            Power index of the Moffat kernel
        scale (float, optional):
            scale to convert the FWHM from arcsec to pixels. Defaults to 0.238.
        img_size (int, optional):
            final size of the kernel. Defaults to 241.

    Returns:
        astropy.convolution.kernels.Moffat2DKernel:
            Moffat kernel
    """
    fwhm = fwhm / scale
    gamma = fwhm/(2*np.sqrt(2**(1/alpha)-1))

    moffat_k = Moffat2DKernel(gamma, alpha, x_size=img_size, y_size=img_size)

    return moffat_k


def apply_mask(image, starmask, nanmask, edge=5):
    """
    Apply the same starmask to 2 images of the same size

    Args:
        image1 (np.ndarray):
            First image
        image2 (np.ndarray):
            Second image
        starmask (np.ndarray):
            array containing the stellar mask
        edge (int, optional):
            number of pixels to be removed at the edge of the images. Defaults to 5.

    Returns:
        np.ndarray:
            Masked and trimmed version of image1
        np.ndarray:
            Masked and trimmed version of image2
        """

    masked = np.ma.masked_array(data=image, mask=nanmask)

    if starmask is not None:
        assert starmask.shape == image.shape, 'Mask and image are of different shape'
        masked[starmask] = np.ma.masked

    if edge != 0:
        return masked[edge:-edge, edge:-edge]
    else:
        return masked


def to_minimize(pars, convolved, reference, starmask, nanmask, fxx, fyy, arrayslices, edge=50, alpha0=None,
                fwhm_bound=[0.4, 2], alpha_bound=[1, 10], dd_bound=[-2, 2], scale=0.2):
    """
    Compute the function to be minimize to measure the PSF properties

    Args:
         pars (list):
            Initial guess for the fitted parameters.
        convolved (np.ndarray):
            array for which the PSF will be measured, convolved for the model of the PSF of the
            reference image.
        reference (np.ndarray):
            reference image
        starmask (np.ndarray):
            boolean mask selecting the pixels associated to stellar emission that should be masked.
        edge (int, optional):
            number of pixels at the edge of the image to be ignored during minimization.
            Defaults to 50.
        alpha0 (float, optional):
            Value of alpha if not among the fittted parameters. Defaults to None.
        fwhm_bound (list, optional):
            minimum and maximum limits that the FWHM parameter can assume. Defaults to [0.4, 2].
        alpha_bound (list, optional):
            minimum and maximum limits that the alpha parameter can assume.. Defaults to [1, 10].
        scale (float, optional):
            pixel scale of the images. Defaults to 0.2.

    Returns:
        np.ndarray:
            Function needed for the minimization
    """

    if len(pars) == 1:
        fwhm = pars[0]
        alpha, dx, dy = alpha0, 0, 0
    if len(pars) == 2:
        fwhm, alpha = pars
        dx, dy = 0, 0
    elif len(pars) == 3:
        fwhm, dx, dy = pars
        alpha = alpha0
    elif len(pars) == 4:
        fwhm, dx, dy, alpha = pars

    factor = 1 #this is a factor that will be used to return very high numbers if the
                #parameters are out of bounds

    if fwhm_bound is not None:
        if fwhm < fwhm_bound[0] or fwhm > fwhm_bound[1]:
            fwhm = 0.4
            factor = 1e10
    if alpha_bound is not None:
        if alpha < alpha_bound[0] or alpha > alpha_bound[1]:
            alpha = 2
            factor = 1e10
    if dd_bound is not None:
         if dx < dd_bound[0] or dx > dd_bound[1]:
            dx = 0
            factor = 1e10
         if dy < dd_bound[0] or dy > dd_bound[1]:
            dy = 0
            factor = 1e10


    # creating model of MUSE PSF
    ker_MUSE = moffat_kernel(fwhm, alpha, scale=scale, img_size=50)

    # convolving WFI image for the model of MUSE PSF
    reference_conv = convolve_fft(reference, ker_MUSE, return_fft=True)
    # import sys; sys.exit()

    reference_conv = apply_offset_fourier(reference_conv, dx, dy, fxx, fyy, arrayslices)

    ref_masked = apply_mask(reference_conv, starmask, nanmask, edge=edge)

    # leastsq requires the array of residuals to be minimized
    function = (convolved-ref_masked)

    return function.ravel() * factor

def apply_offset_fourier(convolved, dx, dy, fxx, fyy, arrayslices):

    ff = fyy*dy+fxx*dx

    convolved *= np.exp(-2j*np.pi*ff)

    convolved = np.fft.ifftn(convolved)

    return convolved[arrayslices[0]: arrayslices[1], arrayslices[2]: arrayslices[3]].real



def plot_results(pars, convolved, reference, starmask, nanmask, fxx, fyy, arrayslices, figname, save=False, show=False,
                 edge=50, alpha0=None, scale=0.2):
    """
    Functions that plot the final results of the PSF fitting

    Args:
        pars (list):
            initial guess for the fitted parameters.
        convolved (np.ndarray):
            array for which the PSF will be measured, convolved for the model of the PSF of the
            reference image.
        reference (np.ndarray):
            reference image
        starmask (np.ndarray):
            boolean mask selecting the pixels associated to stellar emission that should be masked.
        figname (str):
            figure name.
        save (bool, optional):
            if True, save the plot. Defaults to False.
        show (bool, optional):
            if True, show the plot. Defaults to False.
        edge (int, optional):
            number of pixels at the edge of the image to be ignored during minimization.
            Defaults to 50.
        alpha0 (float, optional):
            Value of alpha if not among the fittted parameters. Defaults to None.
        scale (float, optional):
            pixel scale of the images. Defaults to 0.2.
    """

    if len(pars) == 1:
        fwhm = pars[0]
        alpha, dx, dy = alpha0, 0, 0
    if len(pars) == 2:
        fwhm, alpha = pars
        dx, dy = 0, 0
    elif len(pars) == 3:
        fwhm, dx, dy = pars
        alpha = alpha0
    elif len(pars) == 4:
        fwhm, dx, dy, alpha = pars

    # creating model of MUSE PSF
    ker_MUSE = moffat_kernel(fwhm, alpha, scale=scale, img_size=50)

    # convolving WFI image for the model of MUSE PSF
    reference_conv = convolve_fft(reference, ker_MUSE, return_fft=True)
    reference_conv = apply_offset_fourier(reference_conv, dx, dy, fxx, fyy, arrayslices)

    ref_masked = apply_mask(reference_conv, starmask, nanmask, edge=edge)

    # plotting the results of the convolution

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax1.set_title('MUSE')
    ax2.set_title('WFI')
    ax3.set_title('Diff')

    # normalization for a better plot
    interval = vis.PercentileInterval(99.9)
    vmin, vmax = interval.get_limits(convolved)
    norm = vis.ImageNormalize(vmin=vmin, vmax=vmax,
                            stretch=vis.LogStretch(1000))

    # MUSE image
    img1 = ax1.imshow(convolved, norm=norm, origin='lower')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)

    # WFI image
    img2 = ax2.imshow(ref_masked, norm=norm, origin='lower')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)

    # Difference image
    img3 = ax3.matshow(convolved/ref_masked, vmin=0.8,
                    vmax=1.2, origin='lower')
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes('right', size='5%', pad=0.05)

    axes = [ax1, ax2, ax3]

    for ax in axes:
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(True)

    fig.colorbar(img1, cax=cax1)
    fig.colorbar(img2, cax=cax2)
    fig.colorbar(img3, cax=cax3)
    if save:
        plt.savefig(figname, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def run_measure_psf(data, reference, psf, figname=None, alpha=2.8, edge=50, fwhm0=0.8, dx0=0, dy0=0,
                    fit_alpha=False, plot=False, save=False, show=False, scale=0.2,
                    offset=False, **kwargs):
    """
    Functions that performs the fit of the PSF.

    Args:
        data (np.ndarray):
            data for which the PSF should be estimated
        reference (np.ndarray):
            reference array, already resampled to the samw WCS of data, and rescaled to the same
            units of measurement
        psf (np.ndarray):
            model of psf for the reference image, in the same pizelscale of the data array.
        figname (str):
            name of the figure
        alpha (float, optional):
           First guess for aklpha, or alpha to use if fit_alpha=False. Defaults to 2.8.
        edge (int, optional):
            pixels to remove from the edge to avoid edges effects. Defaults to 50.
        fwhm0 (float, optional):
            first guess for the FWHM. Defaults to 0.8.
        fit_alpha (bool, optional):
            If True, alphga is considered a free parameter to be fit. Otherwise, it is considered
            as a False. Defaults to False.
        plot (bool, optional):
            If True, several diagnostic plots will be produced. Defaults to False.
        save (bool, optional):
            If True, the plots will be saved. Defaults to False.
        show (bool, optional):
            If True, the plots will be shown. Defaults to False.
        scale (float, optional):
            Scale fo the data array. Defaults to 0.2.

    Returns:
        dict:
            full results of the fit
        (astropy.table.Table, None):
            Table contasining the position of the masked stars
        np.ndarray:
            boolean mask selecting the pixels associated to stellar emission that should be masked.
    """


    star_pos, starmask = locate_stars(data, **kwargs)

    nanmask = np.isnan(data)

    # computing things to perform the minimization more efficiently
    # creating model of MUSE PSF
    ker_MUSE = moffat_kernel(1, 2.8, scale=scale, img_size=50)
    # convolving WFI image for the model of MUSE PSF
    reference_conv = convolve_fft(reference, ker_MUSE, return_fft=True)

    fx = fftfreq(reference_conv.shape[1])
    fy = fftfreq(reference_conv.shape[0])

    fxx, fyy = np.meshgrid(fx, fy)

    arrayslices = []
    for dimension_conv, dimension in zip(reference_conv.shape, reference.shape):
        center = dimension_conv - (dimension_conv + 1) // 2
        arrayslices += [center - dimension // 2, center + (dimension + 1) // 2]


    convolved = convolve_fft(data, psf)
    convolved = apply_mask(convolved, starmask, nanmask, edge=edge)

    print(f'Using alpha = {alpha}')

    print('Performing the fit')
    fwhm_bound = kwargs.get('fwhm_bound', [0.2, 2])
    alpha_bound = kwargs.get('alpha_bound', [1, 10])
    dd_bound = kwargs.get('dd_bound', [-2, 2])
    # it is possible to fit the alpha parameter or to assume a fixed value
    if offset:
        if fit_alpha:
            p0=[fwhm0, dx0, dy0, alpha]
        else:
            p0=[fwhm0, dx0, dy0]
    else:
        if fit_alpha:
            p0=[fwhm0, alpha]
        else:
            p0=[fwhm0]

    res = leastsq(to_minimize, x0=p0,
                #convolved, reference, starmask, edge, alpha0, fwhm_bound,
                # alpha_bound, scale
                args=(convolved, reference, starmask, nanmask, fxx, fyy, arrayslices, edge, alpha, fwhm_bound,
                        alpha_bound, dd_bound, scale),
                maxfev=600, xtol=1e-9, full_output=True)

    best_fit = res[0]

    print('Fit completed')
    print(f'Measured FWHM = {best_fit[0]:0.2f}')
    if offset:
        print(f'Measured offset x:{best_fit[1]:0.2f} y:{best_fit[2]:0.2f}')
    if fit_alpha:
        print(f'Measured alpha = {best_fit[-1]:0.2f}')

    if plot:
        plot_results(best_fit, convolved, reference, starmask, nanmask, fxx, fyy, arrayslices, save=save, show=show,
                     figname=figname, edge=edge, alpha0=alpha)

    return res, star_pos, starmask

def run_spacepylot(data, reference, figname=None, fwhm=None, psf=None, alpha=2.8, scale=0.2, verbose=False):

    # just to make sure I do not mess up the originals
    newdata = data.copy()
    newreference = reference.copy()

    if fwhm is not None:
        ker_MUSE = moffat_kernel(fwhm, alpha, scale=scale, img_size=50)
        newreference = convolve_fft(newreference, ker_MUSE)
        newdata = convolve_fft(newdata, psf)

    op = spalign.AlignOpticalFlow(newdata, newreference, verbose=verbose)
    op.get_iterate_translation_rotation(5)
    op_plot = pl.AlignmentPlotting.from_align_object(op)

    rotated = alutils.rotate_image(data, op.rotation_deg)
    before = data.copy()
    data = alutils.translate_image(rotated, op.translation)

    if figname is not None:
        op_plot.before_after()
        plt.savefig(figname, dpi=200)
        plt.imshow(before-data)
        plt.savefig(figname.replace('.png', '_diff.png'))

    return data, op.rotation_deg, op.translation