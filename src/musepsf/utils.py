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
from numpy.fft import fftfreq, ifftn


from dynesty import DynamicNestedSampler
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot
from dynesty.pool import Pool

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


def apply_mask(image, starmask, edge=5):
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

    masked = np.ma.masked_array(data=image)

    if starmask is not None:
        assert starmask.shape == image.shape, 'Mask and image1 are of different shape'
        masked.mask=starmask


    if edge != 0:
        return masked[edge:-edge, edge:-edge]
    else:
        return masked


def lnlike(pars, convolved, reference, starmask, fxx, fyy, arrayslices, edge=50, alpha0=None, scale=0.2):

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
        scale (float, optional):
            pixel scale of the images. Defaults to 0.2.

    Returns:
        np.ndarray:
            Function needed for the minimization
    """

    if len(pars) == 2:
        fwhm = pars[0]
        alpha, dx, dy = alpha0, 0, 0
    if len(pars) == 3:
        fwhm, alpha = pars[:2]
        dx, dy = 0, 0
    elif len(pars) == 4:
        fwhm, dx, dy = pars[:3]
        alpha = alpha0
    elif len(pars) == 5:
        fwhm, dx, dy, alpha = pars[:4]

    sigma = pars[-1]

    if sigma <= 0:
        return -np.inf  # Log likelihood is -inf if sigma is not positive

    # creating model of MUSE PSF
    ker_MUSE = moffat_kernel(fwhm, alpha, scale=scale, img_size=50)

    # convolving WFI image for the model of MUSE PSF
    reference_conv = convolve_fft(reference, ker_MUSE, return_fft=True)
    # import sys; sys.exit()

    reference_conv = apply_offset_fourier(reference_conv, dx, dy, fxx, fyy, arrayslices)

    ref_masked = apply_mask(reference_conv, starmask, edge=edge) #muse is already masked

    # leastsq requires the array of residuals to be minimized
    function = (convolved-ref_masked)

    return -0.5 * function.size * np.log(2 * np.pi * sigma**2) - ((function**2) / (2 * sigma**2)).sum()

    #PRIOR
def ptform(params, fwhm_range=[0, 2], alpha_range=[1.5, 3], offset_range=[-1, 1]):
    """
    Prior transform for fitting

    Args:
        params (_type_): _description_
        fwhm_range (_type_): _description_
        alpha_range (_type_): _description_
        offset_range (_type_): _description_

    Returns:
        _type_: _description_
    """

    x = np.zeros_like(params)

    if len(params) == 2:
        x[0] = (fwhm_range[1] - fwhm_range[0]) * params[0] + fwhm_range[0]
    if len(params) == 3:
        x[0] = (fwhm_range[1] - fwhm_range[0]) * params[0] + fwhm_range[0]
        x[1] = (alpha_range[1] - alpha_range[0]) * params[1] + alpha_range[0]
    elif len(params) == 4:
        x[0] = (fwhm_range[1] - fwhm_range[0]) * params[0] + fwhm_range[0]
        x[1] = (offset_range[1] - offset_range[0]) * params[1] + offset_range[0]
        x[2] = (offset_range[1] - offset_range[0]) * params[2] + offset_range[0]
    elif len(params) == 5:
        x[0] = (fwhm_range[1] - fwhm_range[0]) * params[0] + fwhm_range[0]
        x[1] = (offset_range[1] - offset_range[0]) * params[1] + offset_range[0]
        x[2] = (offset_range[1] - offset_range[0]) * params[2] + offset_range[0]
        x[3] = (alpha_range[1] - alpha_range[0]) * params[3] + alpha_range[0]

    x[-1] = params[-1] # uniform from 0 to 2

    return x

def apply_offset_fourier(reference_conv, dx, dy, fxx, fyy, arrayslices):

    ff = fyy*dy+fxx*dx

    reference_conv *= np.exp(-2j*np.pi*ff)

    reference_conv = ifftn(reference_conv)

    return reference_conv[arrayslices[0]: arrayslices[1], arrayslices[2]: arrayslices[3]].real



def plot_results(pars, convolved, reference, starmask, fxx, fyy, arrayslices, figname, save=False, show=False,
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

    if len(pars) == 2:
        fwhm = pars[0]
        alpha, dx, dy = alpha0, 0, 0
    if len(pars) == 3:
        fwhm, alpha = pars[:2]
        dx, dy = 0, 0
    elif len(pars) == 4:
        fwhm, dx, dy = pars[:3]
        alpha = alpha0
    elif len(pars) == 5:
        fwhm, dx, dy, alpha = pars[:4]

    # creating model of MUSE PSF
    ker_MUSE = moffat_kernel(fwhm, alpha, scale=scale, img_size=50)

    # convolving WFI image for the model of MUSE PSF
    reference_conv = convolve_fft(reference, ker_MUSE, return_fft=True)
    reference_conv = apply_offset_fourier(reference_conv, dx, dy, fxx, fyy, arrayslices)

    ref_masked = apply_mask(reference_conv, starmask, edge=edge)

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


def run_measure_psf(data, reference, psf, figname, alpha=2.8, edge=50,
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

    # computing the fxx, fyy and arrayslices to make fitting faster
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

    # end preparation for loglike computation


    convolved = convolve_fft(data, psf)
    convolved = apply_mask(convolved, starmask, edge=edge)

    print(f'Using alpha = {alpha}')

    print('Performing the fit')

    fwhm_bound = kwargs.get('fwhm_bound', [0.2, 2])
    alpha_bound = kwargs.get('alpha_bound', [1, 10])
    dd_bound = kwargs.get('dd_bound', [-2, 2])
    parallelize = kwargs.get('parallelize', False)
    nproc = kwargs.get('nproc', 8)

    logl_kwargs = dict(convolved=convolved,
                       reference=reference,
                       starmask=starmask,
                       fxx=fxx,
                       fyy=fyy,
                       arrayslices=arrayslices,
                       edge=edge,
                       alpha0=alpha,
                       scale=scale)
    ptform_args = [fwhm_bound, alpha_bound, dd_bound]

    # it is possible to fit the alpha parameter or to assume a fixed value
    if offset and fit_alpha:
        ndim = 5
        labels=["fwhm", "dx", "dy", "alpha", "sigma"]
    elif offset and not fit_alpha:
        ndim = 4
        labels=["fwhm", "dx", "dy", "sigma"]
    elif not offset and fit_alpha:
        ndim = 3
        labels=["fwhm", "alpha", "sigma"]
    elif not offset and not fit_alpha:
        ndim = 2
        labels=["fwhm", "sigma"]

    if parallelize:
        print('Parallelizing computation')
        with Pool(nproc, lnlike, ptform, logl_kwargs=logl_kwargs, ptform_args=ptform_args) as pool:
            sampler = DynamicNestedSampler(pool.loglike, pool.prior_transform, ndim=ndim,
                                           pool=pool, use_pool={'prior_transform': True,
                                                                'loglikelihood': True,
                                                                'propose_point':True,
                                                                'update_bound':True})
            sampler.run_nested(print_progress=True, dlogz_init=0.01,
                               nlive_init=250, nlive_batch=50, checkpoint_every=120,
                               maxiter_init=20000, maxiter_batch=500, maxbatch=10)
    else:
        sampler = DynamicNestedSampler(lnlike, ptform, ndim=ndim, nlive=250,
                                       logl_kwargs=logl_kwargs, ptform_args=ptform_args)
        sampler.run_nested(print_progress=True, dlogz_init=0.01,
                           nlive_init=250, nlive_batch=50, checkpoint_every=120,
                           maxiter_init=20000, maxiter_batch=500, maxbatch=10)

    results = sampler.results
    weights = np.exp(results.logwt - results.logz[-1])
    samples = results.samples
    quantiles = [dyfunc.quantile(samps, [0.16, 0.5, 0.84], weights=weights)
                    for samps in samples.T]

    if plot:
        fig = dyplot.cornerplot(results, \
                labels=labels,
                show_titles=True,
                title_kwargs={"fontsize": 10},
                label_kwargs={"fontsize": 14},
                # data_kwargs={"ms": 0.6},
                quantiles=[0.16, 0.5, 0.84])
        if save:
            plt.savefig(figname.replace('_final.png', '_corner.png'), dpi=200)
            plt.close()
        else:
            plt.show()

    best_fit = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), quantiles))

    print('Fit completed')
    print(f'Measured FWHM = {best_fit[0][0]:0.2f} +{best_fit[0][1]:0.2f} -{best_fit[0][2]:0.2f}')
    if offset:
        print(f'Measured offset x:{best_fit[1][0]:0.2f} +{best_fit[1][1]:0.2f} -{best_fit[1][2]:0.2f}')
        print(f'Measured offset y:{best_fit[2][0]:0.2f} +{best_fit[2][1]:0.2f} -{best_fit[2][2]:0.2f}')
    if fit_alpha:
        print(f'Measured alpha = {best_fit[-2][0]:0.2f} +{best_fit[-2][1]:0.2f} -{best_fit[-2][2]:0.2f}')
    print(f'Measured sigma = {best_fit[-1][0]:0.2f} +{best_fit[-1][1]:0.2f} -{best_fit[-1][2]:0.2f}')

    if plot:

        params = [item[0] for item in best_fit]
        plot_results(params, convolved, reference, starmask, fxx, fyy, arrayslices, save=save, show=show,
                     figname=figname, edge=edge, alpha0=alpha)

    return best_fit, star_pos, starmask