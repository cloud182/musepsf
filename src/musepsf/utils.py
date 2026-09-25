from mpl_toolkits.axes_grid1 import make_axes_locatable

import astropy.visualization as vis
import matplotlib.pyplot as plt
import numpy as np

from astropy.io import ascii, fits
from astropy.wcs import WCS
from astropy.convolution import Moffat2DKernel, convolve_fft
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from astropy.modeling import models, fitting
from scipy.optimize import least_squares
from scipy.ndimage import zoom, binary_dilation, binary_fill_holes
from scipy.ndimage import maximum_filter, gaussian_filter
from numpy.fft import fftfreq
from scipy.spatial import cKDTree
from scipy.odr import ODR, Model, RealData

from dataclasses import dataclass
from itertools import product

import logging
import wget
import os
import shutil

from .plotting import _finalize_plot, plot_images

logger = logging.getLogger(__name__)

SDSS_URL = "https://data.sdss.org/sas/dr13/eboss/photo/redux"

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

    # deferred import: astroquery.gaia reaches out to the Gaia archive as a
    # side effect of being imported, so this stays lazy until query_gaia is
    # actually called instead of running at package import time
    from astroquery.gaia import Gaia

    Gaia.ROW_LIMIT = 10000
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

    r = Gaia.query_object_async(coordinate=center, radius=radius)
    r = r['ra', 'dec', 'parallax', 'phot_g_mean_mag', 'classprob_dsc_combmod_star'].copy()
    mask = np.abs(r['classprob_dsc_combmod_star']) > 0.99
    r = r[mask].copy()
    return r


def bin_image(image, bin_size=15, zeromask=None):
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

    if zeromask is not None:
        image = image.copy()
        image[zeromask] = 0

    n_y = image.shape[0] // bin_size
    n_x = image.shape[1] // bin_size
    blocks = image[:n_y*bin_size, :n_x*bin_size].reshape(n_y, bin_size, n_x, bin_size)

    median = np.median(blocks, axis=(1, 3)).ravel()
    std = np.std(blocks, axis=(1, 3)).ravel()

    return median, std

def linear_function(B, x):
    """ Linear function for the ODR fitting routine"""
    return B[0]*x + B[1]

def locate_stars(image, filename=None, **kwargs):
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

    # checking if there are manual entries; if so, skip the (potentially expensive) star search
    if filename is not None:
        stars = ascii.read(filename, names=['xcentroid', 'ycentroid'])
    else:
        # Define a threshold to look for stars
        mean, median, std = sigma_clipped_stats(image, sigma=sigma)
        thresh = mean + 10. * std

        # Initializing and starting the starfinder
        starfinder = DAOStarFinder(threshold=thresh, fwhm=fwhm, brightest=brightest)
        sources = starfinder(image)

        stars = sources['xcentroid', 'ycentroid'] if sources is not None else None

    if stars is not None:
        mask = np.zeros(image.shape, dtype=bool)
        yy, xx = np.mgrid[ :image.shape[0], :image.shape[1]]
        for star in stars:
            distance = np.sqrt((xx-star['xcentroid'])**2+(yy-star['ycentroid'])**2)
            mask[distance < radius] = True
    else:
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


def apply_mask(image, starmask, nanmask):
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
        if starmask.shape != image.shape:
            raise ValueError('Mask and image are of different shape')
        masked[starmask] = np.ma.masked

    return masked


def _unpack_fit_params(pars, alpha0=None):
    """
    Unpack the least_squares parameter vector shared by to_minimize() and plot_results().

    Depending on which of fwhm/offset/alpha are being fit, `pars` has length 1 (fwhm
    only), 2 (fwhm, alpha), 3 (fwhm, dx, dy) or 4 (fwhm, dx, dy, alpha).
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

    return fwhm, alpha, dx, dy


@dataclass
class FitContext:
    """
    Precomputed, per-call state shared by to_minimize(), plot_results() and rmse_test()'s grid
    scan: the star/zero masks, the FFT frequency grid and array slices apply_offset_fourier()
    needs, and the parameter bounds. Built once per run_measure_psf()/rmse_test() call by
    _prepare_fit_arrays(), then threaded through unchanged -- replaces what used to be 8+
    separate positional arguments passed by hand to each of those functions.
    """
    starmask: np.ndarray
    nanmask: np.ndarray
    fxx: np.ndarray
    fyy: np.ndarray
    arrayslices: list
    oversample: int
    scale: float
    fwhm_bound: list
    alpha_bound: list
    dd_bound: list


def _pack_bounds(fit_alpha, offset, fwhm_bound, alpha_bound, dd_bound):
    """
    Build the (lower, upper) bound sequences for least_squares(), matching the same
    fwhm/[alpha]/[dx, dy] parameter ordering that _unpack_fit_params() decodes and that
    run_measure_psf() uses to build the initial guess p0, for a given combination of
    fit_alpha/offset.
    """
    if offset:
        if fit_alpha:
            lower = [fwhm_bound[0], dd_bound[0], dd_bound[0], alpha_bound[0]]
            upper = [fwhm_bound[1], dd_bound[1], dd_bound[1], alpha_bound[1]]
        else:
            lower = [fwhm_bound[0], dd_bound[0], dd_bound[0]]
            upper = [fwhm_bound[1], dd_bound[1], dd_bound[1]]
    else:
        if fit_alpha:
            lower = [fwhm_bound[0], alpha_bound[0]]
            upper = [fwhm_bound[1], alpha_bound[1]]
        else:
            lower = [fwhm_bound[0]]
            upper = [fwhm_bound[1]]

    return lower, upper


def _covariance_from_least_squares(res):
    """
    Parameter covariance matrix from a scipy.optimize.least_squares() result, following the
    same convention as scipy.optimize.curve_fit(absolute_sigma=False):
    cov = inv(J^T J) * (2*cost/dof). Returns a matrix of NaN (same shape) if there are not
    enough data points to constrain the fit (dof <= 0) or if J^T J is singular.
    """
    npars = res.x.size
    dof = res.fun.size - npars

    if dof <= 0:
        return np.full((npars, npars), np.nan)

    s_sq = 2 * res.cost / dof
    try:
        return np.linalg.inv(res.jac.T @ res.jac) * s_sq
    except np.linalg.LinAlgError:
        return np.full((npars, npars), np.nan)


def to_minimize(pars, convolved, reference, ctx, alpha0=None):
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
        ctx (FitContext):
            precomputed masks/FFT grid/bounds shared across calls, see FitContext and
            _prepare_fit_arrays().
        alpha0 (float, optional):
            Value of alpha if not among the fittted parameters. Defaults to None.

    Returns:
        np.ndarray:
            1D array of residuals for the unmasked pixels only (masked pixels -- star mask,
            edges, zeros -- are dropped via .compressed(), not just zeroed out).
    """

    fwhm, alpha, dx, dy = _unpack_fit_params(pars, alpha0)

    factor = 1 #this is a factor that will be used to return very high numbers if the
                #parameters are out of bounds

    if ctx.fwhm_bound is not None:
        if fwhm < ctx.fwhm_bound[0] or fwhm > ctx.fwhm_bound[1]:
            fwhm = 0.4
            factor = 1e10
    if ctx.alpha_bound is not None:
        if alpha < ctx.alpha_bound[0] or alpha > ctx.alpha_bound[1]:
            alpha = 2
            factor = 1e10
    if ctx.dd_bound is not None:
         if dx < ctx.dd_bound[0] or dx > ctx.dd_bound[1]:
            dx = 0
            factor = 1e10
         if dy < ctx.dd_bound[0] or dy > ctx.dd_bound[1]:
            dy = 0
            factor = 1e10


    # creating model of MUSE PSF
    # img_size must be odd: convolve_fft() centers an even-sized kernel array at a half-pixel
    # offset from the true center (verified empirically -- see KNOWN_ISSUES.md #14), which would
    # otherwise leak into the fitted dx/dy as a spurious ~0.5 pixel bias.
    ker_MUSE = moffat_kernel(fwhm, alpha, scale=ctx.scale, img_size=50*ctx.oversample + 1)

    # convolving WFI image for the model of MUSE PSF
    reference_conv = convolve_fft(reference, ker_MUSE, return_fft=True)

    if ctx.oversample > 1:
        reference_conv = apply_offset_fourier(reference_conv, dx*ctx.oversample, dy*ctx.oversample,
                                              ctx.fxx, ctx.fyy, ctx.arrayslices)
    else:
        reference_conv = apply_offset_fourier(reference_conv, dx, dy, ctx.fxx, ctx.fyy, ctx.arrayslices)

    if ctx.oversample > 1:
        reference_conv = rebin(reference_conv, ctx.oversample)

    if reference_conv.shape != ctx.starmask.shape:
        raise ValueError('Starmask and reference_conv have different shapes')
    if reference_conv.shape != convolved.shape:
        raise ValueError('Convolved and reference_conv have different shapes')

    ref_masked = apply_mask(reference_conv, ctx.starmask, ctx.nanmask)

    # the optimizer needs a plain array of residuals: .compressed() drops the masked
    # pixels entirely (star mask / edges / zeros), rather than just hiding them behind
    # a MaskedArray. A MaskedArray here would NOT actually exclude them from the fit --
    # leastsq/least_squares coerce the return value with np.asarray(), which silently
    # discards the mask and keeps the underlying (masked) data values. See KNOWN_ISSUES.md #12.
    function = (convolved-ref_masked)

    return function.compressed() * factor

def apply_offset_fourier(convolved, dx, dy, fxx, fyy, arrayslices):

    ff = fyy*dy+fxx*dx

    convolved *= np.exp(-2j*np.pi*ff)

    convolved = np.fft.ifftn(convolved)

    return convolved[arrayslices[0]: arrayslices[1], arrayslices[2]: arrayslices[3]].real



def plot_results(pars, convolved, reference, ctx, figname, alpha0=None, save=False, show=False):
    """
    Functions that plot the final results of the PSF fitting

    Kept in utils.py rather than moved to plotting.py (Phase 4 of IMPLEMENTATION_PLAN.md): unlike
    the other plotting helpers, this one first reconstructs the fit model from `pars`/`ctx` using
    to_minimize()'s own machinery (moffat_kernel, apply_offset_fourier, rebin, apply_mask) before
    rendering it -- moving it would make plotting.py import back from utils.py, which already
    imports from plotting.py, creating a circular import.

    Args:
        pars (list):
            initial guess for the fitted parameters.
        convolved (np.ndarray):
            array for which the PSF will be measured, convolved for the model of the PSF of the
            reference image.
        reference (np.ndarray):
            reference image
        ctx (FitContext):
            precomputed masks/FFT grid/bounds shared across calls, see FitContext and
            _prepare_fit_arrays().
        figname (str):
            figure name.
        alpha0 (float, optional):
            Value of alpha if not among the fittted parameters. Defaults to None.
        save (bool, optional):
            if True, save the plot. Defaults to False.
        show (bool, optional):
            if True, show the plot. Defaults to False.
    """

    fwhm, alpha, dx, dy = _unpack_fit_params(pars, alpha0)

    # creating model of MUSE PSF (img_size must be odd, see the matching comment in to_minimize())
    ker_MUSE = moffat_kernel(fwhm, alpha, scale=ctx.scale, img_size=50*ctx.oversample + 1)

    # convolving WFI image for the model of MUSE PSF
    reference_conv = convolve_fft(reference, ker_MUSE, return_fft=True)
    reference_conv = apply_offset_fourier(reference_conv, dx, dy, ctx.fxx, ctx.fyy, ctx.arrayslices)

    if ctx.oversample > 1:
        reference_conv = rebin(reference_conv, ctx.oversample)

    ref_masked = apply_mask(reference_conv, ctx.starmask, ctx.nanmask)

    # plotting the results of the convolution

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax1.set_title('MUSE')
    ax2.set_title('WFI')
    ax3.set_title('Diff')

    # compute limits for trimming empty area in plotting
    conv_y = convolved.sum(axis=0)
    conv_x = convolved.sum(axis=1)

    [y0, y1] = np.ma.flatnotmasked_edges(conv_y)
    [x0, x1] = np.ma.flatnotmasked_edges(conv_x)

    # normalization for a better plot
    interval = vis.PercentileInterval(99.9)
    vmin, vmax = interval.get_limits(convolved[y0:y1, x0:x1])
    norm = vis.ImageNormalize(vmin=vmin, vmax=vmax,
                            stretch=vis.LogStretch(1000))

    # MUSE image
    img1 = ax1.imshow(convolved[y0:y1, x0:x1], norm=norm, origin='lower')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)

    # WFI image
    img2 = ax2.imshow(ref_masked[y0:y1, x0:x1], norm=norm, origin='lower')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)

    # Difference image
    img3 = ax3.matshow(convolved[y0:y1, x0:x1]/ref_masked[y0:y1, x0:x1], vmin=0.8,
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
    plt.tight_layout()
    _finalize_plot(fig, figname, save=save, show=show, dpi=150)

def _prepare_fit_arrays(reference, zeromask, starmask, edge, scale, oversample=1,
                        fwhm_bound=[0.4, 2], alpha_bound=[1, 10], dd_bound=[-2, 2]):
    """
    Shared setup for the PSF-fitting minimization, used by both run_measure_psf() and
    rmse_test(): fills mask holes, dilates the zero/edge mask by `edge` pixels, and builds
    the FFT frequency grid and array slices needed by apply_offset_fourier(). Returns a
    FitContext bundling all of that together with `scale` and the parameter bounds -- the
    fwhm_bound/alpha_bound/dd_bound defaults here match to_minimize()'s previous defaults, used
    whenever a caller (e.g. rmse_test()) doesn't override them explicitly.
    """

    # filling eventual holes in the masks caused by the resampling
    zeromask = binary_fill_holes(zeromask)
    if starmask is not None:
        starmask = binary_fill_holes(starmask)
    else:
        starmask = np.zeros_like(zeromask)

    # set the edges to zero
    zeromask[0, :] = True
    zeromask[-1, :] = True
    zeromask[:, 0] = True
    zeromask[:, -1] = True

    if edge != 0:
        strct_array = np.ones((2*edge+1, 2*edge+1), dtype=bool)
        zeromask = binary_dilation(zeromask, structure=strct_array)

    # computing things to perform the minimization more efficiently
    # creating model of MUSE PSF
    # I need this to make sure I create the fxx and fyy correctly.
    # maybe it could be removed?
    # img_size must match to_minimize()/plot_results() exactly (odd, see comment there) so the
    # fxx/fyy/arrayslices derived from its shape here stay valid for their own kernels.
    ker_MUSE = moffat_kernel(1, 2.8, scale=scale, img_size=50*oversample + 1)
    # convolving WFI image for the model of MUSE PSF This is needed only to compute the array slices
    reference_conv = convolve_fft(reference, ker_MUSE, return_fft=True)

    fx = fftfreq(reference_conv.shape[1])
    fy = fftfreq(reference_conv.shape[0])

    fxx, fyy = np.meshgrid(fx, fy)

    arrayslices = []
    for dimension_conv, dimension in zip(reference_conv.shape, reference.shape):
        center = dimension_conv - (dimension_conv + 1) // 2
        arrayslices += [center - dimension // 2, center + (dimension + 1) // 2]

    return FitContext(starmask=starmask, nanmask=zeromask, fxx=fxx, fyy=fyy, arrayslices=arrayslices,
                      oversample=oversample, scale=scale, fwhm_bound=fwhm_bound,
                      alpha_bound=alpha_bound, dd_bound=dd_bound)


def run_measure_psf(data, reference, psf, star_pos, starmask, zeromask, oversample, figname=None, alpha=2.8,
                    edge=50, fwhm0=0.8, dx0=0, dy0=0, fit_alpha=False, plot=False, save=False,
                    show=False, scale=0.2, offset=False, **kwargs):
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
        scipy.optimize.OptimizeResult:
            full result of the fit (from scipy.optimize.least_squares), with an extra `.cov`
            attribute holding the parameter covariance matrix (see _covariance_from_least_squares).
        (astropy.table.Table, None):
            Table contasining the position of the masked stars
        np.ndarray:
            boolean mask selecting the pixels associated to stellar emission that should be masked.
    """

    fwhm_bound = kwargs.get('fwhm_bound', [0.2, 2])
    alpha_bound = kwargs.get('alpha_bound', [1, 10])
    dd_bound = kwargs.get('dd_bound', [-2*oversample, 2*oversample])

    ctx = _prepare_fit_arrays(reference, zeromask, starmask, edge, scale, oversample=oversample,
                              fwhm_bound=fwhm_bound, alpha_bound=alpha_bound, dd_bound=dd_bound)

    # convolving MUSE for the reference PSF
    convolved = convolve_fft(data, psf)

    if oversample > 1:
        convolved = rebin(convolved, oversample)

    if convolved.shape != ctx.starmask.shape:
        raise ValueError('Convolved and starmask have different shapes')

    convolved = apply_mask(convolved, ctx.starmask, ctx.nanmask)

    logger.info(f'Using alpha = {alpha}')

    logger.info('Performing the fit')
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

    bounds = _pack_bounds(fit_alpha, offset, fwhm_bound, alpha_bound, dd_bound)

    res = least_squares(to_minimize, x0=p0, bounds=bounds,
                args=(convolved, reference, ctx, alpha),
                max_nfev=600, xtol=1e-9)
    res.cov = _covariance_from_least_squares(res)

    best_fit = res.x

    logger.info('Fit completed')
    logger.info(f'Measured FWHM = {best_fit[0]:0.3f}')
    if offset:
        logger.info(f'Measured offset x:{best_fit[1]:0.3f} y:{best_fit[2]:0.3f}')
    if fit_alpha:
        logger.info(f'Measured alpha = {best_fit[-1]:0.3f}')

    if plot:
        plot_results(best_fit, convolved, reference, ctx, figname, alpha0=alpha, save=save, show=show)

    return res, star_pos, ctx.starmask


def download_ps_file(run, camcol, frame, rerun, out_dir="ps_field"):
    """Download psFile given various identifying info"""

    ps_url = f"{SDSS_URL}/{rerun}/{int(run)}/objcs/{camcol}/psField-{run:06}-{camcol}-{frame:04}.fit"
    wget.download(ps_url, out=out_dir)


    return True


def reconstruct_psf(ps_file, sdss_filter, row, col):
    """Reconstruct PSF from psFile"""

    filter_idx = 'ugriz'.index(sdss_filter) + 1
    ps_field = fits.open(ps_file)
    ps = ps_field[filter_idx].data

    nrow_b = ps['nrow_b'][0]
    ncol_b = ps['ncol_b'][0]

    rnrow = ps['rnrow'][0]
    rncol = ps['rncol'][0]

    nb = nrow_b * ncol_b
    coeffs = np.zeros(nb.size, float)
    ecoeff = np.zeros(3, float)
    cmat = ps['c']

    rcs = 0.001
    for ii in range(0, nb.size):
        coeffs[ii] = (row * rcs) ** (ii % nrow_b) * (col * rcs) ** (ii / nrow_b)

    for jj in range(0, 3):
        for ii in range(0, nb.size):
            ecoeff[jj] = ecoeff[jj] + cmat[int(ii / nrow_b), ii % nrow_b, jj] * coeffs[ii]

    psf = ps['rrows'][0] * ecoeff[0] + ps['rrows'][1] * ecoeff[1] + ps['rrows'][2] * ecoeff[2]

    psf = np.reshape(psf, (rnrow, rncol))

    return psf


def create_sdss_psf(data, hdr, out_dir, pixscale=0.2, sdss_pixscale=0.396):
    """Read in an SDSS tile and generate PSF at centre of chip"""

    tmp_output = os.path.join(out_dir, 'tmp')
    if not os.path.isdir(tmp_output):
        os.mkdir(tmp_output)

    # We'll just evaluate the PSF at the chip centre
    row, col = np.asarray(data.shape) / 2

    # Pull out filter, run, rerun, camcol, and frame from the header

    PSFs = []

    for i in range(1, 30):
        try:
            file_string = hdr[f'FILE{i:04d}'][:-5].split('-')
        except KeyError:
            break

        sdss_filter = file_string[1]
        run = file_string[2]
        camcol = file_string[3]
        frame = file_string[4]

        # Build the filename
        ps_file = os.path.join(tmp_output, f"psField-{run}-{camcol}-{frame}.fit")

        if not os.path.exists(ps_file):
            download_ps_file(run=run, camcol=camcol, frame=frame, rerun=301,
                            out_dir=tmp_output)

        psf = reconstruct_psf(ps_file, sdss_filter=sdss_filter, row=row, col=col)

        # If the image isn't in native SDSS pixel scale, resample here
        if pixscale != sdss_pixscale:
            psf = zoom(psf, zoom=sdss_pixscale / pixscale)

        # Ensure PSF is normalised
        psf /= np.nansum(psf)

        PSFs.append(psf)

    shutil.rmtree(tmp_output, ignore_errors=True)


    psf = np.mean(PSFs, axis=0)

    return psf


def rebin(image, factor):

    sigma = factor / 2.0
    filtered = gaussian_filter(image, sigma=sigma) #try to removes high frequencies and improve results
    newimage = zoom(filtered, 1/factor, order=3) * factor**2
    return newimage


def find_peaks_2d(data, threshold):
    """
    Find the peaks in a 2D image.

    Args:
        data (np.ndarray): The 2D image.
        threshold (float): Minimum value to consider a peak.

    Returns:
        list: List of (y, x) coordinates of the peaks.
    """
    # apply a maximum filter to find the local maxima
    neighborhood = maximum_filter(data, size=3)
    peaks = (data == neighborhood) & (data > threshold)

    peak_coords = np.argwhere(peaks)

    return peak_coords

def remove_close_stars(coords):
    # Convert coordinates to a 2D array (RA, Dec in degrees)
    coords_array = np.vstack((coords.ra.deg, coords.dec.deg)).T

    # Build a KDTree for efficient distance computation
    tree = cKDTree(coords_array)

    # Find all neighbors within the minimum distance
    min_distance = 3.5 / 3600.0  # Convert arcseconds to degrees
    pairs = tree.query_pairs(min_distance)

    # Create a set of indices to remove
    indices_to_remove = set(i for i, _ in pairs).union(j for _, j in pairs)
    logger.info(f"Removing {len(indices_to_remove)} close stars")

    # Filter the catalog and coordinates
    mask = np.array([i not in indices_to_remove for i in range(len(coords))])
    return mask


def fit_2D_psf(data, mode='moffat'):

    yy, xx = np.indices(data.shape)

    fitter = fitting.LevMarLSQFitter()

    if mode == 'moffat':
        model = models.Moffat2D(amplitude=1, x_0=data.shape[1]/2, y_0=data.shape[0]/2,
                                gamma=1, alpha=2.5)
        fit = fitter(model, xx, yy, data)

        return fit.fwhm

    if mode == 'gaussian':
        model = models.Gaussian2D(amplitude=1, x_mean=data.shape[1]/2, y_mean=data.shape[0]/2,
                                   x_stddev=1, y_stddev=1)
        fit = fitter(model, xx, yy, data)

        return np.mean([fit.x_fwhm, fit.y_fwhm])

    if mode == 'empirical':

        # oversample image
        image_oversampled = zoom(data, 10, order=3)

        # find the peak value of the PSF
        max_value = np.nanmax(image_oversampled)

        # find the peak position
        max_coords = np.unravel_index(np.nanargmax(image_oversampled), image_oversampled.shape)

        # create a radial profile
        y, x = np.indices(image_oversampled.shape)
        r = np.sqrt((x - max_coords[1])**2 + (y - max_coords[0])**2)
        r = r.astype(int)

        tbin = np.bincount(r.ravel(), image_oversampled.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin / nr

        # find the half maximum value
        half_max = max_value / 2.0

        # measure the FWHM
        fwhm = 2 * np.interp(half_max, radialprofile[::-1], np.arange(len(radialprofile))[::-1])

        return fwhm / 10


def check_flux_calibration(muse, reference, zeromask=None, bin_size=15, plot=False,
                           save=False, show=True, outname=None, outname_scatter=None):
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
        resample (bool, optional):
            If True, the reference image will be resampled to match the MUSE WCS. Defaults to False.
    """

    MUSE_median, MUSE_std = bin_image(muse, bin_size, zeromask)
    reference_median, reference_std = bin_image(reference, bin_size)

    #removing nans and 0s
    valid_mask = ~np.isnan(MUSE_median) & ~np.isnan(reference_median) & (MUSE_median != 0) & (reference_median != 0)

    MUSE_median = MUSE_median[valid_mask]
    MUSE_std = MUSE_std[valid_mask]
    reference_median = reference_median[valid_mask]
    reference_std = reference_std[valid_mask]

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
            plt.savefig(outname_scatter)
        if show:
            plt.show()
        else:
            plt.close()

    logger.info('Residual flux correction.')
    logger.info('Gamma: {:0.3f}, b: {:0.3f}' .format(output.beta[0], output.beta[1]))

    orig = muse/reference

    new_muse = output.beta[0]*muse + output.beta[1]

    # making sure that pixels outside the image stays zero
    rescaled = new_muse/reference
    if zeromask is not None:
        new_muse[zeromask] = np.nan
        rescaled[zeromask] = np.nan
        orig[zeromask] = np.nan

    print('Correction applied.')
    if plot:
        plot_images(orig, rescaled, 'Original', 'Rescaled,', outname,
                    save=save, show=show)
    return new_muse

def rmse_test(data, reference, psf, starmask, zeromask, fwhm0, dx0, dy0, figname=None,
              alpha=2.8, edge=50, save=False, show=False, scale=0.2, **kwargs):

    zeromask, starmask, fxx, fyy, arrayslices = _prepare_fit_arrays(
        reference, zeromask, starmask, edge, scale, oversample=1)

    # convolving MUSE for the reference PSF
    convolved = convolve_fft(data, psf)

    if convolved.shape != starmask.shape:
        raise ValueError('Convolved and starmask have different shapes')

    convolved = apply_mask(convolved, starmask, zeromask)

    delta_off = kwargs.get('delta_off', 2)
    delta_fwhm = kwargs.get('delta_fwhm', 0.5)
    num_points = kwargs.get('num_points', 11)

    if num_points % 2 == 0:
        raise ValueError("num_points must be odd to guarantee zero is in the middle.")

    test_yx = np.linspace(-delta_off, delta_off, num_points)
    test_fwhm = np.linspace(-delta_fwhm, delta_fwhm, num_points)

    rmse_cube = np.zeros((num_points, num_points, num_points))

    # Iterates on all the combinations of translation and rotation
    for idx, (dy, dx, df) in enumerate(product(test_yx, test_yx, test_fwhm)):
        i, j, k = np.unravel_index(idx, (num_points, num_points, num_points))
        new_dx = dx0 + dx
        new_dy = dy0 + dy
        new_fhwm = fwhm0 + df
        out = to_minimize([new_fhwm, new_dy, new_dx], convolved, reference, starmask, zeromask,
                          fxx, fyy, arrayslices, 1, alpha, scale=0.2)

        rmse_cube[i, j, k] = np.sqrt(np.nanmean(out**2))

        # isolate the RMSE cube projections for 2 variables

    rmse_proj_yx = np.squeeze(rmse_cube[:, :, test_fwhm == 0])  # (y, x)
    rmse_proj_yf = np.squeeze(rmse_cube[:, test_yx == 0, :])   # (y) e rot
    rmse_proj_xf = np.squeeze(rmse_cube[test_yx == 0, :, :])   # (x) e rot

    # Plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    ax = axes[0]
    im = ax.matshow(rmse_proj_yx, origin='lower', extent=[-delta_off, delta_off, -delta_off, delta_off], cmap='viridis')
    ax.set_xlabel('x offset')
    ax.set_ylabel('y offset')
    ax.set_xticks(test_yx)
    ax.set_yticks(test_yx)
    ax.set_xticklabels([f"{val:.2f}" for val in test_yx])
    ax.set_yticklabels([f"{val:.2f}" for val in test_yx])
    ax.set_title('RMSE Projection (y, x)')
    fig.colorbar(im, ax=ax)

    ax = axes[1]
    im = ax.matshow(rmse_proj_yf, origin='lower', extent=[-delta_fwhm, delta_fwhm, -delta_off, delta_off], cmap='viridis')
    ax.set_xlabel('FWHM offset')
    ax.set_ylabel('y offset')
    ax.set_xticks(test_fwhm)
    ax.set_yticks(test_yx)
    ax.set_xticklabels([f"{val:.2f}" for val in test_fwhm])
    ax.set_yticklabels([f"{val:.2f}" for val in test_yx])
    ax.set_title('RMSE Projection (y, FWHM)')
    fig.colorbar(im, ax=ax)

    ax = axes[2]
    im = ax.matshow(rmse_proj_xf, origin='lower', extent=[-delta_fwhm, delta_fwhm, -delta_off, delta_off], cmap='viridis')
    ax.set_xlabel('FWHM offset')
    ax.set_ylabel('x offset')
    ax.set_xticks(test_fwhm)
    ax.set_yticks(test_yx)
    ax.set_xticklabels([f"{val:.2f}" for val in test_fwhm])
    ax.set_yticklabels([f"{val:.2f}" for val in test_yx])
    ax.set_title('RMSE Projection (x, FWHM)')
    fig.colorbar(im, ax=ax)

    # isolate the RMSE cube projections for single variables

    rmse_proj_x = np.squeeze(rmse_cube[test_yx==0, :, test_fwhm==0] ) # projection onto x offset
    rmse_proj_y = np.squeeze(rmse_cube[:, test_yx==0, test_fwhm==0] ) # projection onto y offset
    rmse_proj_f = np.squeeze(rmse_cube[test_yx==0, test_yx==0, :] ) # projection onto FWHM offset


    ax = axes[3]
    im = ax.plot(test_yx,  rmse_proj_x)
    ax.set_xlabel('x offset')
    ax.set_ylabel('RMSE x')
    ax.set_xticks(test_yx)
    ax.set_xticklabels([f"{val:.2f}" for val in test_yx])

    ax = axes[4]
    im = ax.plot(test_yx,  rmse_proj_y)
    ax.set_xlabel('y offset')
    ax.set_ylabel('RMSE y')
    ax.set_xticks(test_yx)
    ax.set_xticklabels([f"{val:.2f}" for val in test_yx])

    ax = axes[5]
    im = ax.plot(test_fwhm, rmse_proj_f)
    ax.set_xlabel('FWHM offset')
    ax.set_ylabel('RMSE rot')
    ax.set_xticks(test_fwhm)
    ax.set_xticklabels([f"{val:.2f}" for val in test_fwhm])

    plt.tight_layout()

    _finalize_plot(fig, figname, save=save, show=show, dpi=300)
