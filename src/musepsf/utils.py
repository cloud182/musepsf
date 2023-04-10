import astropy.visualization as vis
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Moffat2DKernel
from astropy.stats import sigma_clipped_stats
from astroquery.gaia import Gaia
from photutils.detection import DAOStarFinder

# configure astroquery gaia
Gaia.ROW_LIMIT = 10000
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

def query_gaia(center, radius):

    r = Gaia.query_object_async(coordinate=center, radius=radius)
    r = r['ra', 'dec', 'parallax', 'phot_g_mean_mag', 'classprob_dsc_combmod_star'].copy()
    mask = np.abs(r['classprob_dsc_combmod_star']) > 0.99
    r = r[mask].copy()
    return r

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
    radius = kwargs.get('radius', 20)

    # Define a threshold to look for stars
    mean, median, std = sigma_clipped_stats(image, sigma=sigma)
    thresh = mean + 10. * std

    # Initializing and starting the starfinder
    starfinder = DAOStarFinder(threshold=thresh, fwhm=fwhm, brightest=brightest)
    sources = starfinder(image)

    if sources is not None:
        yy, xx = np.mgrid[ :image.shape[0], :image.shape[1]]
        stars = sources['xcentroid', 'ycentroid']
        for star in stars:
            distance = np.sqrt((xx-star['xcentroid'])**2+(yy-star['ycentroid'])**2)
            mask = distance < radius
    else:
        stars = None
        mask = None

    return stars, mask

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

def apply_mask(image1, image2, starmask, edge=5, radius=20):

    if starmask is not None:
            assert starmask.shape == image1.shape, 'Mask and image1 are of different shape'
            assert starmask.shape == image2.shape, 'Mask and image2 are of different shape'
            image1[starmask] = np.nan
            image2[starmask] = np.nan

    if edge != 0:
        return image1[edge:-edge, edge:-edge], image2[edge:-edge, edge:-edge]
    else:
        return image1, image2