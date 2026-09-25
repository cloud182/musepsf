import os

import astropy.visualization as vis
import matplotlib.pyplot as plt
import numpy as np


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


def _finalize_plot(fig, outname, save=False, show=False, dpi=None):
    """
    Shared save/show/close boilerplate for the diagnostic plotting helpers below.
    dpi=None matches plt.savefig()'s own default (no explicit dpi override).
    """
    if save:
        if dpi is None:
            fig.savefig(outname)
        else:
            fig.savefig(outname, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)


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
    _finalize_plot(fig, name, save=save, show=show, dpi=300)


def plot_psf(data, output_dir, filename, residual=None, save=True, show=False, suffix=''):
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

    img = ax3.imshow(residual, origin='lower')#, vmin=0.8, vmax=1.2)
    plt.colorbar(img)
    ax1.set_title('PSF')
    ax2.set_title('PSF - 3D')
    ax3.set_title('Residuals')
    outname = os.path.join(output_dir, filename.replace('.fits', f'{suffix}.psf.png'))
    _finalize_plot(fig, outname, save=save, show=show, dpi=300)
