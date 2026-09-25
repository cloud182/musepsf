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


def _draw_rmse_grid(fig, rmse_cube, test_yx, test_fwhm, delta_off, delta_fwhm):
    """
    Draw the 6-panel RMSE grid (2D slices (y,x)/(y,FWHM)/(x,FWHM) + 1D projections onto x/y/FWHM)
    onto a fresh set of axes added to `fig`. Shared by _plot_rmse_cube() and plot_qc_summary() so
    the QC overlay (markers, confidence bands, verdict) doesn't have to duplicate the
    panel-drawing code.

    Returns:
        dict: {'axes': flat array of the 6 Axes (order: yx, yf, xf, x, y, f), 'proj_x'/'proj_y'/
        'proj_f': the 1D RMSE projections onto x/y/FWHM offset -- returned so callers can locate
        and annotate their own minima on top of these axes}.
    """
    axes = fig.subplots(2, 3).flatten()

    rmse_proj_yx = np.squeeze(rmse_cube[:, :, test_fwhm == 0])  # (y, x)
    rmse_proj_yf = np.squeeze(rmse_cube[:, test_yx == 0, :])   # (y) e rot
    rmse_proj_xf = np.squeeze(rmse_cube[test_yx == 0, :, :])   # (x) e rot

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
    # aspect='auto': matshow() defaults to aspect='equal', which for this panel's non-square
    # extent (+-delta_fwhm vs +-delta_off) squeezes the plotted heatmap into a narrow sliver of
    # the panel and crams its tick labels into unreadable overlapping text (KNOWN_ISSUES.md #16).
    im = ax.matshow(rmse_proj_yf, origin='lower', extent=[-delta_fwhm, delta_fwhm, -delta_off, delta_off], cmap='viridis', aspect='auto')
    ax.set_xlabel('FWHM offset')
    ax.set_ylabel('y offset')
    ax.set_xticks(test_fwhm)
    ax.set_yticks(test_yx)
    ax.set_xticklabels([f"{val:.2f}" for val in test_fwhm])
    ax.set_yticklabels([f"{val:.2f}" for val in test_yx])
    ax.set_title('RMSE Projection (y, FWHM)')
    fig.colorbar(im, ax=ax)

    ax = axes[2]
    im = ax.matshow(rmse_proj_xf, origin='lower', extent=[-delta_fwhm, delta_fwhm, -delta_off, delta_off], cmap='viridis', aspect='auto')
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
    ax.plot(test_yx,  rmse_proj_x)
    ax.set_xlabel('x offset')
    ax.set_ylabel('RMSE x')
    ax.set_xticks(test_yx)
    ax.set_xticklabels([f"{val:.2f}" for val in test_yx])

    ax = axes[4]
    ax.plot(test_yx,  rmse_proj_y)
    ax.set_xlabel('y offset')
    ax.set_ylabel('RMSE y')
    ax.set_xticks(test_yx)
    ax.set_xticklabels([f"{val:.2f}" for val in test_yx])

    ax = axes[5]
    ax.plot(test_fwhm, rmse_proj_f)
    ax.set_xlabel('FWHM offset')
    ax.set_ylabel('RMSE rot')
    ax.set_xticks(test_fwhm)
    ax.set_xticklabels([f"{val:.2f}" for val in test_fwhm])

    return {'axes': axes, 'proj_x': rmse_proj_x, 'proj_y': rmse_proj_y, 'proj_f': rmse_proj_f}


def _plot_rmse_cube(rmse_cube, test_yx, test_fwhm, delta_off, delta_fwhm, figname,
                    save=False, show=False):
    """Diagnostic plot for an rmse_test() grid -- split out from rmse_test() in Phase 2 so the
    grid can be computed and used for quality control without a mandatory plotting step."""

    fig = plt.figure(figsize=(18, 12))
    _draw_rmse_grid(fig, rmse_cube, test_yx, test_fwhm, delta_off, delta_fwhm)
    plt.tight_layout()

    _finalize_plot(fig, figname, save=save, show=show, dpi=300)


def plot_qc_summary(rmse_cube, test_yx, test_fwhm, qc, confidence, figname=None,
                    save=False, show=False):
    """
    Visualize what qc_from_rmse_cube()/confidence_from_rmse_cube() actually found: the same
    6-panel RMSE grid as _plot_rmse_cube(), with the best fit (grid center) and each panel's own
    minimum marked, the confidence intervals from confidence_from_rmse_cube() shaded on the 1D
    projections, and a PASS/FAIL verdict with qc.reasons printed on the figure.

    Args:
        rmse_cube, test_yx, test_fwhm: as returned by rmse_test().
        qc (QCResult): as returned by qc_from_rmse_cube().
        confidence (dict): as returned by confidence_from_rmse_cube(), i.e.
            {'dx': (lo, hi), 'dy': (lo, hi), 'fwhm': (lo, hi)}.
        figname (str, optional): output figure path (only used if save=True).
        save, show (bool, optional): as in the other plotting helpers.
    """

    delta_off = float(np.max(test_yx))
    delta_fwhm = float(np.max(test_fwhm))

    fig = plt.figure(figsize=(18, 13))
    grid = _draw_rmse_grid(fig, rmse_cube, test_yx, test_fwhm, delta_off, delta_fwhm)
    ax_yx, ax_yf, ax_xf, ax_x, ax_y, ax_f = grid['axes']

    # best fit (grid center, i.e. offset 0) vs. each panel's own minimum
    center_style = dict(marker='+', color='white', markeredgecolor='black', markersize=16, mew=2)
    min_style = dict(marker='x', color='red', markersize=12, mew=2, linestyle='none')

    for ax, proj, grids in (
        (ax_yx, np.squeeze(rmse_cube[:, :, test_fwhm == 0]), (test_yx, test_yx)),
        (ax_yf, np.squeeze(rmse_cube[:, test_yx == 0, :]), (test_yx, test_fwhm)),
        (ax_xf, np.squeeze(rmse_cube[test_yx == 0, :, :]), (test_yx, test_fwhm)),
    ):
        ax.plot(0, 0, **center_style)
        y_grid, x_grid = grids
        min_idx = np.unravel_index(np.argmin(proj), proj.shape)
        min_y, min_x = y_grid[min_idx[0]], x_grid[min_idx[1]]
        if (min_y, min_x) != (0.0, 0.0):
            ax.plot(min_x, min_y, **min_style)

    # confidence intervals shaded on the 1D projections; a vertical line marks the best fit (0)
    for ax, key, grid_axis, proj in (
        (ax_x, 'dx', test_yx, grid['proj_x']),
        (ax_y, 'dy', test_yx, grid['proj_y']),
        (ax_f, 'fwhm', test_fwhm, grid['proj_f']),
    ):
        ax.axvline(0, color='black', linestyle=':', linewidth=1)
        lo, hi = confidence[key]
        if np.isfinite(lo) and np.isfinite(hi):
            ax.axvspan(lo, hi, color='tab:green', alpha=0.2,
                      label=f'{key} 1σ CI [{lo:.3f}, {hi:.3f}]')
            ax.legend(fontsize=8, loc='upper right')
        min_x = grid_axis[np.argmin(proj)]
        if min_x != 0.0:
            ax.axvline(min_x, color='red', linestyle='--', linewidth=1)

    verdict = 'PASS' if qc.passed else 'FAIL'
    color = 'tab:green' if qc.passed else 'tab:red'
    fig.suptitle(f'PSF fit quality control: {verdict}', color=color, fontsize=18, fontweight='bold')

    reasons_text = '\n'.join(f'- {reason}' for reason in qc.reasons) if qc.reasons else 'No issues detected.'
    fig.text(0.5, 0.01, reasons_text, ha='center', va='bottom', fontsize=9,
             color=color, wrap=True)

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    _finalize_plot(fig, figname, save=save, show=show, dpi=300)
