import matplotlib.pyplot as plt
import numpy as np
import pytest

from musepsf.plotting import _draw_rmse_grid, plot_qc_summary
from musepsf.utils import QCResult


def _synthetic_rmse_cube():
    test_yx = np.linspace(-2, 2, 5)
    test_fwhm = np.linspace(-0.5, 0.5, 5)
    dy, dx, df = np.meshgrid(test_yx, test_yx, test_fwhm, indexing='ij')
    rmse_cube = np.sqrt(1 + (dx / 1.0)**2 + (dy / 1.5)**2 + (df / 0.15)**2)
    return rmse_cube, test_yx, test_fwhm


class TestDrawRmseGrid:
    def test_fwhm_axis_panels_use_auto_aspect_not_equal(self):
        # Regression test for KNOWN_ISSUES.md #16: matshow() defaults to aspect='equal', which
        # for the (y, FWHM)/(x, FWHM) panels' non-square extent (+-delta_off vs +-delta_fwhm)
        # squeezed the plotted heatmap into a narrow sliver of the panel and crammed its 11 tick
        # labels into unreadable overlapping text.
        rmse_cube, test_yx, test_fwhm = _synthetic_rmse_cube()
        fig = plt.figure()
        grid = _draw_rmse_grid(fig, rmse_cube, test_yx, test_fwhm, delta_off=2, delta_fwhm=0.5)
        _, ax_yf, ax_xf, _, _, _ = grid['axes']

        assert ax_yf.get_aspect() == 'auto'
        assert ax_xf.get_aspect() == 'auto'
        plt.close(fig)


class TestPlotQcSummary:
    def test_runs_without_error_for_a_passing_qc_result(self):
        rmse_cube, test_yx, test_fwhm = _synthetic_rmse_cube()
        qc = QCResult(passed=True, reasons=[])
        confidence = {'dx': (-0.8, 0.8), 'dy': (-1.2, 1.2), 'fwhm': (-0.1, 0.1)}

        plot_qc_summary(rmse_cube, test_yx, test_fwhm, qc, confidence, save=False, show=False)

    def test_runs_without_error_for_a_failing_qc_result_with_reasons(self):
        rmse_cube, test_yx, test_fwhm = _synthetic_rmse_cube()
        qc = QCResult(passed=False, reasons=[
            'RMSE grid minimum is at index (0, 0, 0), not at the best fit (grid center, (2, 2, 2)).',
            'Best-fit FWHM (1.970) is close to its bound (0.2, 2).',
        ])
        confidence = {'dx': (-0.8, 0.8), 'dy': (-1.2, 1.2), 'fwhm': (-0.1, 0.1)}

        plot_qc_summary(rmse_cube, test_yx, test_fwhm, qc, confidence, save=False, show=False)

    def test_runs_without_error_when_confidence_intervals_are_nan(self):
        # confidence_from_rmse_cube() returns (nan, nan) whenever the grid never crosses the
        # confidence threshold (very common for well-constrained fits on large images, see
        # KNOWN_ISSUES.md discussion) -- the shaded-band overlay must not choke on that.
        rmse_cube, test_yx, test_fwhm = _synthetic_rmse_cube()
        qc = QCResult(passed=True, reasons=[])
        confidence = {'dx': (np.nan, np.nan), 'dy': (np.nan, np.nan), 'fwhm': (np.nan, np.nan)}

        plot_qc_summary(rmse_cube, test_yx, test_fwhm, qc, confidence, save=False, show=False)

    def test_saves_a_figure_file_when_save_is_true(self, tmp_path):
        rmse_cube, test_yx, test_fwhm = _synthetic_rmse_cube()
        qc = QCResult(passed=True, reasons=[])
        confidence = {'dx': (-0.8, 0.8), 'dy': (-1.2, 1.2), 'fwhm': (-0.1, 0.1)}
        figname = str(tmp_path / 'qc.png')

        plot_qc_summary(rmse_cube, test_yx, test_fwhm, qc, confidence,
                        figname=figname, save=True, show=False)

        assert (tmp_path / 'qc.png').exists()
