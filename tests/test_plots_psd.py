"""Test coverage for sgnligo.plots.psd module."""

import logging
from unittest.mock import Mock, patch

import numpy as np
import pytest


@pytest.fixture
def mock_coinc_xmldoc():
    """Create a mock coinc_xmldoc with necessary tables."""
    xmldoc = Mock()

    # Mock CoincTable
    coinc_event = Mock()
    coinc_table = Mock()
    coinc_table.get_table.return_value = [coinc_event]

    # Mock CoincInspiralTable
    coinc_inspiral = Mock()
    coinc_inspiral.end = 1234567890.5
    coinc_inspiral.ifos = set(["H1", "L1"])
    coinc_inspiral_table = Mock()
    coinc_inspiral_table.get_table.return_value = [coinc_inspiral]

    # Mock SnglInspiralTable
    sngl_h1 = Mock()
    sngl_h1.ifo = "H1"
    sngl_h1.mass1 = 1.4
    sngl_h1.mass2 = 1.3
    sngl_h1.snr = 10.5

    sngl_l1 = Mock()
    sngl_l1.ifo = "L1"
    sngl_l1.mass1 = 1.4
    sngl_l1.mass2 = 1.3
    sngl_l1.snr = 8.2

    sngl_inspiral_table = Mock()
    sngl_inspiral_table.get_table.return_value = [sngl_h1, sngl_l1]

    return xmldoc, coinc_table, coinc_inspiral_table, sngl_inspiral_table


@pytest.fixture
def mock_psds():
    """Create mock PSDs for testing."""
    psds = {}

    for ifo in ["H1", "L1", "V1"]:
        psd = Mock()
        psd.f0 = 0.0
        psd.deltaF = 1.0
        psd.data = Mock()
        # Create PSD data spanning 0-1000 Hz
        psd.data.data = np.random.rand(1001) * 1e-46
        psds[ifo] = psd

    # Add a None PSD to test that case
    psds["G1"] = None

    return psds


class TestSummarizeCoinc:
    """Test cases for summarize_coinc_xmldoc function."""

    @patch("sgnligo.plots.psd.lsctables")
    def test_summarize_coinc_xmldoc(self, mock_lsctables, mock_coinc_xmldoc, caplog):
        """Test summarize_coinc_xmldoc with normal input."""
        from sgnligo.plots.psd import summarize_coinc_xmldoc

        xmldoc, coinc_table, coinc_inspiral_table, sngl_inspiral_table = (
            mock_coinc_xmldoc
        )

        # Set up lsctables mocks
        mock_lsctables.CoincTable = coinc_table
        mock_lsctables.CoincInspiralTable = coinc_inspiral_table
        mock_lsctables.SnglInspiralTable = sngl_inspiral_table

        with caplog.at_level(logging.INFO):
            sngl_inspirals, mass1, mass2, end_time, on_instruments = (
                summarize_coinc_xmldoc(xmldoc)
            )

        # Check results
        assert len(sngl_inspirals) == 2
        assert "H1" in sngl_inspirals
        assert "L1" in sngl_inspirals
        assert mass1 == 1.4  # larger mass
        assert mass2 == 1.3  # smaller mass
        assert end_time == 1234567890.5
        assert on_instruments == set(["H1", "L1"])

        # Check logging
        assert "1.4 Msun -- 1.3 Msun event" in caplog.text

    @patch("sgnligo.plots.psd.lsctables")
    def test_summarize_coinc_xmldoc_mass_swap(self, mock_lsctables):
        """Test that masses are swapped so mass1 > mass2."""
        from sgnligo.plots.psd import summarize_coinc_xmldoc

        xmldoc = Mock()

        # Create coinc tables
        coinc_event = Mock()
        coinc_inspiral = Mock()
        coinc_inspiral.end = 1234567890.5
        coinc_inspiral.ifos = set(["H1"])

        # Create single inspiral with mass2 > mass1
        sngl_h1 = Mock()
        sngl_h1.ifo = "H1"
        sngl_h1.mass1 = 1.2  # smaller
        sngl_h1.mass2 = 1.5  # larger
        sngl_h1.snr = 10.0

        # Set up lsctables mocks
        mock_lsctables.CoincTable.get_table.return_value = [coinc_event]
        mock_lsctables.CoincInspiralTable.get_table.return_value = [coinc_inspiral]
        mock_lsctables.SnglInspiralTable.get_table.return_value = [sngl_h1]

        sngl_inspirals, mass1, mass2, end_time, on_instruments = summarize_coinc_xmldoc(
            xmldoc
        )

        # Check that masses were swapped
        assert mass1 == 1.5  # larger mass
        assert mass2 == 1.2  # smaller mass


class TestLatexHorizonDistance:
    """Test cases for latex_horizon_distance function."""

    def test_gpc_range(self):
        """Test conversion to Gpc for large distances."""
        from sgnligo.plots.psd import latex_horizon_distance

        with patch(
            "sgnligo.plots.util.latexnumber", side_effect=lambda x: f"latex({x})"
        ):
            result = latex_horizon_distance(1000.0)  # 1 Gpc
            assert "latex(1)" in result
            assert "Gpc" in result

    def test_mpc_range(self):
        """Test conversion to Mpc for typical distances."""
        from sgnligo.plots.psd import latex_horizon_distance

        with patch(
            "sgnligo.plots.util.latexnumber", side_effect=lambda x: f"latex({x})"
        ):
            result = latex_horizon_distance(100.0)
            assert "latex(100)" in result
            assert "Mpc" in result

    def test_kpc_range(self):
        """Test conversion to kpc for small distances."""
        from sgnligo.plots.psd import latex_horizon_distance

        with patch(
            "sgnligo.plots.util.latexnumber", side_effect=lambda x: f"latex({x})"
        ):
            result = latex_horizon_distance(0.001)  # 1 kpc
            assert "latex(1)" in result
            assert "kpc" in result

    def test_pc_range(self):
        """Test conversion to pc for very small distances."""
        from sgnligo.plots.psd import latex_horizon_distance

        with patch(
            "sgnligo.plots.util.latexnumber", side_effect=lambda x: f"latex({x})"
        ):
            result = latex_horizon_distance(1e-6)  # 1 pc
            assert "latex(1)" in result
            assert "pc" in result

    def test_boundary_values(self):
        """Test boundary values between units."""
        from sgnligo.plots.psd import latex_horizon_distance

        with patch(
            "sgnligo.plots.util.latexnumber", side_effect=lambda x: f"latex({x})"
        ):
            # Just below Gpc threshold
            assert "Mpc" in latex_horizon_distance(255.9)
            # At Gpc threshold
            assert "Gpc" in latex_horizon_distance(256.0)

            # Just below Mpc threshold
            assert "kpc" in latex_horizon_distance(0.24)
            # At Mpc threshold
            assert "Mpc" in latex_horizon_distance(0.25)

            # Just below kpc threshold (2**-12 ≈ 0.000244)
            assert "pc" in latex_horizon_distance(0.0002)
            # At kpc threshold
            assert "kpc" in latex_horizon_distance(0.00025)


class TestAxesPlotCumulativeSnr:
    """Test cases for axes_plot_cumulative_snr function."""

    @patch("sgnligo.plots.psd.HorizonDistance")
    @patch("sgnligo.plots.psd.summarize_coinc_xmldoc")
    @patch("sgnligo.plots.util.colour_from_instruments")
    def test_axes_plot_cumulative_snr(
        self, mock_colour, mock_summarize, mock_horizon_class, mock_psds, caplog
    ):
        """Test axes_plot_cumulative_snr with normal input."""
        from sgnligo.plots.psd import axes_plot_cumulative_snr

        # Set up mocks
        mock_colour.return_value = "red"

        sngl_inspirals = {"H1": Mock(snr=10.5), "L1": Mock(snr=8.2)}
        mock_summarize.return_value = (
            sngl_inspirals,
            1.4,
            1.3,
            1234567890.5,
            set(["H1", "L1"]),
        )

        # Mock HorizonDistance
        mock_horizon_instance = Mock()
        # Return arrays that will match the PSD range when clipped
        f_array = np.arange(10.0, 101.0, 1.0)  # 91 points: 10, 11, ..., 100
        strain_array = np.ones_like(f_array) * 1e-22
        mock_horizon_instance.return_value = (
            100.0,  # horizon distance in Mpc
            (f_array, strain_array),  # f, strain - same length arrays
        )
        mock_horizon_class.return_value = mock_horizon_instance

        # Create axes mock
        axes = Mock()
        axes.get_ylim.return_value = [0, 15]

        # Create more realistic PSDs with proper frequency coverage
        for ifo in ["H1", "L1"]:
            psd = mock_psds[ifo]
            # Ensure PSD covers the frequency range returned by HorizonDistance
            psd.data.data = np.ones(200) * 1e-46  # 0-199 Hz at 1 Hz spacing

        # Only include H1 and L1 in PSDs
        psds = {"H1": mock_psds["H1"], "L1": mock_psds["L1"]}

        with caplog.at_level(logging.INFO):
            axes_plot_cumulative_snr(axes, psds, Mock())

        # Verify grid and minorticks were set
        axes.grid.assert_called_once_with(which="both", linestyle="-", linewidth=0.2)
        axes.minorticks_on.assert_called_once()

        # Verify semilogx was called for each instrument
        assert axes.semilogx.call_count == 2

        # Verify labels and limits were set
        axes.set_ylim.assert_called()
        axes.set_title.assert_called_once()
        axes.set_xlabel.assert_called_once_with(r"Frequency (Hz)")
        axes.set_ylabel.assert_called_once_with(r"Cumulative SNR")
        axes.legend.assert_called_once_with(loc="upper left")

        # Check logging
        assert "found H1 event with SNR 10.5" in caplog.text
        assert "found L1 event with SNR 8.2" in caplog.text

    @patch("sgnligo.plots.psd.HorizonDistance")
    @patch("sgnligo.plots.psd.summarize_coinc_xmldoc")
    @patch("sgnligo.plots.util.colour_from_instruments")
    def test_axes_plot_cumulative_snr_missing_psd(
        self, mock_colour, mock_summarize, mock_horizon_class, mock_psds, caplog
    ):
        """Test axes_plot_cumulative_snr when PSD is missing for an instrument."""
        from sgnligo.plots.psd import axes_plot_cumulative_snr

        # Set up mocks
        mock_colour.return_value = "red"

        sngl_inspirals = {"H1": Mock(snr=10.5), "G1": Mock(snr=9.0)}  # G1 has None PSD
        mock_summarize.return_value = (
            sngl_inspirals,
            1.4,
            1.3,
            1234567890.5,
            set(["H1", "G1"]),
        )

        # Mock HorizonDistance
        mock_horizon_instance = Mock()
        # Return arrays that will match the PSD range when clipped
        f_array = np.arange(10.0, 101.0, 1.0)  # 91 points: 10, 11, ..., 100
        strain_array = np.ones_like(f_array) * 1e-22
        mock_horizon_instance.return_value = (
            100.0,  # horizon distance in Mpc
            (f_array, strain_array),  # f, strain - same length arrays
        )
        mock_horizon_class.return_value = mock_horizon_instance

        axes = Mock()
        axes.get_ylim.return_value = [0, 15]

        # Include H1 (with PSD) and G1 (with None PSD)
        psds = {"H1": mock_psds["H1"], "G1": mock_psds["G1"]}

        with caplog.at_level(logging.INFO):
            axes_plot_cumulative_snr(axes, psds, Mock())

        # Check that missing PSD was logged
        assert "no PSD for G1" in caplog.text

    @patch("sgnligo.plots.psd.summarize_coinc_xmldoc")
    def test_axes_plot_cumulative_snr_psd_not_in_dict(self, mock_summarize, caplog):
        """Test axes_plot_cumulative_snr when instrument not in PSD dict."""
        from sgnligo.plots.psd import axes_plot_cumulative_snr

        sngl_inspirals = {
            "H1": Mock(snr=10.5),
            "K1": Mock(snr=9.0),  # K1 not in PSD dict
        }
        mock_summarize.return_value = (
            sngl_inspirals,
            1.4,
            1.3,
            1234567890.5,
            set(["H1", "K1"]),
        )

        axes = Mock()
        axes.get_ylim.return_value = [0, 15]

        with caplog.at_level(logging.INFO):
            axes_plot_cumulative_snr(axes, {}, Mock())

        # Check that missing PSD was logged
        assert "no PSD for H1" in caplog.text


class TestAxesPlotPsds:
    """Test cases for axes_plot_psds function."""

    @patch("sgnligo.plots.psd.HorizonDistance")
    @patch("sgnligo.plots.psd.summarize_coinc_xmldoc")
    @patch("sgnligo.plots.util.colour_from_instruments")
    def test_axes_plot_psds_with_coinc(
        self, mock_colour, mock_summarize, mock_horizon_class, mock_psds
    ):
        """Test axes_plot_psds with coinc_xmldoc."""
        from sgnligo.plots.psd import axes_plot_psds

        # Set up mocks
        mock_colour.return_value = "red"

        sngl_inspirals = {"H1": Mock(snr=10.5), "L1": Mock(snr=8.2)}
        mock_summarize.return_value = (
            sngl_inspirals,
            1.4,
            1.3,
            1234567890.5,
            set(["H1", "L1"]),
        )

        # Mock HorizonDistance
        mock_horizon_instance = Mock()
        f_array = np.array([10.0, 50.0, 100.0])
        strain_array = np.ones_like(f_array) * 1e-22
        mock_horizon_instance.return_value = (
            100.0,  # horizon distance in Mpc
            (f_array, strain_array),  # f, strain
        )
        mock_horizon_class.return_value = mock_horizon_instance

        # Create axes mock with all necessary methods
        axes = Mock()
        axes.yaxis = Mock()

        axes_plot_psds(axes, mock_psds, coinc_xmldoc=Mock())

        # Verify grid and minorticks were set
        axes.grid.assert_called_once_with(which="both", linestyle="-", linewidth=0.2)
        axes.minorticks_on.assert_called_once()

        # Verify loglog was called for PSDs and inspiral spectra
        assert axes.loglog.call_count >= 3  # At least 3 PSDs

        # Verify labels and limits were set
        axes.set_xlim.assert_called()
        axes.set_ylim.assert_called()
        axes.set_title.assert_called_once()
        axes.set_xlabel.assert_called_once_with(r"Frequency (Hz)")
        axes.set_ylabel.assert_called_once()
        axes.legend.assert_called_once_with(loc="upper right")

    @patch("sgnligo.plots.psd.HorizonDistance")
    @patch("sgnligo.plots.util.colour_from_instruments")
    @patch("sgnligo.plots.psd.latex_horizon_distance")
    def test_axes_plot_psds_without_coinc(
        self, mock_latex_hd, mock_colour, mock_horizon_class, mock_psds
    ):
        """Test axes_plot_psds without coinc_xmldoc."""
        from sgnligo.plots.psd import axes_plot_psds

        # Set up mocks
        mock_colour.return_value = "red"
        mock_latex_hd.return_value = "100 Mpc"

        # Mock HorizonDistance
        mock_horizon_instance = Mock()
        mock_horizon_instance.return_value = (
            100.0,  # horizon distance in Mpc
            (
                np.array([10.0, 50.0, 100.0]),
                np.array([1e-23, 5e-23, 1e-22]),
            ),  # f, strain
        )
        mock_horizon_class.return_value = mock_horizon_instance

        # Create axes mock
        axes = Mock()
        axes.yaxis = Mock()

        axes_plot_psds(axes, mock_psds, coinc_xmldoc=None)

        # Verify defaults were used (1.4 solar mass binaries)
        mock_horizon_class.assert_called()
        call_args = mock_horizon_class.call_args[0]
        assert call_args[3] == 1.4  # mass1
        assert call_args[4] == 1.4  # mass2

        # Verify title doesn't include GPS time
        title_call = axes.set_title.call_args[0][0]
        assert "GPS" not in title_call

    @patch("sgnligo.plots.util.colour_from_instruments")
    def test_axes_plot_psds_empty_psds(self, mock_colour):
        """Test axes_plot_psds with empty PSD dict."""
        from sgnligo.plots.psd import axes_plot_psds

        axes = Mock()
        axes.yaxis = Mock()

        axes_plot_psds(axes, {}, coinc_xmldoc=None)

        # Should still set default xlim
        axes.set_xlim.assert_called_with((6.0, 3000.0))
        # Should not set ylim (no data)
        axes.set_ylim.assert_not_called()

    @patch("sgnligo.plots.psd.HorizonDistance")
    @patch("sgnligo.plots.psd.summarize_coinc_xmldoc")
    @patch("sgnligo.plots.util.colour_from_instruments")
    @patch("sgnligo.plots.psd.latex_horizon_distance")
    def test_axes_plot_psds_offline_instrument(
        self, mock_latex_hd, mock_colour, mock_summarize, mock_horizon_class, mock_psds
    ):
        """Test axes_plot_psds with offline instrument."""
        from sgnligo.plots.psd import axes_plot_psds

        # Set up mocks
        mock_colour.return_value = "red"
        mock_latex_hd.return_value = "100 Mpc"

        # Mock HorizonDistance
        mock_horizon_instance = Mock()
        mock_horizon_instance.return_value = (
            100.0,  # horizon distance in Mpc
            (
                np.array([10.0, 50.0, 100.0]),
                np.array([1e-23, 5e-23, 1e-22]),
            ),  # f, strain
        )
        mock_horizon_class.return_value = mock_horizon_instance

        # V1 is not in on_instruments
        sngl_inspirals = {"H1": Mock(snr=10.5)}
        mock_summarize.return_value = (
            sngl_inspirals,
            1.4,
            1.3,
            1234567890.5,
            set(["H1", "L1"]),
        )

        axes = Mock()
        axes.yaxis = Mock()

        # Include V1 in PSDs but not in on_instruments
        psds = {"H1": mock_psds["H1"], "V1": mock_psds["V1"]}

        axes_plot_psds(axes, psds, coinc_xmldoc=Mock())

        # Check that offline instrument was plotted with different style
        loglog_calls = axes.loglog.call_args_list
        # Find the call for V1
        v1_call_found = False
        for call in loglog_calls:
            if "linestyle" in call[1] and call[1]["linestyle"] == ":":
                v1_call_found = True
                assert call[1]["alpha"] == 0.6
                assert "Off, Last Seen With" in call[1]["label"]
                break
        assert v1_call_found


class TestPlotFunctions:
    """Test cases for high-level plotting functions."""

    @patch("sgnligo.plots.psd.FigureCanvas")
    @patch("sgnligo.plots.psd.figure.Figure")
    @patch("sgnligo.plots.psd.axes_plot_psds")
    @patch("sgnligo.plots.util.golden_ratio", 1.618)
    def test_plot_psds(self, mock_axes_plot, mock_figure_class, mock_canvas, mock_psds):
        """Test plot_psds function."""
        from sgnligo.plots.psd import plot_psds

        # Set up figure mock
        mock_fig = Mock()
        mock_fig.get_dpi.return_value = 100
        mock_axes = Mock()
        mock_fig.gca.return_value = mock_axes
        mock_figure_class.return_value = mock_fig

        result = plot_psds(mock_psds, plot_width=640)

        # Verify figure was created and configured
        mock_figure_class.assert_called_once()
        mock_canvas.assert_called_once_with(mock_fig)
        mock_fig.set_size_inches.assert_called_once()

        # Verify axes_plot_psds was called
        mock_axes_plot.assert_called_once_with(mock_axes, mock_psds, coinc_xmldoc=None)

        # Verify tight_layout was called
        mock_fig.tight_layout.assert_called_once_with(pad=0.8)

        assert result == mock_fig

    @patch("sgnligo.plots.psd.FigureCanvas")
    @patch("sgnligo.plots.psd.figure.Figure")
    @patch("sgnligo.plots.psd.axes_plot_cumulative_snr")
    @patch("sgnligo.plots.util.golden_ratio", 1.618)
    def test_plot_cumulative_snrs(
        self, mock_axes_plot, mock_figure_class, mock_canvas, mock_psds
    ):
        """Test plot_cumulative_snrs function."""
        from sgnligo.plots.psd import plot_cumulative_snrs

        # Set up figure mock
        mock_fig = Mock()
        mock_fig.get_dpi.return_value = 100
        mock_axes = Mock()
        mock_fig.gca.return_value = mock_axes
        mock_figure_class.return_value = mock_fig

        mock_coinc = Mock()

        result = plot_cumulative_snrs(mock_psds, mock_coinc, plot_width=800)

        # Verify figure was created and configured
        mock_figure_class.assert_called_once()
        mock_canvas.assert_called_once_with(mock_fig)
        mock_fig.set_size_inches.assert_called_once()

        # Check size calculation with plot_width=800
        call_args = mock_fig.set_size_inches.call_args[0]
        assert call_args[0] == 800 / 100.0  # width
        assert abs(call_args[1] - 800 / 1.618 / 100.0) < 0.01  # height (golden ratio)

        # Verify axes_plot_cumulative_snr was called
        mock_axes_plot.assert_called_once_with(mock_axes, mock_psds, mock_coinc)

        # Verify tight_layout was called
        mock_fig.tight_layout.assert_called_once_with(pad=0.8)

        assert result == mock_fig


class TestLoggingAndEdgeCases:
    """Test logging output and edge cases."""

    @patch("sgnligo.plots.psd.HorizonDistance")
    @patch("sgnligo.plots.psd.summarize_coinc_xmldoc")
    @patch("sgnligo.plots.util.colour_from_instruments")
    def test_psd_span_logging(
        self, mock_colour, mock_summarize, mock_horizon_class, mock_psds, caplog
    ):
        """Test that PSD frequency span is logged correctly."""
        from sgnligo.plots.psd import axes_plot_psds

        mock_colour.return_value = "red"
        mock_summarize.return_value = ({}, 1.4, 1.3, 1234567890.5, set(["H1"]))

        # Mock HorizonDistance
        mock_horizon_instance = Mock()
        mock_horizon_instance.return_value = (
            100.0,  # horizon distance in Mpc
            (
                np.array([10.0, 50.0, 100.0]),
                np.array([1e-23, 5e-23, 1e-22]),
            ),  # f, strain
        )
        mock_horizon_class.return_value = mock_horizon_instance

        axes = Mock()
        axes.yaxis = Mock()

        with caplog.at_level(logging.INFO):
            axes_plot_psds(axes, {"H1": mock_psds["H1"]}, coinc_xmldoc=Mock())

        # Check PSD span was logged
        assert "found PSD for H1 spanning [0 Hz, 1000 Hz]" in caplog.text

    @patch("sgnligo.plots.psd.HorizonDistance")
    @patch("sgnligo.plots.psd.summarize_coinc_xmldoc")
    def test_min_max_psd_calculation(self, mock_summarize, mock_horizon_class):
        """Test min/max PSD calculation for y-axis limits."""
        from sgnligo.plots.psd import axes_plot_psds

        mock_summarize.return_value = ({}, 1.4, 1.3, None, set())

        # Mock HorizonDistance
        mock_horizon_instance = Mock()
        mock_horizon_instance.return_value = (
            100.0,  # horizon distance in Mpc
            (
                np.array([10.0, 50.0, 100.0]),
                np.array([1e-23, 5e-23, 1e-22]),
            ),  # f, strain
        )
        mock_horizon_class.return_value = mock_horizon_instance

        # Create PSD with known values
        psd = Mock()
        psd.f0 = 0.0
        psd.deltaF = 1.0
        psd.data = Mock()
        # Create data where min is at 100 Hz and max is at 500 Hz
        data = np.ones(1001) * 1e-22
        data[100] = 1e-24  # min at 100 Hz
        data[500] = 1e-20  # max at 500 Hz
        psd.data.data = data

        axes = Mock()
        axes.yaxis = Mock()

        with patch("sgnligo.plots.util.colour_from_instruments", return_value="red"):
            axes_plot_psds(axes, {"H1": psd}, coinc_xmldoc=None)

        # Check that set_ylim was called with appropriate values
        axes.set_ylim.assert_called_once()
        ylim_args = axes.set_ylim.call_args[0][0]
        # Should include the min value (1e-24) with some margin
        assert ylim_args[0] < 1e-23
        # Should include the max value (1e-20)
        assert ylim_args[1] >= 1e-20
