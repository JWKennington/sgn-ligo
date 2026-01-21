"""Test coverage for sgnligo.bin.coinc_to_html module."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import lal
import matplotlib
import numpy as np
import pytest
from lal import LIGOTimeGPS

from sgnligo.bin.coinc_to_html import (
    _fig_to_base64,
    _format_gps_time,
    _generate_html,
    _parse_coinc_xml,
    _plot_psd,
    _plot_snr_timeseries,
    main,
    parse_command_line,
)


@pytest.fixture(autouse=True)
def disable_latex_for_tests():
    """Disable LaTeX rendering for all tests in this module to avoid errors."""
    original_usetex = matplotlib.rcParams["text.usetex"]
    matplotlib.rcParams["text.usetex"] = False
    yield
    matplotlib.rcParams["text.usetex"] = original_usetex


@pytest.fixture
def mock_psd():
    """Create a mock LAL PSD frequency series."""
    psd = lal.CreateREAL8FrequencySeries(
        "test_psd",
        LIGOTimeGPS(0),
        10.0,  # f0
        1.0,  # deltaF
        lal.Unit("strain^2 s"),
        1000,
    )
    # Fill with realistic-looking PSD data
    freqs = psd.f0 + np.arange(psd.data.length) * psd.deltaF
    psd.data.data[:] = 1e-46 * (freqs / 100) ** (-4)  # Approximate shape
    psd.data.data[psd.data.data < 1e-48] = 1e-48  # Floor
    return psd


@pytest.fixture
def mock_snr_timeseries():
    """Create a mock LAL COMPLEX8 time series for SNR."""
    ts = lal.CreateCOMPLEX8TimeSeries(
        "snr_H1",
        LIGOTimeGPS(0),
        0.0,  # f0
        1.0 / 4096,  # deltaT
        lal.Unit(""),
        4096,  # length
    )
    # Create SNR-like data with a peak
    t = np.arange(ts.data.length) * ts.deltaT
    magnitude = 5.0 + 10.0 * np.exp(-((t - 0.5) ** 2) / 0.01)
    phase = np.linspace(-np.pi, np.pi, len(t))
    ts.data.data[:] = magnitude * np.exp(1j * phase)
    return ts


@pytest.fixture
def mock_coinc_inspiral():
    """Create a mock CoincInspiral object."""
    coinc = MagicMock()
    coinc.end_time = 1234567890
    coinc.end_time_ns = 123456789
    coinc.ifos = "H1,L1"
    coinc.snr = 15.5
    coinc.mchirp = 1.2
    coinc.mass = 2.8
    coinc.combined_far = 1e-10
    return coinc


@pytest.fixture
def mock_sngl_inspiral():
    """Create a mock SnglInspiral object."""
    trig = MagicMock()
    trig.ifo = "H1"
    trig.end_time = 1234567890
    trig.end_time_ns = 123456789
    trig.snr = 12.5
    trig.coa_phase = 0.5
    trig.mass1 = 1.4
    trig.mass2 = 1.4
    trig.mchirp = 1.2
    return trig


@pytest.fixture
def mock_coinc_data(
    mock_coinc_inspiral, mock_sngl_inspiral, mock_psd, mock_snr_timeseries
):
    """Create complete mock coinc data dictionary."""
    # Create a second trigger for L1
    trig_l1 = MagicMock()
    trig_l1.ifo = "L1"
    trig_l1.end_time = 1234567890
    trig_l1.end_time_ns = 123456780
    trig_l1.snr = 11.0
    trig_l1.coa_phase = 0.3
    trig_l1.mass1 = 1.4
    trig_l1.mass2 = 1.4
    trig_l1.mchirp = 1.2

    # Create SNR time series for L1
    ts_l1 = lal.CreateCOMPLEX8TimeSeries(
        "snr_L1",
        LIGOTimeGPS(0),
        0.0,
        1.0 / 4096,
        lal.Unit(""),
        4096,
    )
    t = np.arange(ts_l1.data.length) * ts_l1.deltaT
    magnitude = 5.0 + 8.0 * np.exp(-((t - 0.5) ** 2) / 0.01)
    phase = np.linspace(-np.pi, np.pi, len(t))
    ts_l1.data.data[:] = magnitude * np.exp(1j * phase)

    # Create PSDs for both detectors
    psd_l1 = lal.CreateREAL8FrequencySeries(
        "test_psd_l1",
        LIGOTimeGPS(0),
        10.0,
        1.0,
        lal.Unit("strain^2 s"),
        1000,
    )
    freqs = psd_l1.f0 + np.arange(psd_l1.data.length) * psd_l1.deltaF
    psd_l1.data.data[:] = 1e-46 * (freqs / 100) ** (-4)
    psd_l1.data.data[psd_l1.data.data < 1e-48] = 1e-48

    return {
        "coinc_inspiral": mock_coinc_inspiral,
        "triggers": [mock_sngl_inspiral, trig_l1],
        "psds": {"H1": mock_psd, "L1": psd_l1},
        "snr_series": {"H1": mock_snr_timeseries, "L1": ts_l1},
    }


class TestFigToBase64:
    """Test figure to base64 conversion."""

    def test_basic_conversion(self):
        """Test basic figure to base64 conversion."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        result = _fig_to_base64(fig)
        plt.close(fig)

        # Should return a non-empty base64 string
        assert isinstance(result, str)
        assert len(result) > 0
        # Should be valid base64
        import base64

        decoded = base64.b64decode(result)
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes


class TestFormatGpsTime:
    """Test GPS time formatting."""

    def test_format_basic(self):
        """Test basic GPS time formatting."""
        result = _format_gps_time(1234567890, 123456789)
        assert result == "1234567890.123456789"

    def test_format_zero_ns(self):
        """Test GPS time with zero nanoseconds."""
        result = _format_gps_time(1234567890, 0)
        assert result == "1234567890.000000000"

    def test_format_small_ns(self):
        """Test GPS time with small nanoseconds value."""
        result = _format_gps_time(1234567890, 1)
        assert result == "1234567890.000000001"


class TestPlotPsd:
    """Test PSD plotting function."""

    def test_plot_psd(self, mock_psd):
        """Test PSD plot generation."""
        result = _plot_psd(mock_psd, "H1")

        # Should return base64 string
        assert isinstance(result, str)
        assert len(result) > 0

        # Verify it's valid base64 PNG
        import base64

        decoded = base64.b64decode(result)
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


class TestPlotSnrTimeseries:
    """Test SNR time series plotting function."""

    def test_plot_snr(self, mock_snr_timeseries):
        """Test SNR time series plot generation."""
        result = _plot_snr_timeseries(mock_snr_timeseries, "H1", 15.0)

        # Should return base64 string
        assert isinstance(result, str)
        assert len(result) > 0

        # Verify it's valid base64 PNG
        import base64

        decoded = base64.b64decode(result)
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


class TestGenerateHtml:
    """Test HTML generation."""

    def test_generate_html_basic(self, mock_coinc_data):
        """Test basic HTML generation."""
        html = _generate_html(mock_coinc_data, "/path/to/test.xml")

        # Check HTML structure
        assert "<!DOCTYPE html>" in html
        assert "Gravitational Wave Event Report" in html
        assert "Network SNR" in html
        assert "Chirp Mass" in html

        # Check for detector badges
        assert "H1" in html
        assert "L1" in html

        # Check for embedded images (base64)
        assert "data:image/png;base64," in html

    def test_generate_html_without_snr_series(
        self, mock_coinc_inspiral, mock_sngl_inspiral, mock_psd
    ):
        """Test HTML generation without SNR time series."""
        data = {
            "coinc_inspiral": mock_coinc_inspiral,
            "triggers": [mock_sngl_inspiral],
            "psds": {"H1": mock_psd},
            "snr_series": {},  # Empty SNR series
        }

        html = _generate_html(data, "/path/to/test.xml")

        # Should still generate HTML
        assert "<!DOCTYPE html>" in html
        # Should not have SNR Time Series section
        assert "SNR Time Series" not in html or html.count("SNR Time Series") < 2


class TestParseCoincXml:
    """Test coinc XML parsing."""

    @patch("sgnligo.bin.coinc_to_html.ligolw_utils.load_filename")
    @patch("sgnligo.bin.coinc_to_html.lsctables.CoincInspiralTable.get_table")
    @patch("sgnligo.bin.coinc_to_html.lsctables.SnglInspiralTable.get_table")
    @patch("sgnligo.bin.coinc_to_html.lal.series.read_psd_xmldoc")
    def test_parse_coinc_xml(
        self,
        mock_read_psd,
        mock_sngl_table,
        mock_coinc_table,
        mock_load,
        mock_coinc_inspiral,
        mock_sngl_inspiral,
        mock_psd,
    ):
        """Test parsing a coinc XML file."""
        # Setup mocks
        mock_xmldoc = MagicMock()
        mock_load.return_value = mock_xmldoc
        mock_coinc_table.return_value = [mock_coinc_inspiral]
        mock_sngl_table.return_value = [mock_sngl_inspiral]
        mock_read_psd.return_value = {"H1": mock_psd}

        # Mock XML structure for SNR time series
        mock_child = MagicMock()
        mock_child.tagName = "LIGO_LW"
        mock_xmldoc.childNodes = [MagicMock(childNodes=[mock_child])]

        with patch(
            "sgnligo.bin.coinc_to_html.lal.series.parse_COMPLEX8TimeSeries"
        ) as mock_parse_ts:
            mock_ts = MagicMock()
            mock_ts.name = "snr_H1"
            mock_parse_ts.return_value = mock_ts

            result = _parse_coinc_xml("/path/to/test.xml")

        assert "coinc_inspiral" in result
        assert "triggers" in result
        assert "psds" in result
        assert "snr_series" in result


class TestParseCommandLine:
    """Test command line parsing."""

    def test_parse_basic(self):
        """Test basic command line parsing."""
        with patch.object(sys, "argv", ["prog", "input.xml"]):
            args = parse_command_line()
            assert args.input == "input.xml"
            assert args.output is None
            assert args.verbose is False

    def test_parse_with_output(self):
        """Test parsing with output file."""
        with patch.object(sys, "argv", ["prog", "input.xml", "output.html"]):
            args = parse_command_line()
            assert args.input == "input.xml"
            assert args.output == "output.html"

    def test_parse_with_output_flag(self):
        """Test parsing with -o flag."""
        with patch.object(sys, "argv", ["prog", "input.xml", "-o", "custom.html"]):
            args = parse_command_line()
            assert args.input == "input.xml"
            assert args.output_file == "custom.html"

    def test_parse_verbose(self):
        """Test parsing with verbose flag."""
        with patch.object(sys, "argv", ["prog", "input.xml", "-v"]):
            args = parse_command_line()
            assert args.verbose is True


class TestMain:
    """Test main entry point."""

    def test_main_file_not_found(self, capsys):
        """Test main with nonexistent input file."""
        with patch.object(sys, "argv", ["prog", "/nonexistent/file.xml"]):
            result = main()

        assert result == 1
        captured = capsys.readouterr()
        assert "Error: Input file not found" in captured.err

    @patch("sgnligo.bin.coinc_to_html._parse_coinc_xml")
    @patch("sgnligo.bin.coinc_to_html._generate_html")
    def test_main_success(self, mock_generate, mock_parse, mock_coinc_data, capsys):
        """Test successful main execution."""
        mock_parse.return_value = mock_coinc_data
        mock_generate.return_value = "<html></html>"

        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            input_path = f.name

        try:
            with patch.object(sys, "argv", ["prog", input_path]):
                result = main()

            assert result == 0
            captured = capsys.readouterr()
            assert "Report written to" in captured.out
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(input_path).with_suffix(".html").unlink(missing_ok=True)

    @patch("sgnligo.bin.coinc_to_html._parse_coinc_xml")
    @patch("sgnligo.bin.coinc_to_html._generate_html")
    def test_main_with_verbose(
        self, mock_generate, mock_parse, mock_coinc_data, capsys
    ):
        """Test main with verbose flag."""
        mock_parse.return_value = mock_coinc_data
        mock_generate.return_value = "<html></html>"

        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            input_path = f.name

        try:
            with patch.object(sys, "argv", ["prog", input_path, "-v"]):
                result = main()

            assert result == 0
            captured = capsys.readouterr()
            assert "Reading:" in captured.out
            assert "Generating HTML report" in captured.out
            assert "Writing:" in captured.out
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(input_path).with_suffix(".html").unlink(missing_ok=True)

    @patch("sgnligo.bin.coinc_to_html._parse_coinc_xml")
    @patch("sgnligo.bin.coinc_to_html._generate_html")
    def test_main_with_custom_output(
        self, mock_generate, mock_parse, mock_coinc_data, tmp_path
    ):
        """Test main with custom output path."""
        mock_parse.return_value = mock_coinc_data
        mock_generate.return_value = "<html></html>"

        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            input_path = f.name

        output_path = tmp_path / "custom_output.html"

        try:
            with patch.object(
                sys, "argv", ["prog", input_path, "-o", str(output_path)]
            ):
                result = main()

            assert result == 0
            assert output_path.exists()
        finally:
            Path(input_path).unlink(missing_ok=True)

    @patch("sgnligo.bin.coinc_to_html._parse_coinc_xml")
    @patch("sgnligo.bin.coinc_to_html._generate_html")
    def test_main_with_positional_output(
        self, mock_generate, mock_parse, mock_coinc_data, tmp_path
    ):
        """Test main with positional output argument."""
        mock_parse.return_value = mock_coinc_data
        mock_generate.return_value = "<html></html>"

        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            input_path = f.name

        output_path = tmp_path / "positional_output.html"

        try:
            with patch.object(sys, "argv", ["prog", input_path, str(output_path)]):
                result = main()

            assert result == 0
            assert output_path.exists()
        finally:
            Path(input_path).unlink(missing_ok=True)


class TestParseCoincXmlSnrSeries:
    """Test SNR series extraction edge cases."""

    @patch("sgnligo.bin.coinc_to_html.ligolw_utils.load_filename")
    @patch("sgnligo.bin.coinc_to_html.lsctables.CoincInspiralTable.get_table")
    @patch("sgnligo.bin.coinc_to_html.lsctables.SnglInspiralTable.get_table")
    @patch("sgnligo.bin.coinc_to_html.lal.series.read_psd_xmldoc")
    def test_parse_skips_invalid_snr_series(
        self,
        mock_read_psd,
        mock_sngl_table,
        mock_coinc_table,
        mock_load,
        mock_coinc_inspiral,
        mock_sngl_inspiral,
        mock_psd,
    ):
        """Test that invalid SNR time series are skipped gracefully."""
        mock_xmldoc = MagicMock()
        mock_load.return_value = mock_xmldoc
        mock_coinc_table.return_value = [mock_coinc_inspiral]
        mock_sngl_table.return_value = [mock_sngl_inspiral]
        mock_read_psd.return_value = {"H1": mock_psd}

        # Create a child that will raise exception during parsing
        mock_child = MagicMock()
        mock_child.tagName = "LIGO_LW"
        mock_xmldoc.childNodes = [MagicMock(childNodes=[mock_child])]

        with patch(
            "sgnligo.bin.coinc_to_html.lal.series.parse_COMPLEX8TimeSeries"
        ) as mock_parse_ts:
            # Raise exception to test error handling
            mock_parse_ts.side_effect = Exception("Parse error")

            result = _parse_coinc_xml("/path/to/test.xml")

        # Should complete without error, with empty snr_series
        assert result["snr_series"] == {}

    @patch("sgnligo.bin.coinc_to_html.ligolw_utils.load_filename")
    @patch("sgnligo.bin.coinc_to_html.lsctables.CoincInspiralTable.get_table")
    @patch("sgnligo.bin.coinc_to_html.lsctables.SnglInspiralTable.get_table")
    @patch("sgnligo.bin.coinc_to_html.lal.series.read_psd_xmldoc")
    def test_parse_skips_non_snr_timeseries(
        self,
        mock_read_psd,
        mock_sngl_table,
        mock_coinc_table,
        mock_load,
        mock_coinc_inspiral,
        mock_sngl_inspiral,
        mock_psd,
    ):
        """Test that non-SNR time series are skipped."""
        mock_xmldoc = MagicMock()
        mock_load.return_value = mock_xmldoc
        mock_coinc_table.return_value = [mock_coinc_inspiral]
        mock_sngl_table.return_value = [mock_sngl_inspiral]
        mock_read_psd.return_value = {"H1": mock_psd}

        mock_child = MagicMock()
        mock_child.tagName = "LIGO_LW"
        mock_xmldoc.childNodes = [MagicMock(childNodes=[mock_child])]

        with patch(
            "sgnligo.bin.coinc_to_html.lal.series.parse_COMPLEX8TimeSeries"
        ) as mock_parse_ts:
            mock_ts = MagicMock()
            mock_ts.name = "other_H1"  # Not starting with "snr_"
            mock_parse_ts.return_value = mock_ts

            result = _parse_coinc_xml("/path/to/test.xml")

        # Should not include non-snr time series
        assert result["snr_series"] == {}
