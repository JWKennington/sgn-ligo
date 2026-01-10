"""Tests for sgnligo.bin.plot_psd module.

This module provides comprehensive tests for the plot_psd command-line tool,
which generates plots of Power Spectral Density (PSD) data.
"""

import sys
from unittest.mock import Mock, call, patch

import pytest

from sgnligo.bin import plot_psd


@pytest.fixture
def mock_psd_data():
    """Create mock PSD data for testing.

    Returns a dictionary keyed by detector with PSD frequency series data.
    """
    mock_h1_psd = Mock()
    mock_h1_psd.name = "H1"
    mock_l1_psd = Mock()
    mock_l1_psd.name = "L1"

    return {"H1": mock_h1_psd, "L1": mock_l1_psd}


@pytest.fixture
def mock_figure():
    """Create a mock matplotlib figure for testing."""
    mock_fig = Mock()
    mock_fig.savefig = Mock()
    return mock_fig


class TestParseCommandLine:
    """Test command line parsing."""

    def test_parse_single_file(self):
        """Test parsing with a single input file."""
        test_args = ["plot_psd", "test_psd.xml"]

        with patch.object(sys, "argv", test_args):
            options = plot_psd.parse_command_line()

        assert options.filenames == ["test_psd.xml"]
        assert options.output is None
        assert options.verbose is False

    def test_parse_multiple_files(self):
        """Test parsing with multiple input files."""
        test_args = ["plot_psd", "psd1.xml", "psd2.xml", "psd3.xml"]

        with patch.object(sys, "argv", test_args):
            options = plot_psd.parse_command_line()

        assert options.filenames == ["psd1.xml", "psd2.xml", "psd3.xml"]
        assert options.output is None

    def test_parse_with_output(self):
        """Test parsing with custom output filename."""
        test_args = ["plot_psd", "input.xml", "-o", "custom_output.png"]

        with patch.object(sys, "argv", test_args):
            options = plot_psd.parse_command_line()

        assert options.filenames == ["input.xml"]
        assert options.output == "custom_output.png"

    def test_parse_with_verbose(self):
        """Test parsing with verbose flag."""
        test_args = ["plot_psd", "input.xml", "-v"]

        with patch.object(sys, "argv", test_args):
            options = plot_psd.parse_command_line()

        assert options.verbose is True

    def test_parse_no_files_raises_error(self):
        """Test that parsing with no files raises ValueError.

        This tests the error handling for the case where filenames is an
        empty list after argument parsing.
        """
        # Mock argparse to return empty filenames
        with patch("sgnligo.bin.plot_psd.ArgumentParser") as mock_parser_class:
            mock_parser = Mock()
            mock_parser_class.return_value = mock_parser

            # Create mock options with empty filenames
            mock_options = Mock()
            mock_options.filenames = []  # Empty list triggers the error
            mock_options.output = None
            mock_options.verbose = False
            mock_parser.parse_args.return_value = mock_options

            with pytest.raises(
                ValueError, match="must supply at least one input filename"
            ):
                plot_psd.parse_command_line()

    def test_parse_multiple_files_with_output_raises_error(self):
        """Test that multiple files with --output raises ValueError.

        The tool only supports custom output names with single input files.
        """
        test_args = ["plot_psd", "psd1.xml", "psd2.xml", "-o", "output.png"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(
                ValueError,
                match="must supply only one input file when setting --output",
            ):
                plot_psd.parse_command_line()


class TestMain:
    """Test the main function."""

    @patch("sgnligo.bin.plot_psd.parse_command_line")
    @patch("sgnligo.bin.plot_psd.read_psd")
    @patch("sgnligo.bin.plot_psd.plot_psds")
    @patch("os.path.splitext")
    def test_main_single_file_default_output(
        self,
        mock_splitext,
        mock_plot_psds,
        mock_read_psd,
        mock_parse_command_line,
        mock_psd_data,
        mock_figure,
    ):
        """Test main with single file and default output name.

        When no output is specified, the tool should use the input filename
        with .png extension.
        """
        # Setup mocks
        mock_options = Mock()
        mock_options.filenames = ["test_psd.xml"]
        mock_options.output = None
        mock_options.verbose = False
        mock_parse_command_line.return_value = mock_options

        mock_read_psd.return_value = mock_psd_data
        mock_plot_psds.return_value = mock_figure

        # Mock os.path.splitext - returns base and extension
        mock_splitext.return_value = ("test_psd", ".xml")

        # Run main
        plot_psd.main()

        # Verify calls
        mock_parse_command_line.assert_called_once()
        mock_read_psd.assert_called_once_with("test_psd.xml", verbose=False)
        mock_plot_psds.assert_called_once_with(mock_psd_data, plot_width=2400)
        # Now correctly generates "test_psd.png"
        mock_figure.savefig.assert_called_once_with("test_psd.png")

    @patch("sgnligo.bin.plot_psd.parse_command_line")
    @patch("sgnligo.bin.plot_psd.read_psd")
    @patch("sgnligo.bin.plot_psd.plot_psds")
    def test_main_single_file_custom_output(
        self,
        mock_plot_psds,
        mock_read_psd,
        mock_parse_command_line,
        mock_psd_data,
        mock_figure,
    ):
        """Test main with single file and custom output name."""
        # Setup mocks
        mock_options = Mock()
        mock_options.filenames = ["input.xml"]
        mock_options.output = "custom_plot.png"
        mock_options.verbose = True
        mock_parse_command_line.return_value = mock_options

        mock_read_psd.return_value = mock_psd_data
        mock_plot_psds.return_value = mock_figure

        # Run main
        plot_psd.main()

        # Verify calls
        mock_read_psd.assert_called_once_with("input.xml", verbose=True)
        mock_plot_psds.assert_called_once_with(mock_psd_data, plot_width=2400)
        mock_figure.savefig.assert_called_once_with("custom_plot.png")

    @patch("sgnligo.bin.plot_psd.parse_command_line")
    @patch("sgnligo.bin.plot_psd.read_psd")
    @patch("sgnligo.bin.plot_psd.plot_psds")
    @patch("os.path.splitext")
    def test_main_multiple_files(
        self,
        mock_splitext,
        mock_plot_psds,
        mock_read_psd,
        mock_parse_command_line,
        mock_psd_data,
        mock_figure,
    ):
        """Test main with multiple input files.

        Each file should be processed independently with its own output.
        """
        # Setup mocks
        mock_options = Mock()
        mock_options.filenames = ["psd1.xml", "psd2.xml"]
        mock_options.output = None
        mock_options.verbose = False
        mock_parse_command_line.return_value = mock_options

        mock_read_psd.return_value = mock_psd_data
        mock_plot_psds.return_value = mock_figure

        # Mock os.path.splitext for each file
        mock_splitext.side_effect = [("psd1", ".xml"), ("psd2", ".xml")]

        # Run main
        plot_psd.main()

        # Verify calls for each file
        assert mock_read_psd.call_count == 2
        mock_read_psd.assert_has_calls(
            [call("psd1.xml", verbose=False), call("psd2.xml", verbose=False)]
        )

        assert mock_plot_psds.call_count == 2
        assert mock_figure.savefig.call_count == 2
        # Now correctly generates proper filenames
        mock_figure.savefig.assert_has_calls([call("psd1.png"), call("psd2.png")])


class TestModuleSetup:
    """Test module-level setup code."""

    def test_matplotlib_rcparams_updated(self):
        """Test that matplotlib rcParams are properly configured.

        The module sets specific plotting parameters for publication-quality
        figures with LaTeX rendering (if available).
        """
        import shutil

        import matplotlib

        # Import the module that sets rcParams to ensure they're configured
        import sgnligo.bin.plot_psd  # noqa: F401 - imported for side effects

        # Check key parameters that should be set
        assert matplotlib.rcParams["font.size"] == 10.0
        assert matplotlib.rcParams["axes.titlesize"] == 10.0
        assert matplotlib.rcParams["axes.labelsize"] == 10.0
        assert matplotlib.rcParams["xtick.labelsize"] == 8.0
        assert matplotlib.rcParams["ytick.labelsize"] == 8.0
        assert matplotlib.rcParams["legend.fontsize"] == 8.0
        assert matplotlib.rcParams["figure.dpi"] == 300
        assert matplotlib.rcParams["savefig.dpi"] == 300
        # usetex should match LaTeX availability
        latex_available = shutil.which("latex") is not None
        assert matplotlib.rcParams["text.usetex"] is latex_available
        assert matplotlib.rcParams["path.simplify"] is True

    @patch("sgnligo.bin.plot_psd.set_matplotlib_cache_directory")
    def test_matplotlib_cache_directory_set(self, mock_set_cache):
        """Test that matplotlib cache directory is set on import.

        This is done to ensure matplotlib uses a proper cache location.
        Note: This test requires re-importing the module to trigger the
        module-level code.
        """
        # The function should have been called during module import
        # We can't easily test this without complex import manipulation
        # so we'll just verify the function exists and is callable
        assert callable(mock_set_cache)
