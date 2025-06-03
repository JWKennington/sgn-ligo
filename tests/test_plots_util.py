"""Test coverage for sgnligo.plots.util module."""

import math
import os
from unittest.mock import patch

import numpy as np
import pytest

from sgnligo.plots.util import (
    colour_from_instruments,
    golden_ratio,
    latexfilename,
    latexnumber,
    set_matplotlib_cache_directory,
)


class TestGoldenRatio:
    """Test the golden ratio constant."""

    def test_golden_ratio_value(self):
        """Test that golden_ratio has the correct value."""
        expected = (1.0 + math.sqrt(5.0)) / 2.0
        assert golden_ratio == expected
        assert abs(golden_ratio - 1.618033988749895) < 1e-10


class TestSetMatplotlibCacheDirectory:
    """Test cases for set_matplotlib_cache_directory function."""

    @patch.dict(os.environ, {}, clear=True)
    def test_no_condor_scratch_dir(self):
        """Test when _CONDOR_SCRATCH_DIR is not set."""
        # Should not modify MPLCONFIGDIR
        set_matplotlib_cache_directory()
        assert "MPLCONFIGDIR" not in os.environ

    @patch.dict(
        os.environ, {"_CONDOR_SCRATCH_DIR": "/tmp/condor"}, clear=True  # noqa: S108
    )
    def test_with_condor_scratch_dir(self):
        """Test when _CONDOR_SCRATCH_DIR is set."""
        set_matplotlib_cache_directory()
        assert os.environ["MPLCONFIGDIR"] == "/tmp/condor"  # noqa: S108

    @patch.dict(
        os.environ,
        {"_CONDOR_SCRATCH_DIR": "/scratch/job123", "MPLCONFIGDIR": "/old/path"},
        clear=True,
    )
    def test_overwrite_existing_mplconfigdir(self):
        """Test that existing MPLCONFIGDIR is overwritten."""
        assert os.environ["MPLCONFIGDIR"] == "/old/path"
        set_matplotlib_cache_directory()
        assert os.environ["MPLCONFIGDIR"] == "/scratch/job123"


class TestColourFromInstruments:
    """Test cases for colour_from_instruments function."""

    def test_single_instrument_default_colors(self):
        """Test getting color for single instruments with default colors."""
        # Test known instruments
        instruments = ["H1"]
        color = colour_from_instruments(instruments)
        # H1 is red (#ee0000)
        expected = np.array([238 / 255.0, 0, 0])
        np.testing.assert_array_almost_equal(color, expected)

        # Test L1
        color = colour_from_instruments(["L1"])
        # L1 is blue (#4ba6ff)
        expected = np.array([75 / 255.0, 166 / 255.0, 255 / 255.0])
        np.testing.assert_array_almost_equal(color, expected)

    def test_multiple_instruments_default_colors(self):
        """Test mixing colors for multiple instruments."""
        instruments = ["H1", "L1"]
        color = colour_from_instruments(instruments)

        # Colors should be mixed additively and normalized
        assert isinstance(color, np.ndarray)
        assert len(color) == 3
        assert color.max() == 1.0  # Should be normalized

    def test_custom_colors(self):
        """Test using custom color dictionary."""
        custom_colors = {
            "X1": np.array([1.0, 0.0, 0.0]),  # Red
            "Y1": np.array([0.0, 1.0, 0.0]),  # Green
        }
        color = colour_from_instruments(["X1"], colours=custom_colors)
        np.testing.assert_array_equal(color, np.array([1.0, 0.0, 0.0]))

    def test_multiple_instruments_custom_colors(self):
        """Test mixing custom colors for multiple instruments."""
        custom_colors = {
            "X1": np.array([1.0, 0.0, 0.0]),  # Red
            "Y1": np.array([0.0, 1.0, 0.0]),  # Green
            "Z1": np.array([0.0, 0.0, 1.0]),  # Blue
        }
        color = colour_from_instruments(["X1", "Y1"], colours=custom_colors)

        # Should be mixed and normalized
        assert color.max() == 1.0
        # Check that it's yellowish (mix of red and green)
        assert color[0] > 0 and color[1] > 0

    def test_three_instruments(self):
        """Test mixing three instrument colors."""
        instruments = ["H1", "L1", "V1"]
        color = colour_from_instruments(instruments)

        # Should be normalized
        assert color.max() == 1.0
        assert isinstance(color, np.ndarray)
        assert len(color) == 3

    def test_all_default_instruments(self):
        """Test all instruments defined in default colors."""
        default_instruments = ["G1", "H1", "L1", "V1", "K1", "E1", "E2", "E3"]

        for inst in default_instruments:
            color = colour_from_instruments([inst])
            assert isinstance(color, np.ndarray)
            assert len(color) == 3
            assert np.all(color >= 0) and np.all(color <= 1)

    def test_key_error_with_unknown_instrument(self):
        """Test that unknown instrument raises KeyError."""
        with pytest.raises(KeyError):
            colour_from_instruments(["UNKNOWN"])

    def test_desaturation_formula(self):
        """Test the desaturation formula for multiple instruments."""
        # Get individual colors
        h1_color = colour_from_instruments(["H1"])
        l1_color = colour_from_instruments(["L1"])

        # Get mixed color
        mixed_color = colour_from_instruments(["H1", "L1"])

        # The formula should be: (h1 + l1 + 1) / max(h1 + l1 + 1)
        expected_unnormalized = h1_color + l1_color + 1
        expected = expected_unnormalized / expected_unnormalized.max()

        np.testing.assert_array_almost_equal(mixed_color, expected)


class TestLatexNumber:
    """Test cases for latexnumber function."""

    def test_simple_number_unchanged(self):
        """Test that numbers without scientific notation are unchanged."""
        assert latexnumber("123") == "123"
        assert latexnumber("123.456") == "123.456"
        assert latexnumber("-42") == "-42"
        assert latexnumber("+3.14") == "+3.14"

    def test_scientific_notation_with_e(self):
        """Test conversion of scientific notation with lowercase e."""
        result = latexnumber("1.23e10")
        assert result == r"$1.23 \times 10^{10}$"

        result = latexnumber("5.67e-8")
        assert result == r"$5.67 \times 10^{-8}$"

    def test_scientific_notation_with_E(self):
        """Test conversion of scientific notation with uppercase E."""
        result = latexnumber("1.23E10")
        assert result == r"$1.23 \times 10^{10}$"

        result = latexnumber("5.67E-8")
        assert result == r"$5.67 \times 10^{-8}$"

    def test_with_signs(self):
        """Test handling of positive and negative signs."""
        result = latexnumber("-1.5e20")
        assert result == r"$-1.5 \times 10^{20}$"

        result = latexnumber("+2.5E-15")
        assert result == r"$+2.5 \times 10^{-15}$"

    def test_example_from_docstring(self):
        """Test the example given in the docstring."""
        result = latexnumber("%.12g" % (math.pi * 1e18))
        assert result == r"$3.14159265359 \times 10^{18}$"

    def test_zero_exponent(self):
        """Test handling of zero exponent."""
        result = latexnumber("1.0e0")
        assert result == r"$1.0 \times 10^{0}$"

    def test_integer_mantissa(self):
        """Test handling of integer mantissa."""
        result = latexnumber("1e10")
        assert result == r"$1 \times 10^{10}$"

    def test_invalid_format_raises_error(self):
        """Test that invalid scientific notation raises ValueError."""
        with pytest.raises(ValueError, match="floatpatterm match failed"):
            latexnumber("1.23e")  # Missing exponent

        with pytest.raises(ValueError, match="floatpatterm match failed"):
            latexnumber("e10")  # Missing mantissa

    def test_very_small_and_large_numbers(self):
        """Test extreme values."""
        result = latexnumber("9.999999e99")
        assert result == r"$9.999999 \times 10^{99}$"

        result = latexnumber("1.0e-99")
        assert result == r"$1.0 \times 10^{-99}$"


class TestLatexFilename:
    """Test cases for latexfilename function."""

    def test_escape_backslash(self):
        """Test escaping of backslash characters."""
        assert latexfilename("path\\to\\file") == "path\\\\to\\\\file"
        assert latexfilename("\\") == "\\\\"

    def test_escape_underscore(self):
        """Test escaping of underscore characters."""
        assert latexfilename("my_file_name") == "my\\_file\\_name"
        assert latexfilename("_") == "\\_"

    def test_replace_spaces(self):
        """Test replacement of spaces with non-breaking spaces."""
        assert latexfilename("my file name") == "my~file~name"
        assert latexfilename(" ") == "~"

    def test_combined_escaping(self):
        """Test escaping multiple special characters."""
        assert (
            latexfilename("path\\to\\my_file name.txt")
            == "path\\\\to\\\\my\\_file~name.txt"
        )

    def test_empty_string(self):
        """Test empty string handling."""
        assert latexfilename("") == ""

    def test_no_special_characters(self):
        """Test strings without special characters."""
        assert latexfilename("simple.txt") == "simple.txt"
        assert latexfilename("abc123") == "abc123"

    def test_multiple_consecutive_special_chars(self):
        """Test multiple consecutive special characters."""
        assert latexfilename("__test__") == "\\_\\_test\\_\\_"
        assert latexfilename("\\\\path") == "\\\\\\\\path"
        assert latexfilename("  spaces  ") == "~~spaces~~"

    def test_order_of_operations(self):
        """Test that replacements don't interfere with each other."""
        # Make sure backslash escaping happens before underscore
        # so we don't double-escape
        result = latexfilename("\\_")
        assert result == "\\\\\\_"  # \\ (escaped \) + \_ (escaped _)
