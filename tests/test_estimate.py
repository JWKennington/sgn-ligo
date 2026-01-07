"""Unit and integration tests for the high-level estimate_psd utility."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from igwn_ligolw import utils as ligolw_utils
from lal import series

from sgnligo.estimate import estimate_psd, infer_source_sample_rate
from sgnligo.sources.gwdata_noise_source import GWDataNoiseSource

# Module-level switch for enabling expensive or visual plotting tests
# Set this to True (or use an env var) when manually verifying phenomenology
RUN_PLOT_TESTS = os.getenv("SGNLIGO_RUN_PLOTS", "false").lower() == "true"


class TestEstimatePSDPadValidation:
    """Tests for input validation logic."""

    def test_missing_ifo_pad_raises_error(self, tmp_path):
        """Verify ValueError if source lacks output pad for requested IFO."""
        source = GWDataNoiseSource(
            name="noise_h1", channel_dict={"H1": "H1"}, t0=0, end=1
        )
        with pytest.raises(ValueError, match="does not have an output pad named 'L1'"):
            estimate_psd(
                start=0,
                end=1,
                ifos=["L1"],
                sample_rate=4096,
                fname=str(tmp_path / "fail.xml"),
                source_element=source,
            )


class TestInferenceLogic:
    """Tests for the helper function infer_source_sample_rate."""

    def test_infer_gwdata_noise_source(self):
        """Verify it detects 16384 Hz from GWDataNoiseSource."""
        src = GWDataNoiseSource(
            name="noise", channel_dict={"H1": "H1", "L1": "L1"}, t0=0, end=1
        )
        assert infer_source_sample_rate(src, "H1") == 16384.0
        assert infer_source_sample_rate(src, "L1") == 16384.0

    def test_infer_generic_attribute(self):
        """Verify it detects .sample_rate attribute."""
        mock_src = MagicMock()
        mock_src.sample_rate = 2048
        assert infer_source_sample_rate(mock_src, "H1") == 2048.0

    def test_infer_fallback(self):
        """Verify fallback to 4096 Hz."""
        mock_src = MagicMock()
        del mock_src.sample_rate  # Ensure no attribute
        if hasattr(mock_src, "channel_info"):
            del mock_src.channel_info

        assert infer_source_sample_rate(mock_src, "H1") == 4096.0


class TestPipelineConstruction:
    """White-box tests verifying internal pipeline structure using mocks."""

    @patch("sgnligo.estimate.Pipeline")
    @patch("sgnligo.estimate.GWOSCSource")
    @patch("sgnligo.estimate.Whiten")
    @patch("sgnligo.estimate.Resampler")
    @patch("sgnligo.estimate.PSDSink")
    @patch("sgnligo.estimate.NullSeriesSink")
    def test_pipeline_structure_defaults(
        self,
        MockNull,
        MockPSD,
        MockResampler,
        MockWhiten,
        MockGWOSC,
        MockPipeline,
        tmp_path,
    ):
        """Test default pipeline construction (GWOSC 4k -> 4k, no resampler)."""
        # Configure the Mock Source to have the expected 'srcs' dictionary
        mock_source = MockGWOSC.return_value
        # The pipeline checks 'if "H1" not in source.srcs'
        mock_source.srcs = {"H1": MagicMock()}
        # Configure infer_source_sample_rate to see 4096
        mock_source.sample_rate = 4096

        estimate_psd(
            start=100,
            end=200,
            ifos=["H1"],
            sample_rate=4096,
            fname=str(tmp_path / "out.xml"),
        )

        MockGWOSC.assert_called_once()
        MockWhiten.assert_called_once()
        MockResampler.assert_not_called()

    @patch("sgnligo.estimate.Pipeline")
    @patch("sgnligo.estimate.Whiten")
    @patch("sgnligo.estimate.Resampler")
    def test_pipeline_structure_resampling(
        self, MockResampler, MockWhiten, MockPipeline, tmp_path
    ):
        """Test pipeline with resampling (16k -> 4k) via explicit override."""
        mock_source = MagicMock()
        mock_source.srcs = {"H1": MagicMock()}

        estimate_psd(
            start=0,
            end=10,
            ifos=["H1"],
            sample_rate=4096,
            source_sample_rate=16384,
            fname=str(tmp_path / "out.xml"),
            source_element=mock_source,
        )

        MockResampler.assert_called_once()
        _, kwargs = MockResampler.call_args
        assert kwargs["inrate"] == 16384
        assert kwargs["outrate"] == 4096


class TestEstimatePSDIntegration:
    """Integration tests using local noise generation (GWDataNoiseSource)."""

    def test_single_ifo_resampling_inferred(self, tmp_path):
        """
        Test: Single IFO, high-rate source -> low-rate analysis.
        Rate (16384) should be INFERRED from GWDataNoiseSource.
        """
        start = 1000000000
        duration = 4
        output = tmp_path / "psd_resampled.xml"

        source = GWDataNoiseSource(
            name="noise", channel_dict={"H1": "H1"}, t0=start, end=start + duration
        )

        # Analyze @ 2048 Hz. Source is 16384 Hz. Resampler should be auto-created.
        estimate_psd(
            start=start,
            end=start + duration,
            ifos=["H1"],
            sample_rate=2048,
            fname=str(output),
            fft_length=2.0,
            source_element=source,
            verbose=True,
        )

        assert output.exists()
        psd_dict = series.read_psd_xmldoc(
            ligolw_utils.load_filename(
                str(output), verbose=False, contenthandler=series.PSDContentHandler
            )
        )
        assert psd_dict["H1"].data.length > 0

        # Check Nyquist is consistent with 2048 Hz
        psd = psd_dict["H1"]
        max_f = psd.f0 + psd.deltaF * (psd.data.length - 1)
        assert 1000 <= max_f <= 1025

    def test_multi_ifo_npz_output(self, tmp_path):
        """Test: Multiple IFOs processing and NPZ output format."""
        start = 1000000000
        duration = 4
        output = tmp_path / "multi.npz"

        source = GWDataNoiseSource(
            name="noise_multi",
            channel_dict={"H1": "H1", "L1": "L1"},
            t0=start,
            end=start + duration,
        )

        estimate_psd(
            start=start,
            end=start + duration,
            ifos=["H1", "L1"],
            sample_rate=4096,
            fname=str(output),
            source_element=source,
        )

        assert output.exists()
        with np.load(output) as data:
            assert "H1_psd" in data
            assert "L1_psd" in data

    def test_custom_fft_length(self, tmp_path):
        """Test: Custom FFT length affects PSD resolution."""
        start = 1000000000
        duration = 8
        output = tmp_path / "psd_fine.xml"
        fft_len = 4.0

        source = GWDataNoiseSource(
            name="noise", channel_dict={"H1": "H1"}, t0=start, end=start + duration
        )

        estimate_psd(
            start=start,
            end=start + duration,
            ifos=["H1"],
            sample_rate=4096,
            fname=str(output),
            fft_length=fft_len,
            source_element=source,
        )

        psd_dict = series.read_psd_xmldoc(
            ligolw_utils.load_filename(
                str(output), verbose=False, contenthandler=series.PSDContentHandler
            )
        )
        assert np.isclose(psd_dict["H1"].deltaF, 1.0 / fft_len, rtol=0.1)


class TestEstimatePSDLive:
    """Tests utilizing the default GWOSC behavior."""

    def test_gwosc_default_pipeline(self, tmp_path):
        """
        Verify `estimate_psd` works with defaults.
        GWOSCSource rate (4096) should be used/inferred.

        No special skip logic: if this fails, we want to know about it.
        """
        start = 1126259460
        end = 1126259464
        output = tmp_path / "gw150914_default.xml"

        estimate_psd(
            start=start,
            end=end,
            ifos=["H1"],
            sample_rate=4096,
            fname=str(output),
            fft_length=2.0,
            verbose=True,
        )

        assert output.exists()
        psd_dict = series.read_psd_xmldoc(
            ligolw_utils.load_filename(
                str(output), verbose=False, contenthandler=series.PSDContentHandler
            )
        )
        assert "H1" in psd_dict
        assert psd_dict["H1"].data.length > 0

    @pytest.mark.skipif(
        not RUN_PLOT_TESTS,
        reason="Plotting tests disabled (set SGNLIGO_RUN_PLOTS=True to enable)",
    )
    def test_gwosc_plot_psd_phenomenology(self):
        """
        Generates a visual plot of the PSD for manual inspection.
        Useful for verifying the spectral shape (lines, 1/f noise) looks correct.
        """
        # A nice long stretch to get a clean average
        start = 1126259462
        duration = 32
        output_xml = Path("gw150914_plot.xml")
        output_png = Path("gw150914_psd.png")

        estimate_psd(
            start=start,
            end=start + duration,
            ifos=["H1", "L1"],
            sample_rate=4096,
            fname=str(output_xml),
            fft_length=4.0,
            verbose=True,
        )

        # Load Data
        psd_dict = series.read_psd_xmldoc(
            ligolw_utils.load_filename(
                str(output_xml), verbose=False, contenthandler=series.PSDContentHandler
            )
        )

        # Plotting Logic
        plt.figure(figsize=(10, 6))

        for ifo, psd in psd_dict.items():
            # Construct frequency array: f0 + i*deltaF
            freqs = psd.f0 + np.arange(psd.data.length) * psd.deltaF
            data = psd.data.data

            # Mask DC/Zero components for log plotting
            mask = (freqs > 10) & (data > 0)

            plt.loglog(freqs[mask], data[mask], label=f"{ifo} PSD", alpha=0.8)

        plt.title(f"PSD Estimate: GW150914 (T0={start}, Dur={duration}s)")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel(r"PSD [strain$^2$/Hz]")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.4)
        plt.xlim(10, 2048)

        # Save plot
        plt.savefig(output_png)
        plt.close()

        print(f"\nGenerated PSD plot at: {output_png}")
        assert output_png.exists()
