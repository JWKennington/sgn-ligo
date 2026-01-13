#!/usr/bin/env python3

import lal
import lal.series
import numpy as np
import pytest
from igwn_ligolw import ligolw
from igwn_ligolw import utils as ligolw_utils
from sgn import CollectSink, NullSink, Pipeline
from sgnts.base import EventFrame, TSFrame
from sgnts.base.slice_tools import TIME_MAX

from sgnligo.sources import GWDataNoiseSource
from sgnligo.transforms.whiten import (
    DriftCorrectionKernel,
    Kernel,
    Whiten,
    WhiteningKernel,
    correction_kernel_from_psds,
    kernel_from_psd,
)


class TestWhitenInit:
    """Test Whiten initialization."""

    def test_init_with_zero_padding(self, tmp_path):
        """Test initialization creates tukey window when z_whiten > 0."""
        whiten = Whiten(
            name="test_whiten",
            sink_pad_names=("in",),
            instrument="H1",
            psd_pad_name="spectrum",
            whiten_pad_name="hoft",
            input_sample_rate=16384,
            whiten_sample_rate=2048,
            fft_length=8,  # z_whiten = fft_length/4 * sample_rate = 4096 > 0
        )
        assert whiten.tukey is not None
        assert whiten.z_whiten > 0

    def test_init_z_whiten_always_positive(self):
        """Verify z_whiten > 0 with valid parameters.

        Due to constraints in the Whiten class (stride must map to integer
        samples at both input and whiten rates), z_whiten is always > 0
        with valid parameters. An assertion in __post_init__ enforces this.
        """
        whiten = Whiten(
            name="test_whiten",
            sink_pad_names=("in",),
            instrument="H1",
            psd_pad_name="spectrum",
            whiten_pad_name="hoft",
            input_sample_rate=16384,
            whiten_sample_rate=2048,
            fft_length=8,
        )

        # With valid parameters, z_whiten is always > 0 and tukey is set
        assert whiten.z_whiten > 0
        assert whiten.tukey is not None


class TestTukeyWindow:
    """Test tukey_window method."""

    def test_tukey_window_valid_beta(self):
        """Test tukey_window with valid beta values."""
        whiten = Whiten(
            name="test_whiten",
            sink_pad_names=("in",),
            instrument="H1",
            psd_pad_name="spectrum",
            whiten_pad_name="hoft",
        )
        # Test with beta = 0.5
        result = whiten.tukey_window(100, 0.5)
        assert len(result) == 100
        # Middle should be 1.0
        assert result[50] == 1.0

    def test_tukey_window_invalid_beta_negative(self):
        """Test tukey_window raises ValueError for beta < 0 (line 251)."""
        whiten = Whiten(
            name="test_whiten",
            sink_pad_names=("in",),
            instrument="H1",
            psd_pad_name="spectrum",
            whiten_pad_name="hoft",
        )
        with pytest.raises(ValueError, match="Invalid value for beta"):
            whiten.tukey_window(100, -0.1)

    def test_tukey_window_invalid_beta_greater_than_one(self):
        """Test tukey_window raises ValueError for beta > 1 (line 251)."""
        whiten = Whiten(
            name="test_whiten",
            sink_pad_names=("in",),
            instrument="H1",
            psd_pad_name="spectrum",
            whiten_pad_name="hoft",
        )
        with pytest.raises(ValueError, match="Invalid value for beta"):
            whiten.tukey_window(100, 1.5)


class TestPSDMethods:
    """Test PSD-related methods."""

    def test_add_psd_first_call(self):
        """Test add_psd on first call (n_samples=0) (lines 313-314)."""
        import numpy as np

        whiten = Whiten(
            name="test_whiten",
            sink_pad_names=("in",),
            instrument="H1",
            psd_pad_name="spectrum",
            whiten_pad_name="hoft",
        )
        # n_samples starts at 0
        assert whiten.n_samples == 0

        # Create some fake frequency data
        fdata = np.ones(100) + 1j * np.zeros(100)

        # Call add_psd - this should hit lines 312-314
        whiten.add_psd(fdata)

        # After first call, n_samples should be 1
        assert whiten.n_samples == 1
        assert hasattr(whiten, "geometric_mean_square")

    def test_add_psd_subsequent_calls(self):
        """Test add_psd on subsequent calls (lines 315-331)."""
        import numpy as np

        whiten = Whiten(
            name="test_whiten",
            sink_pad_names=("in",),
            instrument="H1",
            psd_pad_name="spectrum",
            whiten_pad_name="hoft",
        )

        # Create some fake frequency data
        fdata = np.ones(100) + 1j * np.zeros(100)

        # First call
        whiten.add_psd(fdata)
        assert whiten.n_samples == 1

        # Second call - this should hit lines 315-331
        whiten.add_psd(fdata * 2)
        assert whiten.n_samples == 2

    def test_get_psd_first_call(self):
        """Test get_psd when n_samples=0 (lines 340-346)."""
        import numpy as np

        whiten = Whiten(
            name="test_whiten",
            sink_pad_names=("in",),
            instrument="H1",
            psd_pad_name="spectrum",
            whiten_pad_name="hoft",
        )
        # n_samples should be 0
        assert whiten.n_samples == 0

        # Create fake frequency data
        fdata = np.ones(whiten.n_whiten // 2 + 1) * 2.0

        # Call get_psd - this should hit lines 339-346
        psd = whiten.get_psd(fdata)

        # DC and Nyquist should be zero
        assert psd[0] == 0
        assert psd[whiten.n_whiten // 2] == 0

    def test_get_psd_after_samples(self):
        """Test get_psd after add_psd has been called (lines 347-351)."""
        import numpy as np

        whiten = Whiten(
            name="test_whiten",
            sink_pad_names=("in",),
            instrument="H1",
            psd_pad_name="spectrum",
            whiten_pad_name="hoft",
        )

        # Create fake frequency data and add to history
        fdata = np.ones(whiten.n_whiten // 2 + 1) * 2.0
        whiten.add_psd(fdata)

        # Now get_psd should return based on geometric mean
        psd = whiten.get_psd(fdata)
        assert psd is not None
        assert len(psd) == len(fdata)


class TestWhitenInternal:
    """Test Whiten internal method."""

    def test_internal_with_highpass_filter(self):
        """Test internal() with highpass_filter=True (lines 490-493)."""
        import pathlib

        from sgn import NullSink
        from sgn.apps import Pipeline
        from sgnts.sources import FakeSeriesSource

        PATH_DATA = pathlib.Path(__file__).parent / "data"
        PATH_PSD = PATH_DATA / "H1L1-GSTLAL-MEDIAN.xml.gz"

        pipeline = Pipeline()

        pipeline.insert(
            FakeSeriesSource(
                name="src",
                source_pad_names=("H1",),
                rate=16384,
                signal_type="white",
                end=10,
            ),
            Whiten(
                name="Whitener",
                sink_pad_names=("H1",),
                instrument="H1",
                psd_pad_name="spectrum",
                whiten_pad_name="hoft",
                input_sample_rate=16384,
                whiten_sample_rate=2048,
                fft_length=4,
                reference_psd=PATH_PSD.as_posix(),
                highpass_filter=True,  # Enable highpass filter
            ),
            NullSink(name="snk", sink_pad_names=("hoft", "spectrum")),
            link_map={
                "Whitener:snk:H1": "src:src:H1",
                "snk:snk:hoft": "Whitener:src:hoft",
                "snk:snk:spectrum": "Whitener:src:spectrum",
            },
        )

        pipeline.run()

    def test_internal_gap_drain_output_history(self):
        """Test internal() draining output history on gap (lines 479-480).

        This tests the code path where we have a gap frame but have previous
        data that needs to be drained. We use SegmentSource + Gate to create
        a controlled gap in the middle of the data stream after Whiten has
        processed some data and populated prev_data.

        The gap drain code path is hit when:
        1. frame.is_gap is True
        2. outoffset_info["noffset"] != 0
        3. self.prev_data is not None
        4. self.prev_data.shape[-1] > 0
        """
        import pathlib

        from sgn import NullSink
        from sgn.apps import Pipeline
        from sgnts.sources import FakeSeriesSource, SegmentSource
        from sgnts.transforms import Gate

        PATH_DATA = pathlib.Path(__file__).parent / "data"
        PATH_PSD = PATH_DATA / "H1L1-GSTLAL-MEDIAN.xml.gz"

        pipeline = Pipeline()

        # Create a data source that runs for 20 seconds
        # and a segment source that creates a gap from 10-12 seconds
        # This ensures Whiten processes data first (populating prev_data)
        # then hits a gap (triggering the drain code)

        pipeline.insert(
            FakeSeriesSource(
                name="datasrc",
                source_pad_names=("H1",),
                rate=16384,
                signal_type="white",
                t0=0,
                end=20,
            ),
            # SegmentSource defines where data should pass through
            # Segments are in nanoseconds: (start_ns, end_ns)
            # Gap will be from 10s to 12s (where there's no segment)
            SegmentSource(
                name="segsrc",
                source_pad_names=("seg",),
                rate=16384,
                t0=0,
                end=20,
                # Data passes through from 0-10s and 12-20s
                # Gap from 10-12s
                segments=(
                    (0, int(10 * 1e9)),  # 0 to 10 seconds
                    (int(12 * 1e9), int(20 * 1e9)),  # 12 to 20 seconds
                ),
            ),
            # Gate uses the segment source to control when data passes
            Gate(
                name="gate",
                source_pad_names=("H1",),
                sink_pad_names=("data", "control"),
                control="control",
            ),
            Whiten(
                name="Whitener",
                sink_pad_names=("H1",),
                instrument="H1",
                psd_pad_name="spectrum",
                whiten_pad_name="hoft",
                input_sample_rate=16384,
                whiten_sample_rate=2048,
                fft_length=4,
                reference_psd=PATH_PSD.as_posix(),
            ),
            NullSink(name="snk", sink_pad_names=("hoft", "spectrum")),
            link_map={
                "gate:snk:data": "datasrc:src:H1",
                "gate:snk:control": "segsrc:src:seg",
                "Whitener:snk:H1": "gate:src:H1",
                "snk:snk:hoft": "Whitener:src:hoft",
                "snk:snk:spectrum": "Whitener:src:spectrum",
            },
        )

        # Run the pipeline - this should:
        # 1. Process data from 0-10s (populating prev_data in Whiten)
        # 2. Hit a gap from 10-12s (triggering gap drain code lines 479-480)
        # 3. Resume processing from 12-20s
        pipeline.run()


class TestYFunction:
    """Test the Y helper function."""

    def test_y_function_length_one(self):
        """Test Y function with length=1."""
        from sgnligo.transforms.whiten import Y

        assert Y(1, 0) == 0

    def test_y_function_odd_length(self):
        """Test Y function with odd length."""
        from sgnligo.transforms.whiten import Y

        # length=5, middle index is 2
        assert Y(5, 0) == -1.0
        assert Y(5, 2) == 0.0
        assert Y(5, 4) == 1.0

    def test_y_function_even_length(self):
        """Test Y function with even length."""
        from sgnligo.transforms.whiten import Y

        # length=6
        assert Y(6, 0) == -1.0
        assert Y(6, 5) == 1.0


class TestInterpolatePSD:
    """Test the interpolate_psd function."""

    def test_interpolate_psd_no_op(self):
        """Test interpolate_psd when deltaF matches."""
        import lal

        from sgnligo.transforms.whiten import interpolate_psd

        psd = lal.CreateREAL8FrequencySeries(
            "test", lal.LIGOTimeGPS(0), 0.0, 1.0, lal.StrainUnit, 100
        )
        psd.data.data[:] = 1.0

        result = interpolate_psd(psd, 1.0)
        # Should return same object when deltaF matches
        assert result is psd


def _make_psd(length=1025, deltaF=1.0, val=1.0):
    """Helper to create a flat LAL REAL8FrequencySeries."""
    series = lal.CreateREAL8FrequencySeries(
        "test_psd", lal.LIGOTimeGPS(0), 0.0, deltaF, lal.StrainUnit, length
    )
    series.data.data[:] = val
    return series


class TestKernelFactories:
    """Test the kernel generation factory functions."""

    def test_kernel_from_psd_zero_latency(self):
        """Test generating a Minimum Phase kernel."""
        psd = _make_psd(val=1.0)
        k = kernel_from_psd(psd, zero_latency=True)

        assert isinstance(k, Kernel)
        assert k.latency == 0
        assert len(k.fir_matrix) > 0
        # Energy check
        assert np.isclose(np.sum(k.fir_matrix**2), 1.0, atol=1e-3)

    def test_kernel_from_psd_linear_phase(self):
        """Test generating a Linear Phase kernel."""
        psd = _make_psd(val=1.0)
        k = kernel_from_psd(psd, zero_latency=False)

        latency = k.latency
        taps = k.fir_matrix
        peak_idx = np.argmax(np.abs(taps))

        # Check peak alignment
        assert abs(peak_idx - latency) <= 1
        # Energy check
        assert np.isclose(np.sum(taps**2), 1.0, atol=1e-3)

    def test_correction_kernel_identity(self):
        psd = _make_psd(val=1.0)
        L = 128
        k = correction_kernel_from_psds(psd_live=psd, psd_ref=psd, truncation_samples=L)

        assert k.latency == L - 1
        assert len(k.fir_matrix) == L
        assert np.isclose(k.fir_matrix[-1], 1.0, atol=1e-3)

    def test_correction_kernel_gain_logic(self):
        # Ref=1, Live=4 -> Gain=2.0
        psd_ref = _make_psd(val=1.0)
        psd_live = _make_psd(val=4.0)
        L = 128
        k = correction_kernel_from_psds(
            psd_live=psd_live, psd_ref=psd_ref, truncation_samples=L
        )
        assert np.isclose(k.fir_matrix[-1], 2.0, atol=1e-2)

        # Ref=1, Live=0.25 -> Gain=0.5
        psd_live_small = _make_psd(val=0.25)
        k_small = correction_kernel_from_psds(
            psd_live=psd_live_small, psd_ref=psd_ref, truncation_samples=L
        )
        assert np.isclose(k_small.fir_matrix[-1], 0.5, atol=1e-2)


class TestWhiteningKernelElement:
    """Test the WhiteningKernel TSTransform element."""

    def test_init(self):
        elem = WhiteningKernel(
            name="wk",
            sink_pad_names=("spec",),
            zero_latency=True,
        )
        assert isinstance(elem, WhiteningKernel)

    def test_process_generates_event(self):
        elem = WhiteningKernel(
            name="wk",
            sink_pad_names=("spec",),
            zero_latency=True,
            verbose=True,
        )
        elem.configure()

        psd = _make_psd()
        input_frame = TSFrame(
            buffers=[],
            metadata={"psd": psd, "epoch": 123456789},
            EOS=False,
        )

        # Output frame must span the full range to accept buffers up to TIME_MAX
        output_frame = EventFrame(data=[], offset=0, noffset=int(TIME_MAX), EOS=False)

        # Bypass decorator to test logic directly
        elem.process.__wrapped__(elem, input_frame, output_frame)

        # Check .data, which holds the list of buffers
        assert len(output_frame.data) == 1
        taps = output_frame.data[0].data[0][0]
        assert len(taps) > 0

    def test_throttling(self):
        elem = WhiteningKernel(
            name="wk",
            sink_pad_names=("spec",),
            min_update_interval=1_000_000_000,
            verbose=True,
        )
        elem.configure()

        psd = _make_psd()
        in1 = TSFrame(buffers=[], metadata={"psd": psd, "epoch": 100})
        # Initialize with full span
        out1 = EventFrame(data=[], offset=0, noffset=int(TIME_MAX))

        elem.process.__wrapped__(elem, in1, out1)
        assert len(out1.data) == 1

        in2 = TSFrame(buffers=[], metadata={"psd": psd, "epoch": 101})
        out2 = EventFrame(data=[], offset=0, noffset=int(TIME_MAX))

        elem.process.__wrapped__(elem, in2, out2)
        assert len(out2.data) == 0


class TestDriftCorrectionKernelElement:
    """Test the DriftCorrectionKernel TSTransform element."""

    def test_missing_reference_psd(self):
        elem = DriftCorrectionKernel(
            name="dk",
            sink_pad_names=("spec",),
            reference_psd=None,
        )
        elem.configure()

        in1 = TSFrame(buffers=[], metadata={"psd": _make_psd(), "epoch": 100})
        out1 = EventFrame(data=[], offset=0, noffset=int(TIME_MAX))

        elem.process.__wrapped__(elem, in1, out1)
        assert len(out1.data) == 0

    def test_correction_generation(self):
        ref_psd = _make_psd(val=1.0)
        live_psd = _make_psd(val=4.0)

        elem = DriftCorrectionKernel(
            name="dk",
            sink_pad_names=("spec",),
            reference_psd=ref_psd,
            truncation_samples=64,
            verbose=True,
        )
        elem.configure()

        input_frame = TSFrame(
            buffers=[],
            metadata={"psd": live_psd, "epoch": 1000},
        )
        output_frame = EventFrame(data=[], offset=0, noffset=int(TIME_MAX))

        elem.process.__wrapped__(elem, input_frame, output_frame)

        assert len(output_frame.data) == 1
        taps = output_frame.data[0].data[0][0]
        assert len(taps) == 64
        assert np.isclose(taps[-1], 2.0, atol=1e-2)


@pytest.fixture
def mock_psd_file(tmp_path):
    """Create a dummy PSD XML file for testing."""
    # Create a flat PSD
    rate = 2048
    deltaF = 0.25
    length = int(rate / 2 / deltaF) + 1

    psd = lal.CreateREAL8FrequencySeries(
        "H1_PSD",
        lal.LIGOTimeGPS(0),
        0.0,
        deltaF,
        lal.StrainUnit**2 / lal.HertzUnit,
        length,
    )
    # Set to 1.0 (white noise level for unit variance)
    # For GWDataNoiseSource, this defines the color of the noise.
    psd.data.data[:] = 1.0

    # Write to XML
    xmldoc = ligolw.Document()
    xmldoc.appendChild(ligolw.LIGO_LW())

    # Create PSD element (using LAL's writing utilities is cleanest if available,
    # but manually wrapping is safer for basic tests if dependencies vary)
    # simpler: use lal.series.make_psd_xmldoc

    psd_dict = {"H1": psd}
    lal.series.make_psd_xmldoc(psd_dict, xmldoc)

    fname = tmp_path / "test_psd.xml.gz"
    ligolw_utils.write_filename(xmldoc, str(fname))

    return str(fname)


class TestPipelineWhiteningKernel:
    """Integration tests for WhiteningKernel using GWDataNoiseSource."""

    def test_pipeline_generation(self):
        """Verify WhiteningKernel produces kernels from a live source."""
        pipeline = Pipeline()

        # 1. Source: GWDataNoiseSource
        # Note: We use channel_dict to map internal channels to pad names.
        src = GWDataNoiseSource(
            name="source",
            channel_dict={"H1": "H1:STRAIN"},
            duration=10.0,
        )

        # Extract rate from source metadata (populated in __post_init__)
        src_info = src.channel_info["H1"]
        rate = src_info["rate"]

        # 2. Whiten: Measure PSD
        whiten = Whiten(
            name="whiten",
            sink_pad_names=("H1",),
            psd_pad_name="spectrum",
            whiten_pad_name="hoft",
            input_sample_rate=rate,
            whiten_sample_rate=2048,
            fft_length=4,
            instrument="H1",
        )

        # 3. Kernel Generator
        kernel_gen = WhiteningKernel(
            name="kernel_gen",
            sink_pad_names=("spectrum",),
            filters_pad_name="filters",
            zero_latency=True,
            verbose=True,
        )

        # 4. Sinks
        # Standard CollectSink captures frames in .collects dict
        kern_sink = CollectSink(
            name="kern_sink",
            sink_pad_names=("filters",),
        )
        # Null sink to consume the 'hoft' output from Whiten
        null_sink = NullSink(
            name="null_sink",
            sink_pad_names=("hoft",),
        )

        pipeline.insert(
            src,
            whiten,
            kernel_gen,
            kern_sink,
            null_sink,
            link_map={
                whiten.snks["H1"]: src.srcs["H1:STRAIN"],
                kernel_gen.snks["spectrum"]: whiten.srcs["spectrum"],
                kern_sink.snks["filters"]: kernel_gen.srcs["filters"],
                null_sink.snks["hoft"]: whiten.srcs["hoft"],
            },
        )

        pipeline.run()

        # Verification
        frames = kern_sink.collects["filters"]
        assert len(frames) > 0, "Sink should have collected frames"

        # Find first valid kernel buffer
        # Frames may be empty (gaps or skipped updates), so we must search for content
        found_taps = None
        for f in frames:
            # Each f is not a frame but a list since EventFrame.data holds iterables
            if len(f) == 0:
                continue

            frame = f[0]  # Get the actual frame from the list
            if getattr(frame, "is_gap", False):
                continue

            # Access buffers (EventFrame typically stores in .data or .buffers)
            bufs = getattr(frame, "data", [])
            if not bufs and hasattr(frame, "buffers"):
                bufs = frame.buffers

            if len(bufs) > 0:
                # Found a frame that has data, check first element of list
                found_taps = bufs[0]
                break

        # Identity-like check for white noise input (MP Identity is peak at 0)
        assert len(found_taps) > 0

        # FIX: Flatten 2D array (e.g. shape (1, N)) to 1D before checking peaks
        if found_taps.ndim == 2:
            found_taps = found_taps.flatten()

        # Find peak location
        peak_idx = np.argmax(np.abs(found_taps))
        peak_val = np.abs(found_taps[peak_idx])

        # Since zero_latency=True, we expect a Minimum Phase filter.
        # Energy should be concentrated at the start (t=0).
        # For colored noise, it won't be a perfect delta, but peak should be at 0.
        assert peak_idx == 0, f"MP Whitener should peak at index 0, got {peak_idx}"
        assert peak_val > 0.1, f"Kernel energy too low? Peak is {peak_val}"


class TestPipelineDriftCorrectionKernel:
    """Integration tests for DriftCorrectionKernel using GWDataNoiseSource."""

    def test_pipeline_drift_correction(self):
        """Verify DriftCorrectionKernel handles spectral drift correctly.

        We simulate a Reference PSD that differs from the Live PSD by a
        known power-law drift term. We then verify that the pipeline produces
        a correction kernel matching the theoretical expectation.
        """
        pipeline = Pipeline()

        # 1. Source (Live Data)
        src = GWDataNoiseSource(
            name="source",
            channel_dict={"H1": "H1:STRAIN"},
            duration=10.0,
        )
        src_info = src.channel_info["H1"]
        rate = src_info["rate"]
        live_psd_lal = src_info["psd"]  # The "Truth" PSD of the source

        # 2. Create Reference PSD = Live * Drift
        # Drift D(f) = (f / 100Hz)^(-1).
        # Ref has HIGHER noise at low freq, LOWER at high freq compared to Live.
        ref_psd_lal = lal.CreateREAL8FrequencySeries(
            "H1_Ref",
            live_psd_lal.epoch,
            live_psd_lal.f0,
            live_psd_lal.deltaF,
            live_psd_lal.sampleUnits,
            len(live_psd_lal.data.data),
        )

        freqs = (
            live_psd_lal.f0
            + np.arange(len(live_psd_lal.data.data)) * live_psd_lal.deltaF
        )
        freqs[0] = freqs[1]  # Avoid DC division

        drift_factor = (freqs / 100.0) ** (-1.0)
        ref_psd_lal.data.data[:] = live_psd_lal.data.data * drift_factor

        # 3. Whiten
        whiten = Whiten(
            name="whiten",
            sink_pad_names=("H1",),
            psd_pad_name="spectrum",
            whiten_pad_name="hoft",
            input_sample_rate=rate,
            whiten_sample_rate=2048,
            fft_length=4,
            instrument="H1",
        )

        # 4. Kernel with drifted reference
        truncation = 64
        drift_corr = DriftCorrectionKernel(
            name="drift_corr",
            sink_pad_names=("spectrum",),
            filters_pad_name="filters",
            reference_psd=ref_psd_lal,
            truncation_samples=truncation,
            verbose=True,
        )

        # 5. Sinks
        kern_sink = CollectSink(
            name="kern_sink",
            sink_pad_names=("filters",),
        )
        null_sink = NullSink(
            name="null_sink",
            sink_pad_names=("hoft",),
        )

        pipeline.insert(
            src,
            whiten,
            drift_corr,
            kern_sink,
            null_sink,
            link_map={
                whiten.snks["H1"]: src.srcs["H1:STRAIN"],
                drift_corr.snks["spectrum"]: whiten.srcs["spectrum"],
                kern_sink.snks["filters"]: drift_corr.srcs["filters"],
                null_sink.snks["hoft"]: whiten.srcs["hoft"],
            },
        )

        pipeline.run()

        # --- Verification ---
        frames = kern_sink.collects["filters"]
        assert len(frames) > 0, "Sink should have collected frames"

        pipeline_kernel_taps = None

        # Iterate backwards to find the latest valid update
        for frame_list in reversed(frames):
            # Guard against empty list (gap/empty frame)
            if not frame_list:
                continue

            # Access first buffer
            buf = frame_list[0]

            # Access data
            if buf.data and len(buf.data) > 0:
                pipeline_kernel_taps = buf.data[0]
                break

        assert pipeline_kernel_taps is not None, "Pipeline produced no valid kernels"

        # Check Dimensions
        assert (
            pipeline_kernel_taps.size == truncation
        ), f"Kernel size {pipeline_kernel_taps.size} != truncation {truncation}"

        # Fix dimensions: (1, 64) -> (64,)
        if pipeline_kernel_taps.ndim == 2:
            pipeline_kernel_taps = pipeline_kernel_taps.flatten()

        # Calculate Expected Kernel
        expected_kernel = correction_kernel_from_psds(
            psd_live=live_psd_lal, psd_ref=ref_psd_lal, truncation_samples=truncation
        )
        expected_taps = expected_kernel.fir_matrix

        # Verify similarity
        norm_pipeline = pipeline_kernel_taps / np.linalg.norm(pipeline_kernel_taps)
        norm_expected = expected_taps / np.linalg.norm(expected_taps)

        similarity = np.dot(norm_pipeline, norm_expected)

        # We expect high similarity
        assert similarity > 0.95, f"Similarity {similarity:.4f} too low"
