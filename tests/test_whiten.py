#!/usr/bin/env python3

import os
from optparse import OptionParser

import pytest
from sgn.apps import Pipeline
from sgnts.sinks import DumpSeriesSink
from sgnts.transforms import Resampler

from sgnligo.sources import FrameReader
from sgnligo.transforms import Whiten


def parse_command_line():
    parser = OptionParser()

    parser.add_option(
        "--instrument", metavar="ifo", help="Instrument to analyze. H1, L1, or V1."
    )
    parser.add_option(
        "--gps-start-time",
        metavar="seconds",
        help="Set the start time of the segment to analyze in GPS seconds.",
    )
    parser.add_option(
        "--gps-end-time",
        metavar="seconds",
        help="Set the end time of the segment to analyze in GPS seconds.",
    )
    parser.add_option(
        "--output-dir", metavar="path", help="Directory to write output data into."
    )
    parser.add_option(
        "--sample-rate",
        metavar="Hz",
        type=int,
        default=16384,
        help="Requested sampling rate of the data.",
    )
    parser.add_option(
        "--buffer-duration",
        metavar="seconds",
        type=int,
        default=1,
        help="Length of output buffers in seconds. Default is 1 second.",
    )
    parser.add_option(
        "--frame-cache",
        metavar="file",
        help="Set the path to the frame cache file to analyze.",
    )
    parser.add_option(
        "--channel-name", metavar="channel", help="Name of the data channel to analyze."
    )
    parser.add_option(
        "--reference-psd",
        metavar="file",
        help="load the spectrum from this LIGO light-weight XML file (optional).",
    )
    parser.add_option(
        "--track-psd",
        action="store_true",
        help="Enable dynamic PSD tracking.  Always enabled if --reference-psd is not"
        " given.",
    )

    options, args = parser.parse_args()

    return options, args


@pytest.mark.skip(reason="Not currently pytest compatible")
def test_whitengraph(capsys):

    # parse arguments
    options, args = parse_command_line()

    os.makedirs(options.output_dir, exist_ok=True)

    num_samples = options.sample_rate * options.buffer_duration

    if not (options.gps_start_time and options.gps_end_time):
        raise ValueError("Must provide both --gps-start-time and --gps-end-time.")

    if options.reference_psd is None:
        options.track_psd = True  # FIXME not implemented

    pipeline = Pipeline()

    #
    #          ------   H1   -------
    #         | src1 | ---- | snk2  |
    #          ------   SR1  -------
    #         /
    #     H1 /
    #   ----------
    #  |  whiten  |
    #   ----------
    #          \
    #       H1  \
    #           ------
    #          | snk1 |
    #           ------
    #

    pipeline.insert(
        FrameReader(
            name="FrameReader",
            source_pad_names=("frsrc",),
            rate=options.sample_rate,
            num_samples=num_samples,
            framecache=options.frame_cache,
            channel_name=options.channel_name,
            instrument=options.instrument,
            gps_start_time=options.gps_start_time,
            gps_end_time=options.gps_end_time,
        ),
        Resampler(
            name="Resampler",
            source_pad_names=("resamp",),
            sink_pad_names=("frsrc",),
            inrate=options.sample_rate,
            outrate=2048,
        ),
        Whiten(
            name="Whitener",
            source_pad_names=("hoft", "spectrum"),
            sink_pad_names=("resamp",),
            instrument=options.instrument,
            sample_rate=2048,
            fft_length=4,
            reference_psd=options.reference_psd,
            psd_pad_name="spectrum",
            whiten_pad_name="hoft",
        ),
        DumpSeriesSink(
            name="HoftSnk",
            sink_pad_names=("hoft",),
            fname=os.path.join(options.output_dir, "out.txt"),
        ),
        DumpSeriesSink(
            name="SpectrumSnk",
            sink_pad_names=("spectrum",),
            fname=os.path.join(options.output_dir, "psd_out.txt"),
        ),
    )

    pipeline.insert(
        DumpSeriesSink(name="RawSnk", sink_pad_names=("frsrc",), fname="in.txt")
    )
    pipeline.insert(
        link_map={
            "Resampler:snk:frsrc": "FrameReader:src:frsrc",
            "Whitener:snk:resamp": "Resampler:src:resamp",
            "HoftSnk:snk:hoft": "Whitener:src:hoft",
            "SpectrumSnk:snk:spectrum": "Whitener:src:spectrum",
            "RawSnk:snk:frsrc": "FrameReader:src:frsrc",
        }
    )

    pipeline.run()


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

    def test_init_z_whiten_positive(self, tmp_path):
        """Test that z_whiten is always positive with valid parameters.

        Due to the calculation z_whiten = int(fft_length / 4 * whiten_sample_rate),
        z_whiten is always > 0 with valid sample rates and fft_length values.
        The else branch (self.tukey = None) is marked with pragma: no cover
        as it's unreachable with valid parameters.
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

        # z_whiten is always > 0 with valid parameters
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


if __name__ == "__main__":
    test_whitengraph(None)
