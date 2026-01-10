"""Test coverage for sgnligo.sinks.psd_sink module."""

from unittest.mock import patch

import lal
import numpy as np
import pytest
from lal import LIGOTimeGPS
from sgn.apps import Pipeline
from sgn.base import Frame

from sgnligo.sinks.psd_sink import PSDSink
from sgnligo.sources.gwdata_noise_source import GWDataNoiseSource
from sgnligo.transforms.whiten import Whiten


@pytest.fixture
def mock_lal_psd():
    """Create a dummy LAL FrequencySeries for testing."""

    def _create(epoch=0):
        series = lal.CreateREAL8FrequencySeries(
            "psd",
            LIGOTimeGPS(epoch),
            0.0,
            1.0,
            lal.Unit("strain^2 s"),
            100,
        )
        series.data.data[:] = 1.0
        return series

    return _create


class TestPSDSinkInit:
    """Test initialization and property validation of PSDSink."""

    def test_defaults(self):
        """Verify that default values match legacy behavior."""
        # Must specify sink_pad_names to satisfy SinkElement assertion
        sink = PSDSink(sink_pad_names=["H1"])

        assert sink.fname == "psd.xml"
        assert sink.write_interval is None  # Disabled by default
        assert sink.output_format is None  # Inferred
        assert sink.verbose is False
        assert sink._current_psds == {}
        assert sink._last_write_gps == -1.0
        assert "H1" in sink.sink_pad_names

    def test_custom_init(self):
        """Verify custom initialization."""
        sink = PSDSink(
            name="custom_sink",
            sink_pad_names=["L1"],
            fname="test_{gps}.txt",
            output_format="txt",
            write_interval=10.0,
            verbose=True,
        )
        assert sink.fname == "test_{gps}.txt"
        assert sink.output_format == "txt"
        assert sink.write_interval == 10.0
        assert sink.verbose is True
        assert "L1" in sink.sink_pad_names


class TestPSDSinkLegacyBehavior:
    """
    Test that the sink behaves exactly like the legacy version
    when configured with defaults (write only at EOS).
    """

    @patch("sgnligo.sinks.psd_sink.PSDWriter")
    def test_write_only_at_eos(self, mock_writer, mock_lal_psd):
        """Ensure no writes occur until EOS is triggered."""
        # Initialize with a pad named "H1" so the stored PSD key is "H1"
        sink = PSDSink(fname="output.xml", sink_pad_names=["H1"])

        # Get the actual pad object created by the element
        pad = sink.snks["H1"]

        # 1. Pull Data (Not EOS)
        frame1 = Frame(metadata={"psd": mock_lal_psd(epoch=100)})
        sink.pull(pad, frame1)

        # Manually trigger internal loop
        sink.internal()

        # Should NOT have written yet
        mock_writer.write.assert_not_called()
        assert len(sink._current_psds) == 1
        assert "H1" in sink._current_psds

        # 2. Pull EOS
        frame_eos = Frame(EOS=True)
        sink.pull(pad, frame_eos)

        # Trigger internal loop (simulating pipeline loop)
        sink.internal()

        # Should have written ONCE
        mock_writer.write.assert_called_once()
        args, _ = mock_writer.write.call_args
        assert args[0] == "output.xml"
        # The dictionary passed to write should contain the key "H1"
        assert "H1" in args[1]


class TestPSDSinkPeriodic:
    """
    Test the new periodic writing feature based on Data Time (GPS).
    """

    @patch("sgnligo.sinks.psd_sink.PSDWriter")
    def test_periodic_gps_logic(self, mock_writer, mock_lal_psd):
        """
        Verify that writing is triggered by GPS time progression.
        """
        sink = PSDSink(
            fname="psd-{gps}.xml", write_interval=10.0, sink_pad_names=["H1"]
        )
        pad = sink.snks["H1"]

        # Helper to simulate pipeline step
        def step(gps_time):
            frame = Frame(metadata={"psd": mock_lal_psd(epoch=gps_time)})
            sink.pull(pad, frame)
            sink.internal()

        # T=100: Init state. _last_write_gps becomes 100. NO WRITE.
        step(100)
        mock_writer.write.assert_not_called()
        assert sink._last_write_gps == 100.0

        # T=105: Diff is 5. 5 < 10. NO WRITE.
        step(105)
        mock_writer.write.assert_not_called()

        # T=110: Diff is 10. 10 >= 10. WRITE.
        step(110)
        assert mock_writer.write.call_count == 1
        args, _ = mock_writer.write.call_args
        assert args[0] == "psd-110.xml"

        mock_writer.write.reset_mock()

        # T=115: Diff from 110 is 5. NO WRITE.
        step(115)
        mock_writer.write.assert_not_called()

        # T=125: Diff from 110 is 15. WRITE.
        step(125)
        mock_writer.write.assert_called_once()
        args, _ = mock_writer.write.call_args
        assert args[0] == "psd-125.xml"


class TestPSDSinkPipeline:
    """
    Full integration test running a real pipeline with fake data.
    """

    def test_integration_pipeline(self, tmp_path):
        """
        Run: GWDataNoiseSource -> Whiten -> PSDSink
        """
        pipe = Pipeline()

        # 1. Source: Fake Gaussian Noise
        # NOTE: GWDataNoiseSource generates data at 16384 Hz based on the fake PSD.
        src = GWDataNoiseSource(
            name="noise_src",
            channel_dict={"H1": "H1"},
            t0=1000000000,
            end=1000000004,  # 4 seconds
        )

        # 2. Transform: Whiten (estimates PSD)
        # Must match source rate (16384 Hz) and have explicit sink pad
        whitener = Whiten(
            name="whitener",
            sink_pad_names=["H1"],  # <--- FIXED: Must explicitly name input pad
            whiten_pad_name="hoft",
            psd_pad_name="psd",
            instrument="H1",
            input_sample_rate=16384,  # Matched to GWDataNoiseSource
            whiten_sample_rate=4096,  # Downsample output if desired
            fft_length=2,
            nmed=3,
            navg=2,
        )

        # 3. Sink: PSDSink
        output_file = tmp_path / "final_psd.xml"
        psd_sink = PSDSink(
            name="psd_sink",
            fname=str(output_file),
            sink_pad_names=["psd_in"],
            verbose=True,
        )

        # 4. Sink: Null sink for data
        from sgnts.sinks import NullSeriesSink

        null_sink = NullSeriesSink(name="null_sink", sink_pad_names=["data"])

        # Insert elements
        pipe.insert(src, whitener, psd_sink, null_sink)

        # Link elements using dictionary syntax {sink_pad: source_pad}
        pipe.link(
            {
                whitener.snks["H1"]: src.srcs["H1"],
                psd_sink.snks["psd_in"]: whitener.srcs["psd"],
                null_sink.snks["data"]: whitener.srcs["hoft"],
            }
        )

        # Run
        pipe.run()

        # Validation
        assert output_file.exists()

        # Verify content using LAL
        from igwn_ligolw import utils as ligolw_utils
        from lal.series import read_psd_xmldoc

        psd_dict = read_psd_xmldoc(
            ligolw_utils.load_filename(
                str(output_file),
                verbose=False,
                contenthandler=lal.series.PSDContentHandler,
            )
        )

        assert "psd_in" in psd_dict
        psd = psd_dict["psd_in"]

        assert psd.data.length > 0
        assert np.all(psd.data.data > 0)

    def test_integration_pipeline_periodic(self, tmp_path):
        """
        Run: GWDataNoiseSource -> Whiten -> PSDSink (Periodic)
        Verify: Multiple PSD files are written corresponding to data time intervals.
        """
        pipe = Pipeline()

        # 1. Source: 6 seconds of data
        # GWDataNoiseSource defaults to ~1s buffers, so 6s duration ensures
        # we trigger the 2s write_interval multiple times.
        start_gps = 1000000000
        duration = 12
        write_interval = 1.0

        src = GWDataNoiseSource(
            name="noise_src",
            channel_dict={"H1": "H1"},
            t0=start_gps,
            end=start_gps + duration,
        )

        # 2. Transform: Whiten
        whitener = Whiten(
            name="whitener",
            sink_pad_names=["H1"],
            whiten_pad_name="hoft",
            psd_pad_name="psd",
            instrument="H1",
            input_sample_rate=16384,
            whiten_sample_rate=4096,
            fft_length=2,
            nmed=3,
            navg=2,
        )

        # 3. Sink: PSDSink with Periodic Writing
        # We use a template filename to verify unique files are created
        base_name = "test_psd-{gps}.xml"
        output_template = tmp_path / base_name

        psd_sink = PSDSink(
            name="psd_sink",
            fname=str(output_template),
            sink_pad_names=["psd_in"],
            write_interval=write_interval,
            verbose=True,
        )

        # 4. Sink: Null sink for data stream
        from sgnts.sinks import NullSeriesSink

        null_sink = NullSeriesSink(name="null_sink", sink_pad_names=["data"])

        # Link
        pipe.insert(src, whitener, psd_sink, null_sink)
        pipe.link(
            {
                whitener.snks["H1"]: src.srcs["H1"],
                psd_sink.snks["psd_in"]: whitener.srcs["psd"],
                null_sink.snks["data"]: whitener.srcs["hoft"],
            }
        )

        # Run pipeline
        pipe.run()

        # Validation
        # With 6 seconds of data and 2s interval, we expect roughly 3 writes.
        # We check for the existence of multiple files to confirm periodic behavior.
        generated_files = list(tmp_path.glob("test_psd-*.xml"))
        generated_files.sort()

        # 1. Verify we got multiple files (proving periodic write)
        assert (
            len(generated_files) >= 2
        ), f"Expected periodic writes, but found only: {generated_files}"

        # 2. Verify filenames contain GPS timestamps (integers)
        for f in generated_files:
            gps_part = f.stem.split("-")[-1]
            assert gps_part.isdigit(), f"Filename {f.name} does not contain integer GPS"

            # Verify GPS is within our data range
            gps_val = int(gps_part)
            assert start_gps <= gps_val <= start_gps + duration

        # 3. Verify content of the last file
        from igwn_ligolw import utils as ligolw_utils
        from lal.series import read_psd_xmldoc

        last_file = generated_files[-1]
        psd_dict = read_psd_xmldoc(
            ligolw_utils.load_filename(
                str(last_file),
                verbose=False,
                contenthandler=lal.series.PSDContentHandler,
            )
        )
        assert "psd_in" in psd_dict
        assert psd_dict["psd_in"].data.length > 0
