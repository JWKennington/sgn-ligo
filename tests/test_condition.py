#!/usr/bin/env python3
import os
import pathlib
from argparse import ArgumentParser

import pytest
from sgn import NullSink
from sgn.apps import Pipeline
from sgnts.sources import FakeSeriesSource
from sgnts.transforms import Threshold

from sgnligo.transforms import Whiten
from sgnligo.transforms.condition import ConditionInfo, condition

PATH_DATA = pathlib.Path(__file__).parent / "data"
PATH_PSD = PATH_DATA / "H1L1-GSTLAL-MEDIAN.xml.gz"
# Optional plotting/graph tests flag (mirrors SciVal tests)
OPT_PLOT_ENV = "SGN_SHOW_OPTIONAL_PLOT_TESTS"
SKIP_OPT_PLOTS = pytest.mark.skipif(
    # False,
    not os.environ.get(OPT_PLOT_ENV),
    reason=f"optional plotting tests disabled; set {OPT_PLOT_ENV}=1 to enable",
)


class TestCondition:
    """Test group for testing conditioning"""

    @pytest.fixture
    def pipeline_manual(self):
        """Build the pipeline as a fixture"""
        instrument = "H1"
        sample_rate = 16384
        pipeline = Pipeline()

        pipeline.insert(
            FakeSeriesSource(
                name=f"{instrument}_white",
                source_pad_names=("frsrc",),
                rate=sample_rate,
                signal_type="white",
                impulse_position=None,
                end=10,
            ),
            Whiten(
                name="Whitener",
                sink_pad_names=("resamp",),
                instrument=instrument,
                input_sample_rate=sample_rate,
                whiten_sample_rate=2048,
                fft_length=4,
                reference_psd=PATH_PSD.as_posix(),
                psd_pad_name="spectrum",
                whiten_pad_name="hoft",
            ),
            Threshold(
                name="Threshold",
                source_pad_names=("threshold",),
                sink_pad_names=("data",),
                threshold=7,
                startwn=1024,
                stopwn=1024,
                invert=True,
            ),
            NullSink(
                name="HoftSnk",
                sink_pad_names=("hoft", "spectrum"),
            ),
        )
        pipeline.link(
            link_map={
                "Whitener:snk:resamp": f"{instrument}_white:src:frsrc",
                "Threshold:snk:data": "Whitener:src:hoft",
                "HoftSnk:snk:hoft": "Threshold:src:threshold",
                "HoftSnk:snk:spectrum": "Whitener:src:spectrum",
            }
        )
        return pipeline

    @pytest.fixture
    def pipeline(self):
        """Build the pipeline via condition() to mirror pipeline_manual behavior."""
        instrument = "H1"
        sample_rate = 16384
        pipeline = Pipeline()

        # Upstream white noise source (matches manual fixture duration and type)
        pipeline.insert(
            FakeSeriesSource(
                name=f"{instrument}_white",
                source_pad_names=(instrument,),
                rate=sample_rate,
                signal_type="white",
                impulse_position=None,
                end=10,
            )
        )

        # Condition subgraph using ConditionInfo matching manual parameters
        info = ConditionInfo(
            psd_fft_length=4,
            reference_psd=PATH_PSD.as_posix(),
            whiten_sample_rate=2048,
            ht_gate_threshold=7,  # enable gating like manual Threshold
        )
        cond_links, spec_links, _ = condition(
            pipeline=pipeline,
            condition_info=info,
            ifos=[instrument],
            data_source="white",
            input_sample_rate=sample_rate,
            input_links={instrument: f"{instrument}_white:src:{instrument}"},
            whiten_latency=False,
            highpass_filter=False,
        )

        # Downstream sinks to collect whitened data and PSD (QA)
        pipeline.insert(
            NullSink(
                name="HoftSnk",
                sink_pad_names=("hoft", "spectrum"),
            ),
            link_map={
                "HoftSnk:snk:hoft": cond_links[instrument],
                "HoftSnk:snk:spectrum": spec_links[instrument],
            },
        )

        return pipeline

    def test_pipeline_equivalence(self, pipeline, pipeline_manual):
        """Test that the two pipeline building methods yield the same structure.

        We compare the element-level edge graphs. The condition() wiring prefixes
        some element names with the IFO (e.g., H1_Whitener vs Whitener). Normalize
        those names before comparing.
        """
        # Get element-level edges (no pads)
        edges_manual = set(pipeline_manual.edges(pads=False))
        edges_cond = set(pipeline.edges(pads=False))

        # Derive instrument prefix from any '*_white' element name (source)
        def infer_ifo(names: set[str]) -> str | None:
            for n in names:
                if n.endswith("_white") and "_" in n:
                    return n.split("_", 1)[0]
            return None

        # Collect unique element names from condition pipeline
        elem_names_cond = {e for pair in edges_cond for e in pair}
        ifo = infer_ifo(elem_names_cond)

        # Normalize only known IFO-prefixed processing elements
        KNOWN_PREFIXED = {"Whitener", "Threshold"}

        def norm(name: str) -> str:
            if ifo and name.startswith(f"{ifo}_"):
                rest = name[len(ifo) + 1 :]
                if rest in KNOWN_PREFIXED:
                    return rest
            return name

        edges_cond_norm = {(norm(a), norm(b)) for (a, b) in edges_cond}

        if edges_cond_norm != edges_manual:
            extra_in_cond = sorted(edges_cond_norm - edges_manual)
            extra_in_manual = sorted(edges_manual - edges_cond_norm)
            raise AssertionError(
                "Pipelines differ after normalization.\n"
                f"Extra edges in condition(): {extra_in_cond}\n"
                f"Missing edges (present only in manual): {extra_in_manual}"
            )

    def test_run(self, pipeline):
        """Test Running the pipeline"""
        pipeline.run()

    @SKIP_OPT_PLOTS
    def test_visualize_pipeline_graph(self, pipeline):
        """Optionally render and save the classic conditioning pipeline graph."""
        out = pathlib.Path("condition_standard.png")
        # Render element graph by default
        pipeline.visualize(str(out))
        assert out.exists()


class TestConditionInfo:
    """Test ConditionInfo dataclass."""

    def test_init_defaults(self):
        """Test ConditionInfo with default values."""
        info = ConditionInfo()
        assert info.whiten_sample_rate == 2048
        assert info.psd_fft_length == 8
        assert info.reference_psd is None
        assert info.ht_gate_threshold == float("+inf")
        assert info.track_psd is True
        assert info.zero_latency is False
        assert info.detailed_latency is False

    def test_init_custom_values(self):
        """Test ConditionInfo with custom values."""
        info = ConditionInfo(
            whiten_sample_rate=4096,
            psd_fft_length=16,
            reference_psd=PATH_PSD.as_posix(),
            ht_gate_threshold=10.0,
            track_psd=False,
            zero_latency=True,
            detailed_latency=True,
        )
        assert info.whiten_sample_rate == 4096
        assert info.psd_fft_length == 16
        assert info.reference_psd == PATH_PSD.as_posix()
        assert info.ht_gate_threshold == 10.0
        assert info.track_psd is False
        assert info.zero_latency is True
        assert info.detailed_latency is True

    def test_validate_raises_without_psd_and_track_psd_false(self):
        """Test validate raises ValueError when no psd and tracking disabled."""
        with pytest.raises(ValueError, match="Must enable track_psd"):
            ConditionInfo(reference_psd=None, track_psd=False)

    def test_append_options(self):
        """Test append_options adds argument groups to parser."""
        parser = ArgumentParser()
        ConditionInfo.append_options(parser)

        # Parse with defaults
        args = parser.parse_args([])
        assert args.psd_fft_length == 8
        assert args.reference_psd is None
        assert args.track_psd is True
        assert args.whiten_sample_rate == 2048
        assert args.ht_gate_threshold == float("+inf")
        assert args.zero_latency is False
        assert args.detailed_latency is False

    def test_append_options_with_values(self):
        """Test append_options parses custom values."""
        parser = ArgumentParser()
        ConditionInfo.append_options(parser)

        args = parser.parse_args(
            [
                "--psd-fft-length",
                "16",
                "--reference-psd",
                "/path/to/psd.xml",
                "--track-psd",
                "--whiten-sample-rate",
                "4096",
                "--ht-gate-threshold",
                "10.0",
                "--zero-latency",
                "--detailed-latency",
            ]
        )
        assert args.psd_fft_length == 16
        assert args.reference_psd == "/path/to/psd.xml"
        assert args.track_psd is True
        assert args.whiten_sample_rate == 4096
        assert args.ht_gate_threshold == 10.0
        assert args.zero_latency is True
        assert args.detailed_latency is True

    def test_from_options(self):
        """Test from_options creates ConditionInfo from parsed options."""
        parser = ArgumentParser()
        ConditionInfo.append_options(parser)

        args = parser.parse_args(
            [
                "--psd-fft-length",
                "16",
                "--reference-psd",
                PATH_PSD.as_posix(),
                "--whiten-sample-rate",
                "4096",
                "--ht-gate-threshold",
                "10.0",
                "--zero-latency",
            ]
        )

        info = ConditionInfo.from_options(args)
        assert info.whiten_sample_rate == 4096
        assert info.psd_fft_length == 16
        assert info.reference_psd == PATH_PSD.as_posix()
        assert info.ht_gate_threshold == 10.0
        assert info.track_psd is True
        assert info.zero_latency is True


class TestConditionFunction:
    """Test the condition function."""

    def test_condition_without_gate(self):
        """Test condition function without ht_gate (infinite threshold)."""
        pipeline = Pipeline()

        # Add a fake source
        pipeline.insert(
            FakeSeriesSource(
                name="H1_src",
                source_pad_names=("H1",),
                rate=16384,
                signal_type="white",
                end=2,
            )
        )

        condition_info = ConditionInfo(
            reference_psd=PATH_PSD.as_posix(),
            ht_gate_threshold=float("+inf"),  # No gating
        )

        cond_out, spec_out, lat_out = condition(
            pipeline=pipeline,
            condition_info=condition_info,
            ifos=["H1"],
            data_source="white",
            input_sample_rate=16384,
            input_links={"H1": "H1_src:src:H1"},
            whiten_latency=False,
        )

        assert cond_out["H1"] == "H1_Whitener:src:H1"
        assert spec_out["H1"] == "H1_Whitener:src:spectrum_H1"
        assert lat_out is None

    def test_condition_with_gate(self):
        """Test condition function with ht_gate (finite threshold)."""
        pipeline = Pipeline()

        pipeline.insert(
            FakeSeriesSource(
                name="L1_src",
                source_pad_names=("L1",),
                rate=16384,
                signal_type="white",
                end=2,
            )
        )

        condition_info = ConditionInfo(
            reference_psd=PATH_PSD.as_posix(),
            ht_gate_threshold=10.0,  # Apply gating
        )

        cond_out, spec_out, lat_out = condition(
            pipeline=pipeline,
            condition_info=condition_info,
            ifos=["L1"],
            data_source="white",
            input_sample_rate=16384,
            input_links={"L1": "L1_src:src:L1"},
            whiten_latency=False,
        )

        # With gating, output comes from Threshold
        assert cond_out["L1"] == "L1_Threshold:src:L1"
        assert spec_out["L1"] == "L1_Whitener:src:spectrum_L1"
        assert lat_out is None

    def test_condition_with_latency(self):
        """Test condition function with whiten_latency=True."""
        pipeline = Pipeline()

        pipeline.insert(
            FakeSeriesSource(
                name="H1_src",
                source_pad_names=("H1",),
                rate=16384,
                signal_type="white",
                end=2,
            )
        )

        condition_info = ConditionInfo(
            reference_psd=PATH_PSD.as_posix(),
        )

        cond_out, spec_out, lat_out = condition(
            pipeline=pipeline,
            condition_info=condition_info,
            ifos=["H1"],
            data_source="white",
            input_sample_rate=16384,
            input_links={"H1": "H1_src:src:H1"},
            whiten_latency=True,
        )

        assert cond_out["H1"] == "H1_Whitener:src:H1"
        assert spec_out["H1"] == "H1_Whitener:src:spectrum_H1"
        assert lat_out["H1"] == "H1_Latency:src:H1"

    def test_condition_custom_whiten_sample_rate(self):
        """Test condition function with custom whiten_sample_rate."""
        pipeline = Pipeline()

        pipeline.insert(
            FakeSeriesSource(
                name="H1_src",
                source_pad_names=("H1",),
                rate=16384,
                signal_type="white",
                end=2,
            )
        )

        condition_info = ConditionInfo(
            reference_psd=PATH_PSD.as_posix(),
            whiten_sample_rate=2048,
        )

        # Pass a different whiten_sample_rate to override
        cond_out, spec_out, lat_out = condition(
            pipeline=pipeline,
            condition_info=condition_info,
            ifos=["H1"],
            data_source="white",
            input_sample_rate=16384,
            input_links={"H1": "H1_src:src:H1"},
            whiten_sample_rate=4096,  # Override
        )

        assert cond_out["H1"] == "H1_Whitener:src:H1"

    def test_condition_multiple_ifos(self):
        """Test condition function with multiple IFOs."""
        pipeline = Pipeline()

        for ifo in ["H1", "L1"]:
            pipeline.insert(
                FakeSeriesSource(
                    name=f"{ifo}_src",
                    source_pad_names=(ifo,),
                    rate=16384,
                    signal_type="white",
                    end=2,
                )
            )

        condition_info = ConditionInfo(
            reference_psd=PATH_PSD.as_posix(),
        )

        cond_out, spec_out, lat_out = condition(
            pipeline=pipeline,
            condition_info=condition_info,
            ifos=["H1", "L1"],
            data_source="white",
            input_sample_rate=16384,
            input_links={
                "H1": "H1_src:src:H1",
                "L1": "L1_src:src:L1",
            },
        )

        assert cond_out["H1"] == "H1_Whitener:src:H1"
        assert cond_out["L1"] == "L1_Whitener:src:L1"
        assert spec_out["H1"] == "H1_Whitener:src:spectrum_H1"
        assert spec_out["L1"] == "L1_Whitener:src:spectrum_L1"

    def test_condition_with_gate_and_latency(self):
        """Test condition function with both gating and latency."""
        pipeline = Pipeline()

        pipeline.insert(
            FakeSeriesSource(
                name="H1_src",
                source_pad_names=("H1",),
                rate=16384,
                signal_type="white",
                end=2,
            )
        )

        condition_info = ConditionInfo(
            reference_psd=PATH_PSD.as_posix(),
            ht_gate_threshold=10.0,
        )

        cond_out, spec_out, lat_out = condition(
            pipeline=pipeline,
            condition_info=condition_info,
            ifos=["H1"],
            data_source="white",
            input_sample_rate=16384,
            input_links={"H1": "H1_src:src:H1"},
            whiten_latency=True,
        )

        assert cond_out["H1"] == "H1_Threshold:src:H1"
        assert spec_out["H1"] == "H1_Whitener:src:spectrum_H1"
        assert lat_out["H1"] == "H1_Latency:src:H1"


class TestConditionZeroLatency:
    """Tests for zero-latency conditioning wiring."""

    @pytest.fixture
    def pipeline_config(self):
        pipeline = Pipeline()
        # Simple white noise source at 4096 Hz
        pipeline.insert(
            FakeSeriesSource(
                name="H1_white",
                source_pad_names=("H1",),
                rate=4096,
                signal_type="white",
                impulse_position=None,
                end=5,
            )
        )
        # Build condition subgraph with zero-latency enabled via ConditionInfo
        # We downsample to 2048 Hz to trigger the Resampler path
        info = ConditionInfo(
            psd_fft_length=4,
            reference_psd=PATH_PSD.as_posix(),
            whiten_sample_rate=2048,
            zero_latency=True,
            ht_gate_threshold=7,
        )
        cond_links, spec_links, lat_links = condition(
            pipeline=pipeline,
            condition_info=info,
            ifos=["H1"],
            data_source="white",
            input_sample_rate=4096,
            input_links={"H1": "H1_white:src:H1"},
            whiten_latency=False,
        )
        # Attach a sink so the pipeline has a terminal consumer and can run
        pipeline.insert(
            NullSink(
                name="HoftSnk",
                sink_pad_names=("hoft", "spectrum"),
            ),
            link_map={
                "HoftSnk:snk:hoft": cond_links["H1"],
                "HoftSnk:snk:spectrum": spec_links["H1"],
            },
        )
        return pipeline, cond_links, spec_links

    @pytest.fixture
    def pipeline(self, pipeline_config):
        pipeline, _, _ = pipeline_config
        return pipeline

    def test_zero_latency_pipeline_runs(self, pipeline):
        """Test that the zero-latency pipeline constructs and runs without error."""
        pipeline.run()

    def test_zero_latency_rejects_upsampling(self):
        """Ensure zero-latency path refuses Resampler upsampling."""
        pipeline = Pipeline()
        pipeline.insert(
            FakeSeriesSource(
                name="H1_white",
                source_pad_names=("H1",),
                rate=2048,
                signal_type="white",
                end=2,
            )
        )
        info = ConditionInfo(
            psd_fft_length=4,
            reference_psd=None,
            whiten_sample_rate=4096,  # 4096 > 2048 -> Upsampling
            zero_latency=True,
        )
        with pytest.raises(ValueError, match="Zero-latency path requires downsampling"):
            condition(
                pipeline=pipeline,
                condition_info=info,
                ifos=["H1"],
                data_source="white",
                input_sample_rate=2048,
                input_links={"H1": "H1_white:src:H1"},
            )

    @SKIP_OPT_PLOTS
    def test_visualize_zero_latency_pipeline_graph(self, pipeline):
        """Render the zero-latency conditioning pipeline graph."""
        out = pathlib.Path("condition_zero_latency.png")
        pipeline.visualize(str(out))
        assert out.exists()


class TestConditionDetailedLatency:
    """Tests for detailed latency telemetry wiring."""

    @pytest.fixture
    def pipeline(self):
        pipeline = Pipeline()
        pipeline.insert(
            FakeSeriesSource(
                name="H1_white",
                source_pad_names=("H1",),
                rate=4096,
                signal_type="white",
                end=2,
            )
        )
        info = ConditionInfo(
            psd_fft_length=4,
            reference_psd=PATH_PSD.as_posix(),
            whiten_sample_rate=2048,
            zero_latency=True,
            detailed_latency=True,
        )

        # NOTE: We disable whiten_latency (final) to ensure the detailed
        # elements (like H1_Lat_Drift) are inserted for verification.
        # If whiten_latency=True, H1_Lat_Drift would be optimized out.
        cond_links, _, lat_links = condition(
            pipeline=pipeline,
            condition_info=info,
            ifos=["H1"],
            data_source="white",
            input_sample_rate=4096,
            input_links={"H1": "H1_white:src:H1"},
            whiten_latency=False,
        )

        pipeline.insert(
            NullSink(name="HoftSnk", sink_pad_names=("hoft",)),
            link_map={"HoftSnk:snk:hoft": cond_links["H1"]},
        )

        # Sink detailed latency outputs dynamically based on returned links
        if lat_links:
            sink_pads = tuple(f"lat_{k}" for k in lat_links.keys())
            lat_link_map = {
                f"LatencySnk:snk:lat_{k}": link for k, link in lat_links.items()
            }
            pipeline.insert(
                NullSink(name="LatencySnk", sink_pad_names=sink_pads),
                link_map=lat_link_map,
            )

        return pipeline

    def test_detailed_latency_structure(self, pipeline):
        """Check that intermediate Latency elements are inserted."""
        element_names = [e.name for e in pipeline.elements]
        assert "H1_Lat_Resamp" in element_names
        assert "H1_Lat_Whiten" in element_names
        assert "H1_Lat_Drift" in element_names
        # H1_Latency should NOT be present because whiten_latency=False
        assert "H1_Latency" not in element_names

        pipeline.run()

    @SKIP_OPT_PLOTS
    def test_visualize_detailed_latency_graph(self, pipeline):
        """Render the detailed latency pipeline graph."""
        out = pathlib.Path("condition_detailed_latency.png")
        pipeline.visualize(str(out))
        assert out.exists()
