#!/usr/bin/env python3

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


def build_pipeline(
    instrument: str,
    sample_rate: int = 16384,
):
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


class TestCondition:
    """Test group for testing conditioning"""

    @pytest.fixture(scope="class", autouse=True)
    def pipeline(self):
        """Build the pipeline as a fixture"""
        return build_pipeline(
            instrument="H1",
            sample_rate=16384,
        )

    def test_run(self, pipeline):
        """Test Running the pipeline"""
        pipeline.run()


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

    def test_init_custom_values(self):
        """Test ConditionInfo with custom values."""
        info = ConditionInfo(
            whiten_sample_rate=4096,
            psd_fft_length=16,
            reference_psd=PATH_PSD.as_posix(),
            ht_gate_threshold=10.0,
            track_psd=False,
        )
        assert info.whiten_sample_rate == 4096
        assert info.psd_fft_length == 16
        assert info.reference_psd == PATH_PSD.as_posix()
        assert info.ht_gate_threshold == 10.0
        assert info.track_psd is False

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
            ]
        )
        assert args.psd_fft_length == 16
        assert args.reference_psd == "/path/to/psd.xml"
        assert args.track_psd is True
        assert args.whiten_sample_rate == 4096
        assert args.ht_gate_threshold == 10.0

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
            ]
        )

        info = ConditionInfo.from_options(args)
        assert info.whiten_sample_rate == 4096
        assert info.psd_fft_length == 16
        assert info.reference_psd == PATH_PSD.as_posix()
        assert info.ht_gate_threshold == 10.0
        assert info.track_psd is True


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
