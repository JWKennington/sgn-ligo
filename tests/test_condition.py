#!/usr/bin/env python3

import pathlib

import pytest
from sgn import NullSink
from sgn.apps import Pipeline
from sgnts.sources import FakeSeriesSource
from sgnts.transforms import Threshold

from sgnligo.transforms import Whiten

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
