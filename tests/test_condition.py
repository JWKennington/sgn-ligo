#!/usr/bin/env python3

import pathlib

import pytest
from sgn import NullSink
from sgn.apps import Pipeline
from sgnts.sources import FakeSeriesSrc
from sgnts.transforms import Resampler, Threshold

from sgnligo.transforms import Whiten

PATH_DATA = pathlib.Path(__file__).parent / "data"
PATH_PSD = PATH_DATA / "H1L1-GSTLAL-MEDIAN.xml.gz"


def build_pipeline(
    instrument: str,
    whitening_method,
    sample_rate: int = 16384,
):
    pipeline = Pipeline()

    pipeline.insert(
        FakeSeriesSrc(
            name=f"{instrument}_white",
            source_pad_names=("frsrc",),
            rate=sample_rate,
            signal_type="white",
            impulse_position=None,
            verbose=False,
            end=10,
        ),
        Resampler(
            name="Resampler",
            source_pad_names=("resamp",),
            sink_pad_names=("frsrc",),
            inrate=sample_rate,
            outrate=2048,
        ),
        Whiten(
            name="Whitener",
            source_pad_names=("hoft",),
            sink_pad_names=("resamp",),
            instrument=instrument,
            sample_rate=2048,
            fft_length=4,
            whitening_method=whitening_method,
            reference_psd=PATH_PSD.as_posix(),
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
            sink_pad_names=("hoft",),
        ),
    )
    pipeline.link(
        link_map={
            "Resampler:sink:frsrc": f"{instrument}_white:src:frsrc",
            "Whitener:sink:resamp": "Resampler:src:resamp",
            "Threshold:sink:data": "Whitener:src:hoft",
            "HoftSnk:sink:hoft": "Threshold:src:threshold",
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
            whitening_method="gstlal",
            sample_rate=16384,
        )

    def test_graph(self, pipeline):
        """Test the pipeline graph"""
        dot_str = pipeline.to_dot()
        assert dot_str.split("\n") == [
            "digraph {",
            "\tH1_white [label=H1_white]",
            "\tHoftSnk [label=HoftSnk]",
            "\tResampler [label=Resampler]",
            "\tThreshold [label=Threshold]",
            "\tWhitener [label=Whitener]",
            "\tH1_white -> Resampler",
            "\tResampler -> Whitener",
            "\tThreshold -> HoftSnk",
            "\tWhitener -> Threshold",
            "}",
            "",
        ]

    def test_run(self, pipeline):
        """Test Running the pipeline"""
        pipeline.run()
