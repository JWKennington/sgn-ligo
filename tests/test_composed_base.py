"""Tests for ComposedSourceBase.

These tests verify the base class functionality for composed source elements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Dict, List

import pytest
from sgnts.compose import TSCompose, TSComposedSourceElement
from sgnts.sources import FakeSeriesSource

from sgnligo.sources.composed_base import ComposedSourceBase

# --- Test fixtures: concrete implementations ---


@dataclass
class SimpleSource(ComposedSourceBase):
    """Minimal implementation for testing."""

    source_type: ClassVar[str] = "simple"
    description: ClassVar[str] = "Simple test source"

    sample_rate: int
    t0: float
    end: float

    def _build(self) -> TSComposedSourceElement:
        compose = TSCompose()
        fake = FakeSeriesSource(
            name=f"{self.name}_fake",
            source_pad_names=("output",),
            rate=self.sample_rate,
            t0=self.t0,
            end=self.end,
            signal_type="white",
        )
        compose.insert(fake)
        return compose.as_source(name=self.name)


@dataclass
class MultiChannelSource(ComposedSourceBase):
    """Source with multiple output channels."""

    source_type: ClassVar[str] = "multi"
    description: ClassVar[str] = "Multi-channel test source"

    ifos: List[str]
    channel_dict: Dict[str, str]
    sample_rate: int
    t0: float
    end: float

    def _validate(self) -> None:
        if set(self.ifos) != set(self.channel_dict.keys()):
            raise ValueError(
                f"ifos {self.ifos} must match channel_dict keys "
                f"{list(self.channel_dict.keys())}"
            )
        if self.t0 >= self.end:
            raise ValueError("t0 must be less than end")

    def _build(self) -> TSComposedSourceElement:
        compose = TSCompose()
        for ifo in self.ifos:
            channel = self.channel_dict[ifo]
            pad_name = f"{ifo}:{channel}"
            fake = FakeSeriesSource(
                name=f"{self.name}_{ifo}",
                source_pad_names=(pad_name,),
                rate=self.sample_rate,
                t0=self.t0,
                end=self.end,
                signal_type="white",
            )
            compose.insert(fake)
        return compose.as_source(name=self.name)


@dataclass
class SourceWithDefaults(ComposedSourceBase):
    """Source with default parameter values."""

    source_type: ClassVar[str] = "defaults"
    description: ClassVar[str] = "Source with defaults"

    t0: float
    end: float
    sample_rate: int = 4096
    verbose: bool = False

    def _build(self) -> TSComposedSourceElement:
        compose = TSCompose()
        fake = FakeSeriesSource(
            name=f"{self.name}_fake",
            source_pad_names=("output",),
            rate=self.sample_rate,
            t0=self.t0,
            end=self.end,
            signal_type="white",
        )
        compose.insert(fake)
        if self.verbose:
            print(f"Created source with sample_rate={self.sample_rate}")
        return compose.as_source(name=self.name)


# --- Tests ---


class TestComposedSourceBase:
    """Tests for ComposedSourceBase functionality."""

    def test_simple_instantiation(self):
        """Test basic instantiation with required parameters."""
        source = SimpleSource(
            name="test",
            sample_rate=4096,
            t0=1000.0,
            end=1010.0,
        )
        assert source.name == "test"
        assert source.sample_rate == 4096
        assert source.t0 == 1000.0
        assert source.end == 1010.0

    def test_srcs_property(self):
        """Test that .srcs returns the composed element's source pads."""
        source = SimpleSource(
            name="test",
            sample_rate=4096,
            t0=1000.0,
            end=1010.0,
        )
        assert "output" in source.srcs
        assert len(source.srcs) == 1

    def test_multi_channel_srcs(self):
        """Test source with multiple output channels."""
        source = MultiChannelSource(
            name="multi",
            ifos=["H1", "L1"],
            channel_dict={"H1": "STRAIN", "L1": "STRAIN"},
            sample_rate=4096,
            t0=1000.0,
            end=1010.0,
        )
        assert "H1:STRAIN" in source.srcs
        assert "L1:STRAIN" in source.srcs
        assert len(source.srcs) == 2

    def test_class_metadata(self):
        """Test source_type and description class variables."""
        assert SimpleSource.source_type == "simple"
        assert SimpleSource.description == "Simple test source"

        source = SimpleSource(
            name="test",
            sample_rate=4096,
            t0=1000.0,
            end=1010.0,
        )
        # Class vars accessible on instance too
        assert source.source_type == "simple"
        assert source.description == "Simple test source"

    def test_default_parameters(self):
        """Test that default parameter values work."""
        source = SourceWithDefaults(
            name="test",
            t0=1000.0,
            end=1010.0,
        )
        assert source.sample_rate == 4096  # default
        assert source.verbose is False  # default

    def test_override_defaults(self):
        """Test that default parameters can be overridden."""
        source = SourceWithDefaults(
            name="test",
            t0=1000.0,
            end=1010.0,
            sample_rate=16384,
            verbose=True,
        )
        assert source.sample_rate == 16384
        assert source.verbose is True


class TestValidation:
    """Tests for validation behavior."""

    def test_validation_called_before_build(self):
        """Test that _validate() is called before _build()."""
        with pytest.raises(ValueError, match="t0 must be less than end"):
            MultiChannelSource(
                name="test",
                ifos=["H1"],
                channel_dict={"H1": "STRAIN"},
                sample_rate=4096,
                t0=1010.0,  # After end!
                end=1000.0,
            )

    def test_validation_ifo_mismatch(self):
        """Test validation catches mismatched ifos and channel_dict."""
        with pytest.raises(ValueError, match="must match channel_dict keys"):
            MultiChannelSource(
                name="test",
                ifos=["H1", "L1"],  # Two IFOs
                channel_dict={"H1": "STRAIN"},  # Only one channel
                sample_rate=4096,
                t0=1000.0,
                end=1010.0,
            )

    def test_base_class_validation_is_noop(self):
        """Test that base _validate() does nothing by default."""
        # SimpleSource doesn't override _validate(), should work fine
        source = SimpleSource(
            name="test",
            sample_rate=4096,
            t0=1000.0,
            end=1010.0,
        )
        assert source.name == "test"


class TestDelegation:
    """Tests for attribute delegation to inner composed element."""

    def test_getattr_delegates_to_composed(self):
        """Test that unknown attributes are delegated to _composed."""
        source = SimpleSource(
            name="test",
            sample_rate=4096,
            t0=1000.0,
            end=1010.0,
        )
        # TSComposedSourceElement has these attributes
        assert hasattr(source, "internal_elements")
        assert hasattr(source, "sink_pads")

    def test_private_attrs_raise_attribute_error(self):
        """Test that private attributes don't delegate."""
        source = SimpleSource(
            name="test",
            sample_rate=4096,
            t0=1000.0,
            end=1010.0,
        )
        with pytest.raises(AttributeError, match="no attribute '_nonexistent'"):
            _ = source._nonexistent

    def test_nonexistent_attr_raises_from_composed(self):
        """Test that truly nonexistent attributes raise AttributeError."""
        source = SimpleSource(
            name="test",
            sample_rate=4096,
            t0=1000.0,
            end=1010.0,
        )
        with pytest.raises(AttributeError):
            _ = source.totally_nonexistent_attribute


class TestRepr:
    """Tests for repr behavior."""

    def test_repr_excludes_composed(self):
        """Test that _composed is excluded from repr (repr=False)."""
        source = SimpleSource(
            name="test",
            sample_rate=4096,
            t0=1000.0,
            end=1010.0,
        )
        repr_str = repr(source)
        assert "name='test'" in repr_str
        assert "sample_rate=4096" in repr_str
        # _composed should be excluded
        assert "_composed" not in repr_str


class TestElementProperty:
    """Tests for the .element property."""

    def test_element_returns_composed(self):
        """Test that .element returns the inner TSComposedSourceElement."""
        from sgnts.compose import TSComposedSourceElement

        source = SimpleSource(
            name="test",
            sample_rate=4096,
            t0=1000.0,
            end=1010.0,
        )
        assert isinstance(source.element, TSComposedSourceElement)
        assert source.element is source._composed


class TestPipelineIntegration:
    """Tests for pipeline integration."""

    def test_connect_to_sink(self):
        """Test that composed source can be connected to a sink via .element."""
        from sgn.apps import Pipeline
        from sgn.sinks import NullSink

        source = SimpleSource(
            name="test",
            sample_rate=4096,
            t0=1000.0,
            end=1001.0,  # Short duration
        )

        sink = NullSink(name="sink", sink_pad_names=["output"])

        pipeline = Pipeline()
        # Use .element for pipeline integration
        pipeline.connect(source.element, sink)
        pipeline.run()

    def test_multi_channel_pipeline(self):
        """Test multi-channel source in pipeline."""
        from sgn.apps import Pipeline
        from sgn.sinks import NullSink

        source = MultiChannelSource(
            name="multi",
            ifos=["H1", "L1"],
            channel_dict={"H1": "STRAIN", "L1": "STRAIN"},
            sample_rate=4096,
            t0=1000.0,
            end=1001.0,
        )

        sink = NullSink(
            name="sink",
            sink_pad_names=list(source.srcs.keys()),
        )

        pipeline = Pipeline()
        # Use .element for pipeline integration
        pipeline.connect(source.element, sink)
        pipeline.run()
