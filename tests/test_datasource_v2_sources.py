"""Tests for datasource_v2 source classes.

These tests verify the new dataclass-based source classes work correctly.
"""

from __future__ import annotations

import pytest
from sgn.apps import Pipeline
from sgn.sinks import NullSink
from sgnts.compose import TSComposedSourceElement

from sgnligo.sources.composed_base import ComposedSourceBase
from sgnligo.sources.datasource_v2.sources import (
    DevShmComposedSource,
    FramesComposedSource,
    GWDataNoiseComposedSource,
    GWDataNoiseRealtimeComposedSource,
    ImpulseComposedSource,
    ImpulseRealtimeComposedSource,
    SinComposedSource,
    SinRealtimeComposedSource,
    WhiteComposedSource,
    WhiteRealtimeComposedSource,
)

# Arrakis is optional (requires sgn_arrakis)
try:
    from sgnligo.sources.datasource_v2.sources.arrakis import ArrakisComposedSource

    HAS_ARRAKIS = True
except ImportError:
    HAS_ARRAKIS = False


class TestWhiteComposedSource:
    """Tests for WhiteComposedSource."""

    def test_inherits_from_composed_base(self):
        """WhiteComposedSource should inherit from ComposedSourceBase."""
        assert issubclass(WhiteComposedSource, ComposedSourceBase)

    def test_instantiation(self):
        """Test basic instantiation."""
        source = WhiteComposedSource(
            name="test",
            ifos=["H1"],
            sample_rate=4096,
            t0=1000,
            end=1010,
        )
        assert source.name == "test"
        assert source.ifos == ["H1"]
        assert source.sample_rate == 4096
        assert source.t0 == 1000
        assert source.end == 1010

    def test_class_metadata(self):
        """Test source_type and description."""
        assert WhiteComposedSource.source_type == "white"
        assert WhiteComposedSource.description == "Gaussian white noise"

    def test_element_property(self):
        """Test .element returns TSComposedSourceElement."""
        source = WhiteComposedSource(
            name="test",
            ifos=["H1"],
            sample_rate=4096,
            t0=1000,
            end=1010,
        )
        assert isinstance(source.element, TSComposedSourceElement)

    def test_srcs_property(self):
        """Test source pads are created correctly."""
        source = WhiteComposedSource(
            name="test",
            ifos=["H1", "L1"],
            sample_rate=4096,
            t0=1000,
            end=1010,
        )
        assert "H1" in source.srcs
        assert "L1" in source.srcs

    def test_pipeline_integration(self):
        """Test source works in pipeline."""
        source = WhiteComposedSource(
            name="test",
            ifos=["H1"],
            sample_rate=4096,
            t0=1000,
            end=1001,
        )
        sink = NullSink(name="sink", sink_pad_names=["H1"])
        pipeline = Pipeline()
        pipeline.connect(source.element, sink)
        pipeline.run()

    def test_validation_t0_before_end(self):
        """t0 must be before end."""
        with pytest.raises(ValueError, match="t0 must be less than end"):
            WhiteComposedSource(
                name="test",
                ifos=["H1"],
                sample_rate=4096,
                t0=1010,
                end=1000,
            )

    def test_validation_positive_sample_rate(self):
        """sample_rate must be positive."""
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            WhiteComposedSource(
                name="test",
                ifos=["H1"],
                sample_rate=0,
                t0=1000,
                end=1010,
            )


class TestSinComposedSource:
    """Tests for SinComposedSource."""

    def test_instantiation(self):
        """Test basic instantiation."""
        source = SinComposedSource(
            name="test",
            ifos=["H1"],
            sample_rate=4096,
            t0=1000,
            end=1010,
        )
        assert source.name == "test"
        assert SinComposedSource.source_type == "sin"
        assert SinComposedSource.description == "Sinusoidal test signal"

    def test_pipeline_integration(self):
        """Test source works in pipeline."""
        source = SinComposedSource(
            name="test",
            ifos=["H1"],
            sample_rate=4096,
            t0=1000,
            end=1001,
        )
        sink = NullSink(name="sink", sink_pad_names=["H1"])
        pipeline = Pipeline()
        pipeline.connect(source.element, sink)
        pipeline.run()


class TestImpulseComposedSource:
    """Tests for ImpulseComposedSource."""

    def test_instantiation(self):
        """Test basic instantiation."""
        source = ImpulseComposedSource(
            name="test",
            ifos=["H1"],
            sample_rate=4096,
            t0=1000,
            end=1010,
        )
        assert source.name == "test"
        assert ImpulseComposedSource.source_type == "impulse"
        assert ImpulseComposedSource.description == "Impulse test signal"

    def test_impulse_position(self):
        """Test impulse_position parameter."""
        source = ImpulseComposedSource(
            name="test",
            ifos=["H1"],
            sample_rate=4096,
            t0=1000,
            end=1010,
            impulse_position=100,
        )
        assert source.impulse_position == 100

    def test_default_impulse_position(self):
        """Test default impulse_position is -1 (random)."""
        source = ImpulseComposedSource(
            name="test",
            ifos=["H1"],
            sample_rate=4096,
            t0=1000,
            end=1010,
        )
        assert source.impulse_position == -1

    def test_pipeline_integration(self):
        """Test source works in pipeline."""
        source = ImpulseComposedSource(
            name="test",
            ifos=["H1"],
            sample_rate=4096,
            t0=1000,
            end=1001,
        )
        sink = NullSink(name="sink", sink_pad_names=["H1"])
        pipeline = Pipeline()
        pipeline.connect(source.element, sink)
        pipeline.run()


class TestWhiteRealtimeComposedSource:
    """Tests for WhiteRealtimeComposedSource."""

    def test_instantiation(self):
        """Test basic instantiation."""
        source = WhiteRealtimeComposedSource(
            name="test",
            ifos=["H1"],
            sample_rate=4096,
        )
        assert source.name == "test"
        assert source.ifos == ["H1"]
        assert source.sample_rate == 4096
        assert WhiteRealtimeComposedSource.source_type == "white-realtime"

    def test_default_t0(self):
        """Test default t0 is 0."""
        source = WhiteRealtimeComposedSource(
            name="test",
            ifos=["H1"],
            sample_rate=4096,
        )
        assert source.t0 == 0

    def test_element_property(self):
        """Test .element returns TSComposedSourceElement."""
        source = WhiteRealtimeComposedSource(
            name="test",
            ifos=["H1"],
            sample_rate=4096,
        )
        assert isinstance(source.element, TSComposedSourceElement)

    def test_srcs_property(self):
        """Test source pads are created correctly."""
        source = WhiteRealtimeComposedSource(
            name="test",
            ifos=["H1", "L1"],
            sample_rate=4096,
        )
        assert "H1" in source.srcs
        assert "L1" in source.srcs


class TestSinRealtimeComposedSource:
    """Tests for SinRealtimeComposedSource."""

    def test_instantiation(self):
        """Test basic instantiation."""
        source = SinRealtimeComposedSource(
            name="test",
            ifos=["H1"],
            sample_rate=4096,
        )
        assert source.name == "test"
        assert SinRealtimeComposedSource.source_type == "sin-realtime"


class TestImpulseRealtimeComposedSource:
    """Tests for ImpulseRealtimeComposedSource."""

    def test_instantiation(self):
        """Test basic instantiation."""
        source = ImpulseRealtimeComposedSource(
            name="test",
            ifos=["H1"],
            sample_rate=4096,
        )
        assert source.name == "test"
        assert ImpulseRealtimeComposedSource.source_type == "impulse-realtime"

    def test_impulse_position(self):
        """Test impulse_position parameter."""
        source = ImpulseRealtimeComposedSource(
            name="test",
            ifos=["H1"],
            sample_rate=4096,
            impulse_position=50,
        )
        assert source.impulse_position == 50


class TestMultiIFO:
    """Tests for multi-IFO sources."""

    def test_white_source_multi_ifo(self):
        """Test WhiteComposedSource with multiple IFOs."""
        source = WhiteComposedSource(
            name="test",
            ifos=["H1", "L1", "V1"],
            sample_rate=4096,
            t0=1000,
            end=1001,
        )
        assert "H1" in source.srcs
        assert "L1" in source.srcs
        assert "V1" in source.srcs

        sink = NullSink(name="sink", sink_pad_names=["H1", "L1", "V1"])
        pipeline = Pipeline()
        pipeline.connect(source.element, sink)
        pipeline.run()


# --- Tests for GWDataNoiseComposedSource ---


class TestGWDataNoiseComposedSource:
    """Tests for GWDataNoiseComposedSource."""

    def test_inherits_from_composed_base(self):
        """GWDataNoiseComposedSource should inherit from ComposedSourceBase."""
        assert issubclass(GWDataNoiseComposedSource, ComposedSourceBase)

    def test_instantiation(self):
        """Test basic instantiation."""
        source = GWDataNoiseComposedSource(
            name="test",
            ifos=["H1"],
            t0=1000,
            end=1010,
        )
        assert source.name == "test"
        assert source.ifos == ["H1"]
        assert source.t0 == 1000
        assert source.end == 1010

    def test_class_metadata(self):
        """Test source_type and description."""
        assert GWDataNoiseComposedSource.source_type == "gwdata-noise"
        assert (
            GWDataNoiseComposedSource.description
            == "Colored Gaussian noise with LIGO PSD"
        )

    def test_element_property(self):
        """Test .element returns TSComposedSourceElement."""
        source = GWDataNoiseComposedSource(
            name="test",
            ifos=["H1"],
            t0=1000,
            end=1010,
        )
        assert isinstance(source.element, TSComposedSourceElement)

    def test_srcs_property(self):
        """Test source pads are created correctly."""
        source = GWDataNoiseComposedSource(
            name="test",
            ifos=["H1", "L1"],
            t0=1000,
            end=1010,
        )
        # Without state vector gating, pads use channel names
        assert "H1:FAKE-STRAIN" in source.srcs
        assert "L1:FAKE-STRAIN" in source.srcs

    def test_pipeline_integration(self):
        """Test source works in pipeline."""
        source = GWDataNoiseComposedSource(
            name="test",
            ifos=["H1"],
            t0=1000,
            end=1001,
        )
        sink = NullSink(name="sink", sink_pad_names=["H1:FAKE-STRAIN"])
        pipeline = Pipeline()
        pipeline.connect(source.element, sink)
        pipeline.run()

    def test_validation_t0_before_end(self):
        """t0 must be before end."""
        with pytest.raises(ValueError, match="t0 must be less than end"):
            GWDataNoiseComposedSource(
                name="test",
                ifos=["H1"],
                t0=1010,
                end=1000,
            )

    def test_custom_channel_pattern(self):
        """Test custom channel pattern."""
        source = GWDataNoiseComposedSource(
            name="test",
            ifos=["H1"],
            t0=1000,
            end=1010,
            channel_pattern="{ifo}:GDS-CALIB_STRAIN",
        )
        assert "H1:GDS-CALIB_STRAIN" in source.srcs

    def test_state_vector_validation(self):
        """Test state_vector_on_dict keys must match ifos."""
        with pytest.raises(
            ValueError, match="state_vector_on_dict keys must match ifos"
        ):
            GWDataNoiseComposedSource(
                name="test",
                ifos=["H1", "L1"],
                t0=1000,
                end=1010,
                state_vector_on_dict={"H1": 3},  # Missing L1
            )


class TestGWDataNoiseRealtimeComposedSource:
    """Tests for GWDataNoiseRealtimeComposedSource."""

    def test_instantiation(self):
        """Test basic instantiation."""
        source = GWDataNoiseRealtimeComposedSource(
            name="test",
            ifos=["H1"],
        )
        assert source.name == "test"
        assert source.ifos == ["H1"]

    def test_class_metadata(self):
        """Test source_type and description."""
        assert GWDataNoiseRealtimeComposedSource.source_type == "gwdata-noise-realtime"
        assert "Real-time" in GWDataNoiseRealtimeComposedSource.description

    def test_optional_time_range(self):
        """Test that t0 and end are optional."""
        source = GWDataNoiseRealtimeComposedSource(
            name="test",
            ifos=["H1"],
        )
        assert source.t0 is None
        assert source.end is None

    def test_element_property(self):
        """Test .element returns TSComposedSourceElement."""
        source = GWDataNoiseRealtimeComposedSource(
            name="test",
            ifos=["H1"],
        )
        assert isinstance(source.element, TSComposedSourceElement)


# --- Tests for FramesComposedSource ---


class TestFramesComposedSource:
    """Tests for FramesComposedSource."""

    def test_inherits_from_composed_base(self):
        """FramesComposedSource should inherit from ComposedSourceBase."""
        assert issubclass(FramesComposedSource, ComposedSourceBase)

    def test_class_metadata(self):
        """Test source_type and description."""
        assert FramesComposedSource.source_type == "frames"
        assert FramesComposedSource.description == "Read from GWF frame files"

    def test_validation_missing_frame_cache(self, tmp_path):
        """Must provide existing frame_cache."""
        with pytest.raises(ValueError, match="Frame cache file does not exist"):
            FramesComposedSource(
                name="test",
                ifos=["H1"],
                frame_cache="/nonexistent/path.cache",
                channel_dict={"H1": "GDS-CALIB_STRAIN"},
                t0=1000,
                end=1010,
            )

    def test_validation_channel_dict_keys(self, tmp_path):
        """channel_dict keys must match ifos."""
        cache_file = tmp_path / "test.cache"
        cache_file.write_text("")
        with pytest.raises(ValueError, match="channel_dict keys must match ifos"):
            FramesComposedSource(
                name="test",
                ifos=["H1", "L1"],
                frame_cache=str(cache_file),
                channel_dict={"H1": "GDS-CALIB_STRAIN"},  # Missing L1
                t0=1000,
                end=1010,
            )

    def test_validation_t0_before_end(self, tmp_path):
        """t0 must be before end."""
        cache_file = tmp_path / "test.cache"
        cache_file.write_text("")
        with pytest.raises(ValueError, match="t0 must be less than end"):
            FramesComposedSource(
                name="test",
                ifos=["H1"],
                frame_cache=str(cache_file),
                channel_dict={"H1": "GDS-CALIB_STRAIN"},
                t0=1010,
                end=1000,
            )

    def test_validation_segments_name_required(self, tmp_path):
        """Must specify segments_name when segments_file is set."""
        cache_file = tmp_path / "test.cache"
        cache_file.write_text("")
        segments_file = tmp_path / "segments.xml"
        segments_file.write_text("")
        with pytest.raises(ValueError, match="Must specify segments_name"):
            FramesComposedSource(
                name="test",
                ifos=["H1"],
                frame_cache=str(cache_file),
                channel_dict={"H1": "GDS-CALIB_STRAIN"},
                t0=1000,
                end=1010,
                segments_file=str(segments_file),
            )

    def test_validation_injection_channel_required(self, tmp_path):
        """Must specify noiseless_inj_channel_dict when injection cache is set."""
        cache_file = tmp_path / "test.cache"
        cache_file.write_text("")
        inj_cache_file = tmp_path / "inj.cache"
        inj_cache_file.write_text("")
        with pytest.raises(ValueError, match="Must specify noiseless_inj_channel_dict"):
            FramesComposedSource(
                name="test",
                ifos=["H1"],
                frame_cache=str(cache_file),
                channel_dict={"H1": "GDS-CALIB_STRAIN"},
                t0=1000,
                end=1010,
                noiseless_inj_frame_cache=str(inj_cache_file),
            )


# --- Tests for DevShmComposedSource ---


class TestDevShmComposedSource:
    """Tests for DevShmComposedSource."""

    def test_inherits_from_composed_base(self):
        """DevShmComposedSource should inherit from ComposedSourceBase."""
        assert issubclass(DevShmComposedSource, ComposedSourceBase)

    def test_class_metadata(self):
        """Test source_type and description."""
        assert DevShmComposedSource.source_type == "devshm"
        assert DevShmComposedSource.description == "Read from shared memory"

    def test_validation_channel_dict_keys(self):
        """channel_dict keys must match ifos."""
        with pytest.raises(ValueError, match="channel_dict keys must match ifos"):
            DevShmComposedSource(
                name="test",
                ifos=["H1", "L1"],
                channel_dict={"H1": "GDS-CALIB_STRAIN"},  # Missing L1
                shared_memory_dict={
                    "H1": "/dev/shm/H1",  # noqa: S108
                    "L1": "/dev/shm/L1",  # noqa: S108
                },
                state_channel_dict={"H1": "STATE", "L1": "STATE"},
                state_vector_on_dict={"H1": 3, "L1": 3},
            )

    def test_validation_shared_memory_dict_keys(self):
        """shared_memory_dict keys must match ifos."""
        with pytest.raises(ValueError, match="shared_memory_dict keys must match ifos"):
            DevShmComposedSource(
                name="test",
                ifos=["H1", "L1"],
                channel_dict={"H1": "STRAIN", "L1": "STRAIN"},
                shared_memory_dict={"H1": "/dev/shm/H1"},  # noqa: S108; Missing L1
                state_channel_dict={"H1": "STATE", "L1": "STATE"},
                state_vector_on_dict={"H1": 3, "L1": 3},
            )

    def test_validation_state_channel_dict_keys(self):
        """state_channel_dict keys must match ifos."""
        with pytest.raises(ValueError, match="state_channel_dict keys must match ifos"):
            DevShmComposedSource(
                name="test",
                ifos=["H1", "L1"],
                channel_dict={"H1": "STRAIN", "L1": "STRAIN"},
                shared_memory_dict={
                    "H1": "/dev/shm/H1",  # noqa: S108
                    "L1": "/dev/shm/L1",  # noqa: S108
                },
                state_channel_dict={"H1": "STATE"},  # Missing L1
                state_vector_on_dict={"H1": 3, "L1": 3},
            )

    def test_validation_state_vector_on_dict_keys(self):
        """state_vector_on_dict keys must match ifos."""
        with pytest.raises(
            ValueError, match="state_vector_on_dict keys must match ifos"
        ):
            DevShmComposedSource(
                name="test",
                ifos=["H1", "L1"],
                channel_dict={"H1": "STRAIN", "L1": "STRAIN"},
                shared_memory_dict={
                    "H1": "/dev/shm/H1",  # noqa: S108
                    "L1": "/dev/shm/L1",  # noqa: S108
                },
                state_channel_dict={"H1": "STATE", "L1": "STATE"},
                state_vector_on_dict={"H1": 3},  # Missing L1
            )


# --- Tests for ArrakisComposedSource ---


@pytest.mark.skipif(not HAS_ARRAKIS, reason="sgn_arrakis not available")
class TestArrakisComposedSource:
    """Tests for ArrakisComposedSource."""

    def test_inherits_from_composed_base(self):
        """ArrakisComposedSource should inherit from ComposedSourceBase."""
        assert issubclass(ArrakisComposedSource, ComposedSourceBase)

    def test_class_metadata(self):
        """Test source_type and description."""
        assert ArrakisComposedSource.source_type == "arrakis"
        assert ArrakisComposedSource.description == "Read from Arrakis"

    def test_validation_channel_dict_keys(self):
        """channel_dict keys must match ifos."""
        with pytest.raises(ValueError, match="channel_dict keys must match ifos"):
            ArrakisComposedSource(
                name="test",
                ifos=["H1", "L1"],
                channel_dict={"H1": "GDS-CALIB_STRAIN"},  # Missing L1
            )

    def test_validation_t0_before_end(self):
        """t0 must be before end."""
        with pytest.raises(ValueError, match="t0 must be less than end"):
            ArrakisComposedSource(
                name="test",
                ifos=["H1"],
                channel_dict={"H1": "GDS-CALIB_STRAIN"},
                t0=1010,
                end=1000,
            )

    def test_validation_state_channel_without_bitmask(self):
        """Must specify state_vector_on_dict when state_channel_dict is set."""
        with pytest.raises(ValueError, match="Must specify state_vector_on_dict"):
            ArrakisComposedSource(
                name="test",
                ifos=["H1"],
                channel_dict={"H1": "GDS-CALIB_STRAIN"},
                state_channel_dict={"H1": "STATE"},
            )

    def test_validation_bitmask_without_state_channel(self):
        """Must specify state_channel_dict when state_vector_on_dict is set."""
        with pytest.raises(ValueError, match="Must specify state_channel_dict"):
            ArrakisComposedSource(
                name="test",
                ifos=["H1"],
                channel_dict={"H1": "GDS-CALIB_STRAIN"},
                state_vector_on_dict={"H1": 3},
            )

    @pytest.mark.xfail(reason="ArrakisSource needs threading context")
    def test_optional_time_range(self):
        """Test that t0 and end are optional."""
        source = ArrakisComposedSource(
            name="test",
            ifos=["H1"],
            channel_dict={"H1": "GDS-CALIB_STRAIN"},
        )
        assert source.t0 is None
        assert source.end is None

    @pytest.mark.xfail(reason="ArrakisSource needs threading context")
    def test_element_property(self):
        """Test .element returns TSComposedSourceElement."""
        source = ArrakisComposedSource(
            name="test",
            ifos=["H1"],
            channel_dict={"H1": "GDS-CALIB_STRAIN"},
        )
        assert isinstance(source.element, TSComposedSourceElement)


# =============================================================================
# Tests for DataSource uber element and CLI support
# =============================================================================


class TestComposedRegistry:
    """Tests for the composed source registry."""

    def test_list_composed_source_types(self):
        """Test listing all registered source types."""
        from sgnligo.sources.datasource_v2 import list_composed_source_types

        source_types = list_composed_source_types()
        assert isinstance(source_types, list)
        # Check some expected source types
        assert "white" in source_types
        assert "sin" in source_types
        assert "impulse" in source_types
        assert "gwdata-noise" in source_types
        assert "frames" in source_types
        assert "devshm" in source_types
        assert "white-realtime" in source_types

    def test_get_composed_source_class(self):
        """Test getting a source class by type."""
        from sgnligo.sources.datasource_v2 import get_composed_source_class

        cls = get_composed_source_class("white")
        assert cls is WhiteComposedSource

        cls = get_composed_source_class("gwdata-noise")
        assert cls is GWDataNoiseComposedSource

    def test_get_composed_source_class_unknown(self):
        """Test getting unknown source type raises ValueError."""
        from sgnligo.sources.datasource_v2 import get_composed_source_class

        with pytest.raises(ValueError, match="Unknown source type"):
            get_composed_source_class("nonexistent-source")

    def test_get_composed_registry(self):
        """Test getting the full registry."""
        from sgnligo.sources.datasource_v2 import get_composed_registry

        registry = get_composed_registry()
        assert isinstance(registry, dict)
        assert "white" in registry
        assert registry["white"] is WhiteComposedSource


class TestDataSource:
    """Tests for the DataSource uber element."""

    def test_instantiation_with_white(self):
        """Test DataSource dispatches to WhiteComposedSource."""
        from sgnligo.sources.datasource_v2 import DataSource

        source = DataSource(
            data_source="white",
            name="test",
            ifos=["H1"],
            sample_rate=4096,
            t0=1000,
            end=1010,
        )
        assert source.data_source == "white"
        assert source.name == "test"
        assert isinstance(source.element, TSComposedSourceElement)

    def test_instantiation_with_sin(self):
        """Test DataSource dispatches to SinComposedSource."""
        from sgnligo.sources.datasource_v2 import DataSource

        source = DataSource(
            data_source="sin",
            name="sine_test",
            ifos=["H1", "L1"],
            sample_rate=4096,
            t0=1000,
            end=1010,
        )
        assert source.data_source == "sin"
        assert "H1" in source.srcs
        assert "L1" in source.srcs

    def test_instantiation_with_gwdata_noise(self):
        """Test DataSource dispatches to GWDataNoiseComposedSource."""
        from sgnligo.sources.datasource_v2 import DataSource

        source = DataSource(
            data_source="gwdata-noise",
            name="noise_test",
            ifos=["H1"],
            t0=1000,
            end=1010,
        )
        assert source.data_source == "gwdata-noise"
        assert isinstance(source.element, TSComposedSourceElement)

    def test_unknown_source_type(self):
        """Test unknown source type raises ValueError."""
        from sgnligo.sources.datasource_v2 import DataSource

        with pytest.raises(ValueError, match="Unknown source type"):
            DataSource(
                data_source="nonexistent-source",
                name="test",
                ifos=["H1"],
            )

    def test_element_property_delegation(self):
        """Test element property delegates to inner source."""
        from sgnligo.sources.datasource_v2 import DataSource

        source = DataSource(
            data_source="white",
            name="test",
            ifos=["H1"],
            sample_rate=4096,
            t0=1000,
            end=1010,
        )
        # element property should be the same as inner element
        assert source.element is source._inner.element

    def test_srcs_property_delegation(self):
        """Test srcs property delegates to inner source."""
        from sgnligo.sources.datasource_v2 import DataSource

        source = DataSource(
            data_source="white",
            name="test",
            ifos=["H1", "L1"],
            sample_rate=4096,
            t0=1000,
            end=1010,
        )
        assert "H1" in source.srcs
        assert "L1" in source.srcs

    def test_pipeline_integration(self):
        """Test DataSource works in pipeline."""
        from sgnligo.sources.datasource_v2 import DataSource

        source = DataSource(
            data_source="white",
            name="test",
            ifos=["H1"],
            sample_rate=4096,
            t0=1000,
            end=1001,
        )
        sink = NullSink(name="sink", sink_pad_names=["H1"])
        pipeline = Pipeline()
        pipeline.connect(source.element, sink)
        pipeline.run()

    def test_list_sources(self):
        """Test list_sources static method."""
        from sgnligo.sources.datasource_v2 import DataSource

        source_types = DataSource.list_sources()
        assert "white" in source_types
        assert "gwdata-noise" in source_types

    def test_get_source_class(self):
        """Test get_source_class static method."""
        from sgnligo.sources.datasource_v2 import DataSource

        cls = DataSource.get_source_class("white")
        assert cls is WhiteComposedSource

    def test_from_argv(self):
        """Test DataSource.from_argv creates source from argv."""
        from sgnligo.sources.datasource_v2 import DataSource

        argv = [
            "--data-source",
            "white",
            "--ifos",
            "H1",
            "--gps-start-time",
            "1000",
            "--gps-end-time",
            "1010",
            "--sample-rate",
            "4096",
        ]
        source = DataSource.from_argv(name="test", argv=argv)
        assert source.data_source == "white"
        assert isinstance(source.element, TSComposedSourceElement)

    def test_from_parser(self):
        """Test DataSource.from_parser with custom arguments."""
        from sgnligo.sources.datasource_v2 import DataSource

        parser = DataSource.create_cli_parser()
        parser.add_argument("--snr-threshold", type=float, default=8.0)

        argv = [
            "--data-source",
            "white",
            "--ifos",
            "H1",
            "--gps-start-time",
            "1000",
            "--gps-end-time",
            "1010",
            "--sample-rate",
            "4096",
            "--snr-threshold",
            "10.0",
        ]
        source, args = DataSource.from_parser(parser, name="test", argv=argv)
        assert source.data_source == "white"
        assert args.snr_threshold == 10.0
        assert isinstance(source.element, TSComposedSourceElement)

    def test_from_parser_returns_tuple(self):
        """Test from_parser returns (source, args) tuple."""
        from sgnligo.sources.datasource_v2 import DataSource

        parser = DataSource.create_cli_parser()
        argv = [
            "--data-source",
            "white",
            "--ifos",
            "H1",
            "--gps-start-time",
            "1000",
            "--gps-end-time",
            "1010",
            "--sample-rate",
            "4096",
        ]
        result = DataSource.from_parser(parser, name="test", argv=argv)
        assert isinstance(result, tuple)
        assert len(result) == 2
        source, args = result
        assert isinstance(source, DataSource)
        assert hasattr(args, "data_source")


class TestCLISupport:
    """Tests for CLI support functions."""

    def test_build_composed_cli_parser(self):
        """Test building CLI parser."""
        from sgnligo.sources.datasource_v2.cli import build_composed_cli_parser

        parser = build_composed_cli_parser()
        # Check parser has required options
        args = parser.parse_args(["--data-source", "white"])
        assert args.data_source == "white"

    def test_check_composed_help_options_list_sources(self, capsys):
        """Test --list-sources option."""
        from sgnligo.sources.datasource_v2.cli import check_composed_help_options

        result = check_composed_help_options(["--list-sources"])
        assert result is True

        captured = capsys.readouterr()
        assert "Available data sources" in captured.out
        assert "white" in captured.out

    def test_check_composed_help_options_help_source(self, capsys):
        """Test --help-source option."""
        from sgnligo.sources.datasource_v2.cli import check_composed_help_options

        result = check_composed_help_options(["--help-source", "white"])
        assert result is True

        captured = capsys.readouterr()
        assert "white" in captured.out
        assert "Gaussian white noise" in captured.out

    def test_check_composed_help_options_unknown_source(self, capsys):
        """Test --help-source with unknown source."""
        from sgnligo.sources.datasource_v2.cli import check_composed_help_options

        result = check_composed_help_options(["--help-source", "nonexistent"])
        assert result is True

        captured = capsys.readouterr()
        assert "Unknown source type" in captured.out

    def test_check_composed_help_options_no_help(self):
        """Test without help options returns False."""
        from sgnligo.sources.datasource_v2.cli import check_composed_help_options

        result = check_composed_help_options(["--data-source", "white"])
        assert result is False

    def test_namespace_to_datasource_kwargs(self):
        """Test converting argparse namespace to kwargs.

        WhiteComposedSource uses IfosOnlyMixin which derives ifos from channel_name.
        Only kwargs for the source's actual mixins are returned.
        """
        import argparse

        from sgnligo.sources.datasource_v2.cli import namespace_to_datasource_kwargs

        args = argparse.Namespace(
            data_source="white",
            channel_name=["H1=STRAIN", "L1=STRAIN"],
            gps_start_time=1000.0,
            gps_end_time=1010.0,
            sample_rate=4096,
            verbose=True,
        )

        kwargs = namespace_to_datasource_kwargs(args)
        assert kwargs["data_source"] == "white"
        # ifos derived from channel_name by IfosOnlyMixin
        assert kwargs["ifos"] == ["H1", "L1"]
        # WhiteComposedSource uses IfosOnlyMixin not ChannelOptionsMixin,
        # so channel_dict is NOT included in kwargs
        assert "channel_dict" not in kwargs
        assert kwargs["t0"] == 1000.0
        assert kwargs["end"] == 1010.0
        assert kwargs["sample_rate"] == 4096
        assert kwargs["verbose"] is True

    def test_namespace_to_datasource_kwargs_creates_source(self):
        """Test creating DataSource from namespace via kwargs."""
        import argparse

        from sgnligo.sources.datasource_v2 import DataSource
        from sgnligo.sources.datasource_v2.cli import namespace_to_datasource_kwargs

        args = argparse.Namespace(
            data_source="white",
            channel_name=["H1=STRAIN"],
            gps_start_time=1000.0,
            gps_end_time=1010.0,
            sample_rate=4096,
            verbose=False,
            frame_cache=None,
            segments_file=None,
            segments_name=None,
            noiseless_inj_frame_cache=None,
            noiseless_inj_channel_name=None,
            state_channel_name=None,
            state_vector_on_bits=None,
            state_segments_file=None,
            state_sample_rate=16,
            shared_memory_dir=None,
            discont_wait_time=60.0,
            queue_timeout=1.0,
            impulse_position=-1,
        )

        kwargs = namespace_to_datasource_kwargs(args)
        source = DataSource(name="cli_test", **kwargs)
        assert source.data_source == "white"
        assert source.name == "cli_test"
        assert isinstance(source.element, TSComposedSourceElement)

    def test_create_cli_parser_classmethod(self):
        """Test DataSource.create_cli_parser classmethod."""
        from sgnligo.sources.datasource_v2 import DataSource

        parser = DataSource.create_cli_parser()
        args = parser.parse_args(["--data-source", "sin"])
        assert args.data_source == "sin"

    def test_check_composed_help_options_missing_source_arg(self, capsys):
        """Test --help-source without source type argument."""
        from sgnligo.sources.datasource_v2.cli import check_composed_help_options

        result = check_composed_help_options(["--help-source"])
        assert result is True

        captured = capsys.readouterr()
        assert "requires a source type argument" in captured.out

    def test_check_composed_help_options_default_argv(self, monkeypatch):
        """Test check_composed_help_options uses sys.argv[1:] when argv is None."""
        import sys

        from sgnligo.sources.datasource_v2.cli import check_composed_help_options

        # Set sys.argv to simulate no special help options
        monkeypatch.setattr(sys, "argv", ["prog", "--data-source", "white"])
        result = check_composed_help_options()  # argv=None, should use sys.argv[1:]
        assert result is False

    def test_namespace_to_datasource_kwargs_with_all_options(self):
        """Test converting namespace with options for frames source.

        Only checks for args that FramesComposedSource actually uses.
        Each source is self-describing via its mixins - args not handled
        by a source's mixins are ignored.
        """
        import argparse

        from sgnligo.sources.datasource_v2.cli import namespace_to_datasource_kwargs

        args = argparse.Namespace(
            data_source="frames",
            channel_name=["H1=STRAIN"],
            gps_start_time=1000.0,
            gps_end_time=1010.0,
            verbose=True,
            frame_cache="/path/to/frames.cache",
            segments_file="/path/to/segments.xml",
            segments_name="SCIENCE",
            noiseless_inj_frame_cache="/path/to/inj.cache",
            noiseless_inj_channel_name=["H1=INJ-STRAIN"],
        )

        kwargs = namespace_to_datasource_kwargs(args)

        # Verify args that FramesComposedSource uses (via its mixins)
        assert kwargs["frame_cache"] == "/path/to/frames.cache"
        assert kwargs["segments_file"] == "/path/to/segments.xml"
        assert kwargs["segments_name"] == "SCIENCE"
        assert kwargs["noiseless_inj_frame_cache"] == "/path/to/inj.cache"
        assert kwargs["noiseless_inj_channel_dict"] == {"H1": "INJ-STRAIN"}
        assert kwargs["channel_dict"] == {"H1": "STRAIN"}
        assert kwargs["ifos"] == ["H1"]
        assert kwargs["t0"] == 1000.0
        assert kwargs["end"] == 1010.0
        assert kwargs["verbose"] is True

    def test_get_source_required_fields(self):
        """Test get_source_required_fields returns required fields."""
        from sgnligo.sources.datasource_v2.cli import get_source_required_fields

        required = get_source_required_fields(WhiteComposedSource)
        assert "ifos" in required
        assert "sample_rate" in required
        assert "t0" in required
        assert "end" in required
        # Optional fields should not be in required
        assert "verbose" not in required

    def test_get_source_optional_fields(self):
        """Test get_source_optional_fields returns optional fields."""
        from sgnligo.sources.datasource_v2.cli import get_source_optional_fields

        optional = get_source_optional_fields(WhiteComposedSource)
        assert "verbose" in optional
        assert optional["verbose"] is False

    def test_format_composed_source_help(self):
        """Test format_composed_source_help generates help text."""
        from sgnligo.sources.datasource_v2.cli import format_composed_source_help

        help_text = format_composed_source_help("white")
        assert "white" in help_text
        assert "Gaussian white noise" in help_text
        assert "Required Options" in help_text
        assert "Optional Options" in help_text

    def test_format_composed_source_list(self):
        """Test format_composed_source_list generates list text."""
        from sgnligo.sources.datasource_v2.cli import format_composed_source_list

        list_text = format_composed_source_list()
        assert "Available data sources" in list_text
        assert "Offline Sources" in list_text
        assert "Real-time Sources" in list_text
        assert "white" in list_text


# =============================================================================
# Tests for composed registry edge cases
# =============================================================================


class TestComposedRegistryEdgeCases:
    """Test edge cases for composed registry."""

    def test_register_source_with_empty_source_type(self):
        """Test registering a source with empty source_type raises ValueError."""
        from dataclasses import dataclass

        from sgnligo.sources.composed_base import ComposedSourceBase
        from sgnligo.sources.datasource_v2.composed_registry import (
            register_composed_source,
        )

        # Create a class with empty source_type
        @dataclass
        class BadSource(ComposedSourceBase):
            source_type = ""
            description = "Bad source"

            def _validate(self):
                pass

            def _build(self):
                pass

        with pytest.raises(ValueError, match="must define source_type"):
            register_composed_source(BadSource)

    def test_register_duplicate_source_type(self):
        """Test registering a duplicate source_type raises ValueError."""
        from dataclasses import dataclass

        from sgnligo.sources.composed_base import ComposedSourceBase
        from sgnligo.sources.datasource_v2.composed_registry import (
            register_composed_source,
        )

        # Try to register a source with the same type as WhiteComposedSource
        @dataclass
        class DuplicateWhite(ComposedSourceBase):
            source_type = "white"  # Already registered!
            description = "Duplicate"

            def _validate(self):
                pass

            def _build(self):
                pass

        with pytest.raises(ValueError, match="already registered"):
            register_composed_source(DuplicateWhite)


# =============================================================================
# Tests for DataSource attribute delegation
# =============================================================================


class TestDataSourceDelegation:
    """Test DataSource attribute delegation."""

    def test_getattr_delegates_to_inner(self):
        """Test unknown attributes are delegated to inner source."""
        from sgnligo.sources.datasource_v2 import DataSource

        source = DataSource(
            data_source="white",
            name="test",
            ifos=["H1"],
            sample_rate=4096,
            t0=1000,
            end=1010,
        )
        # Access inner source attributes via delegation
        assert source.ifos == ["H1"]
        assert source.sample_rate == 4096

    def test_getattr_raises_for_private_attrs(self):
        """Test private attributes raise AttributeError."""
        from sgnligo.sources.datasource_v2 import DataSource

        source = DataSource(
            data_source="white",
            name="test",
            ifos=["H1"],
            sample_rate=4096,
            t0=1000,
            end=1010,
        )
        with pytest.raises(AttributeError):
            _ = source._nonexistent_private


# =============================================================================
# Tests for GWDataNoiseComposedSource state vector gating
# =============================================================================


class TestGWDataNoiseWithoutStateGating:
    """Tests for GWDataNoiseComposedSource without state vector gating."""

    def test_offline_without_state_gating(self):
        """Test GWDataNoiseComposedSource without state vector gating."""
        source = GWDataNoiseComposedSource(
            name="test",
            ifos=["H1"],
            t0=1000,
            end=1002,
        )
        # Without state vector gating, output pads use channel names
        # Default channel pattern is {ifo}:FAKE-STRAIN
        assert "H1:FAKE-STRAIN" in source.srcs
        # Run pipeline to exercise _build
        sink = NullSink(name="sink", sink_pad_names=["H1:FAKE-STRAIN"])
        pipeline = Pipeline()
        pipeline.connect(source.element, sink)
        pipeline.run()

    def test_realtime_without_state_gating(self):
        """Test GWDataNoiseRealtimeComposedSource without state vector gating."""
        source = GWDataNoiseRealtimeComposedSource(
            name="test",
            ifos=["H1"],
            t0=1000,
            end=1002,
        )
        # Without state vector gating, output pads use channel names
        # Default channel pattern is {ifo}:FAKE-STRAIN
        assert "H1:FAKE-STRAIN" in source.srcs

    def test_offline_load_state_segments_returns_none_when_no_state_dict(self):
        """Test _load_state_segments returns None when state_vector_on_dict is None.

        This tests the defensive early-return code path that is normally not reached
        since _load_state_segments is only called when state_vector_on_dict is set.
        """
        source = GWDataNoiseComposedSource(
            name="test",
            ifos=["H1"],
            t0=1000,
            end=1002,
        )
        # Call _load_state_segments directly with state_vector_on_dict=None
        segments, values = source._load_state_segments()
        assert segments is None
        assert values is None

    def test_realtime_load_state_segments_returns_none_when_no_state_dict(self):
        """Test _load_state_segments returns None when state_vector_on_dict is None.

        This tests the defensive early-return code path that is normally not reached
        since _load_state_segments is only called when state_vector_on_dict is set.
        """
        source = GWDataNoiseRealtimeComposedSource(
            name="test",
            ifos=["H1"],
            t0=1000,
            end=1002,
        )
        # Call _load_state_segments directly with state_vector_on_dict=None
        segments, values = source._load_state_segments()
        assert segments is None
        assert values is None


class TestGWDataNoiseStateVectorGating:
    """Tests for GWDataNoiseComposedSource with state vector gating."""

    def test_state_vector_gating_with_defaults(self):
        """Test state vector gating using default segments."""
        source = GWDataNoiseComposedSource(
            name="test",
            ifos=["H1"],
            t0=1000,
            end=1002,
            state_vector_on_dict={"H1": 3},
        )
        # With state vector gating, output pads use IFO names
        assert "H1" in source.srcs

    def test_state_vector_gating_verbose(self, capsys):
        """Test verbose output with state vector gating."""
        GWDataNoiseComposedSource(
            name="test",
            ifos=["H1"],
            t0=1000,
            end=1002,
            state_vector_on_dict={"H1": 3},
            verbose=True,
        )
        captured = capsys.readouterr()
        assert "Added state vector gating for H1" in captured.out

    def test_state_vector_gating_pipeline(self):
        """Test running pipeline with state vector gating."""
        source = GWDataNoiseComposedSource(
            name="test",
            ifos=["H1"],
            t0=1000,
            end=1001,
            state_vector_on_dict={"H1": 3},
        )
        sink = NullSink(name="sink", sink_pad_names=["H1"])
        pipeline = Pipeline()
        pipeline.connect(source.element, sink)
        pipeline.run()


class TestGWDataNoiseRealtimeStateVectorGating:
    """Tests for GWDataNoiseRealtimeComposedSource with state vector gating."""

    def test_state_vector_gating_with_t0(self):
        """Test state vector gating with t0 provided."""
        source = GWDataNoiseRealtimeComposedSource(
            name="test",
            ifos=["H1"],
            t0=1000,
            state_vector_on_dict={"H1": 3},
        )
        assert "H1" in source.srcs

    def test_state_vector_gating_with_t0_and_end(self):
        """Test state vector gating with t0 and end provided."""
        source = GWDataNoiseRealtimeComposedSource(
            name="test",
            ifos=["H1"],
            t0=1000,
            end=1100,
            state_vector_on_dict={"H1": 3},
        )
        assert "H1" in source.srcs

    def test_state_vector_gating_verbose(self, capsys):
        """Test verbose output with state vector gating."""
        GWDataNoiseRealtimeComposedSource(
            name="test",
            ifos=["H1"],
            t0=1000,
            state_vector_on_dict={"H1": 3},
            verbose=True,
        )
        captured = capsys.readouterr()
        assert "Added state vector gating for H1" in captured.out

    def test_state_vector_gating_requires_t0(self):
        """Test state vector gating without segments_file requires t0."""
        with pytest.raises(
            ValueError, match="Must provide either state_segments_file or t0"
        ):
            source = GWDataNoiseRealtimeComposedSource(
                name="test",
                ifos=["H1"],
                state_vector_on_dict={"H1": 3},
            )
            # Access element to trigger build
            _ = source.element

    def test_validation_state_segments_file_not_found(self, tmp_path):
        """Test validation fails for non-existent state segments file."""
        with pytest.raises(ValueError, match="State segments file does not exist"):
            GWDataNoiseRealtimeComposedSource(
                name="test",
                ifos=["H1"],
                state_segments_file="/nonexistent/file.txt",
            )

    def test_validation_state_vector_on_dict_keys(self):
        """Test state_vector_on_dict keys must match ifos."""
        with pytest.raises(
            ValueError, match="state_vector_on_dict keys must match ifos"
        ):
            GWDataNoiseRealtimeComposedSource(
                name="test",
                ifos=["H1", "L1"],
                t0=1000,
                state_vector_on_dict={"H1": 3},  # Missing L1
            )

    def test_validation_t0_before_end(self):
        """t0 must be before end."""
        with pytest.raises(ValueError, match="t0 must be less than end"):
            GWDataNoiseRealtimeComposedSource(
                name="test",
                ifos=["H1"],
                t0=1010,
                end=1000,
            )


class TestGWDataNoiseStateSegmentsFile:
    """Tests for state segments file loading."""

    def test_state_vector_with_segments_file(self, tmp_path):
        """Test state vector gating with segments file."""
        # Create a simple segments file
        segments_file = tmp_path / "segments.txt"
        segments_file.write_text("1000000000000 1001000000000 3\n")

        source = GWDataNoiseComposedSource(
            name="test",
            ifos=["H1"],
            t0=1000,
            end=1002,
            state_segments_file=str(segments_file),
            state_vector_on_dict={"H1": 3},
            verbose=True,
        )
        assert "H1" in source.srcs

    def test_state_segments_file_not_found(self, tmp_path):
        """Test validation fails for non-existent state segments file."""
        with pytest.raises(ValueError, match="State segments file does not exist"):
            GWDataNoiseComposedSource(
                name="test",
                ifos=["H1"],
                t0=1000,
                end=1010,
                state_segments_file="/nonexistent/file.txt",
            )


# =============================================================================
# Tests for FakeSource segment gating
# =============================================================================


class TestFakeSourceSegmentGating:
    """Tests for FakeSourceBase segment gating."""

    def test_segments_name_required_with_segments_file(self, tmp_path):
        """Must specify segments_name when segments_file is set."""
        segments_file = tmp_path / "segments.xml"
        segments_file.write_text("")

        with pytest.raises(ValueError, match="Must specify segments_name"):
            WhiteComposedSource(
                name="test",
                ifos=["H1"],
                sample_rate=4096,
                t0=1000,
                end=1010,
                segments_file=str(segments_file),
            )

    def test_segments_file_not_found(self, tmp_path):
        """Test validation fails for non-existent segments file."""
        with pytest.raises(ValueError, match="Segments file does not exist"):
            WhiteComposedSource(
                name="test",
                ifos=["H1"],
                sample_rate=4096,
                t0=1000,
                end=1010,
                segments_file="/nonexistent/file.xml",
                segments_name="SCIENCE",
            )


class TestRealtimeFakeSourceValidation:
    """Tests for RealtimeFakeSourceBase validation."""

    def test_sample_rate_must_be_positive(self):
        """sample_rate must be positive."""
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            WhiteRealtimeComposedSource(
                name="test",
                ifos=["H1"],
                sample_rate=0,
            )


# =============================================================================
# Tests for Arrakis state vector gating
# =============================================================================


@pytest.mark.skipif(not HAS_ARRAKIS, reason="sgn_arrakis not available")
class TestArrakisStateVectorGating:
    """Tests for ArrakisComposedSource state vector gating."""

    def test_state_channel_dict_keys_must_match_ifos(self):
        """state_channel_dict keys must match ifos."""
        with pytest.raises(ValueError, match="state_channel_dict keys must match ifos"):
            ArrakisComposedSource(
                name="test",
                ifos=["H1", "L1"],
                channel_dict={"H1": "STRAIN", "L1": "STRAIN"},
                state_channel_dict={"H1": "STATE"},  # Missing L1
                state_vector_on_dict={"H1": 3, "L1": 3},
            )

    def test_state_vector_on_dict_keys_must_match_ifos(self):
        """state_vector_on_dict keys must match ifos."""
        with pytest.raises(
            ValueError, match="state_vector_on_dict keys must match ifos"
        ):
            ArrakisComposedSource(
                name="test",
                ifos=["H1", "L1"],
                channel_dict={"H1": "STRAIN", "L1": "STRAIN"},
                state_channel_dict={"H1": "STATE", "L1": "STATE"},
                state_vector_on_dict={"H1": 3},  # Missing L1
            )


# =============================================================================
# Tests for integrated latency tracking
# =============================================================================


class TestLatencyIntegration:
    """Tests for integrated latency tracking in composed sources."""

    def test_latency_pads_not_present_by_default(self):
        """Test that latency pads are NOT present when latency_interval is None."""
        source = WhiteComposedSource(
            name="src",
            ifos=["H1"],
            sample_rate=4096,
            t0=1000,
            end=1001,
        )

        # No latency pads should exist
        assert "H1" in source.srcs
        assert "H1_latency" not in source.srcs

    def test_latency_pads_present_when_enabled(self):
        """Test that latency pads are present when latency_interval is set."""
        source = WhiteComposedSource(
            name="src",
            ifos=["H1"],
            sample_rate=4096,
            t0=1000,
            end=1001,
            latency_interval=1.0,
        )

        # Both strain and latency pads should exist
        assert "H1" in source.srcs
        assert "H1_latency" in source.srcs

    def test_latency_pads_multi_ifo(self):
        """Test latency pads for multiple IFOs."""
        source = WhiteComposedSource(
            name="src",
            ifos=["H1", "L1"],
            sample_rate=4096,
            t0=1000,
            end=1001,
            latency_interval=0.5,
        )

        # All strain and latency pads should exist
        assert "H1" in source.srcs
        assert "H1_latency" in source.srcs
        assert "L1" in source.srcs
        assert "L1_latency" in source.srcs


# =============================================================================
# Tests for add_state_vector_gating utility function
# =============================================================================


class TestAddStateVectorGating:
    """Tests for add_state_vector_gating utility function."""

    def test_add_state_vector_gating_directly(self):
        """Test add_state_vector_gating function directly."""
        from sgnts.compose import TSCompose
        from sgnts.sources import FakeSeriesSource, SegmentSource

        from sgnligo.sources.datasource_v2.sources.utils import add_state_vector_gating

        # Create strain source
        strain_source = FakeSeriesSource(
            name="strain",
            source_pad_names=("strain_out",),
            rate=4096,
            signal_type="white",
            t0=1000,
            end=1001,
        )

        # Create state vector source
        state_source = SegmentSource(
            name="state",
            source_pad_names=("state_out",),
            rate=16,
            t0=1000,
            end=1001,
            segments=((1000000000000, 1001000000000),),
            values=(3,),
        )

        compose = TSCompose()

        # Add state vector gating
        add_state_vector_gating(
            compose=compose,
            strain_source=strain_source,
            state_source=state_source,
            ifo="H1",
            bit_mask=3,
            strain_pad="strain_out",
            state_pad="state_out",
            output_pad="H1",
        )

        # Build the composed element
        element = compose.as_source(name="test")
        assert element is not None


# =============================================================================
# Tests for CLI edge cases
# =============================================================================


class TestCLIEdgeCases:
    """Test CLI edge cases for better coverage."""

    def test_get_source_optional_fields_with_default_factory(self):
        """Test optional fields with default_factory are handled."""
        from dataclasses import dataclass, field

        from sgnligo.sources.datasource_v2.cli import get_source_optional_fields

        @dataclass
        class TestClassWithFactory:
            name: str
            items: list = field(default_factory=list)

        optional = get_source_optional_fields(TestClassWithFactory)
        assert "items" in optional
        assert optional["items"] == []  # Should call the factory

    def test_format_composed_source_help_includes_default_values(self):
        """Test help format shows default values correctly."""
        from sgnligo.sources.datasource_v2.cli import format_composed_source_help

        # GWDataNoiseComposedSource has optional fields with defaults
        help_text = format_composed_source_help("gwdata-noise")
        assert "gwdata-noise" in help_text
        # Should have some default values shown
        assert "Optional Options" in help_text

    def test_format_composed_source_help_bool_defaults(self):
        """Test help format handles boolean defaults (like verbose=False)."""
        from sgnligo.sources.datasource_v2.cli import format_composed_source_help

        help_text = format_composed_source_help("white")
        # verbose is a boolean with default False, should be displayed
        # without "(default: ...)"
        assert "--verbose" in help_text


# =============================================================================
# Additional DataSource delegation tests
# =============================================================================


class TestDataSourceAttributeDelegation:
    """Additional tests for DataSource attribute delegation."""

    def test_getattr_delegates_signal_type(self):
        """Test delegation of signal_type class attribute from inner source."""
        from sgnligo.sources.datasource_v2 import DataSource

        source = DataSource(
            data_source="white",
            name="test",
            ifos=["H1"],
            sample_rate=4096,
            t0=1000,
            end=1010,
        )
        # signal_type is a ClassVar on WhiteComposedSource, not on DataSource
        # This tests the __getattr__ delegation path
        assert source.signal_type == "white"


# =============================================================================
# Tests for gwdata_noise state segments file loading
# =============================================================================


class TestGWDataNoiseRealtimeStateSegmentsFile:
    """Tests for state segments file loading in realtime source."""

    def test_state_vector_with_segments_file(self, tmp_path):
        """Test state vector gating with segments file."""
        # Create a simple segments file
        segments_file = tmp_path / "segments.txt"
        segments_file.write_text("1000000000000 1100000000000 3\n")

        source = GWDataNoiseRealtimeComposedSource(
            name="test",
            ifos=["H1"],
            t0=1000,
            state_segments_file=str(segments_file),
            state_vector_on_dict={"H1": 3},
            verbose=True,
        )
        assert "H1" in source.srcs


# =============================================================================
# Tests for segment gating in fake sources (using mocking)
# =============================================================================


class TestFakeSourceWithSegments:
    """Tests for FakeSource segment loading paths using mocking."""

    def test_white_source_with_segments_pipeline(self, tmp_path):
        """Test WhiteComposedSource with segment gating using mocked segments."""
        from unittest.mock import MagicMock, patch

        import igwn_segments as segments
        from lal import LIGOTimeGPS

        # Create a dummy segments file (content doesn't matter, we mock the loader)
        segments_xml = tmp_path / "segments.xml"
        segments_xml.write_text("<dummy/>")

        # Create mock segment list
        mock_seglist = segments.segmentlistdict()
        mock_seglist["H1"] = segments.segmentlist(
            [segments.segment(LIGOTimeGPS(1000), LIGOTimeGPS(1002))]
        )

        mock_result = MagicMock()
        mock_result.coalesce.return_value = mock_seglist

        fake_mod = "sgnligo.sources.datasource_v2.sources.fake"
        with patch(f"{fake_mod}.ligolw_utils.load_filename"):
            with patch(
                f"{fake_mod}.ligolw_segments.segmenttable_get_by_name",
                return_value=mock_result,
            ):
                source = WhiteComposedSource(
                    name="test",
                    ifos=["H1"],
                    sample_rate=4096,
                    t0=1000,
                    end=1002,
                    segments_file=str(segments_xml),
                    segments_name="SCIENCE",
                    verbose=True,
                )

                # The source should be created successfully with segment gating
                assert "H1" in source.srcs

                # Run the pipeline
                sink = NullSink(name="sink", sink_pad_names=["H1"])
                pipeline = Pipeline()
                pipeline.connect(source.element, sink)
                pipeline.run()

    def test_white_source_with_empty_segments_for_ifo(self, tmp_path):
        """Test WhiteComposedSource when segment dict has IFO but empty list."""
        from unittest.mock import MagicMock, patch

        import igwn_segments as segments

        segments_xml = tmp_path / "segments.xml"
        segments_xml.write_text("<dummy/>")

        # Create mock segment list with empty H1 segments
        mock_seglist = segments.segmentlistdict()
        mock_seglist["H1"] = segments.segmentlist([])  # Empty!

        mock_result = MagicMock()
        mock_result.coalesce.return_value = mock_seglist

        fake_mod = "sgnligo.sources.datasource_v2.sources.fake"
        with patch(f"{fake_mod}.ligolw_utils.load_filename"):
            with patch(
                f"{fake_mod}.ligolw_segments.segmenttable_get_by_name",
                return_value=mock_result,
            ):
                source = WhiteComposedSource(
                    name="test",
                    ifos=["H1"],
                    sample_rate=4096,
                    t0=1000,
                    end=1002,
                    segments_file=str(segments_xml),
                    segments_name="SCIENCE",
                )

                # Source should still work - H1 is in segments_dict but has no segments
                assert "H1" in source.srcs

    def test_white_source_with_no_ifo_in_segments(self, tmp_path):
        """Test WhiteComposedSource when IFO is not in segments dict at all."""
        from unittest.mock import MagicMock, patch

        import igwn_segments as segments
        from lal import LIGOTimeGPS

        segments_xml = tmp_path / "segments.xml"
        segments_xml.write_text("<dummy/>")

        # Create mock segment list with only L1, not H1
        mock_seglist = segments.segmentlistdict()
        mock_seglist["L1"] = segments.segmentlist(
            [segments.segment(LIGOTimeGPS(1000), LIGOTimeGPS(1002))]
        )

        mock_result = MagicMock()
        mock_result.coalesce.return_value = mock_seglist

        fake_mod = "sgnligo.sources.datasource_v2.sources.fake"
        with patch(f"{fake_mod}.ligolw_utils.load_filename"):
            with patch(
                f"{fake_mod}.ligolw_segments.segmenttable_get_by_name",
                return_value=mock_result,
            ):
                source = WhiteComposedSource(
                    name="test",
                    ifos=["H1"],
                    sample_rate=4096,
                    t0=1000,
                    end=1002,
                    segments_file=str(segments_xml),
                    segments_name="SCIENCE",
                )

                # Source should work - H1 falls through to else branch (no gating)
                assert "H1" in source.srcs


# =============================================================================
# Tests for frames source (requires frame files - mark xfail for missing files)
# =============================================================================


class TestFramesComposedSourceBuild:
    """Tests for FramesComposedSource build method."""

    def test_validation_injection_cache_not_found(self, tmp_path):
        """Test validation fails for non-existent injection cache."""
        cache_file = tmp_path / "test.cache"
        cache_file.write_text("")

        with pytest.raises(ValueError, match="Injection frame cache does not exist"):
            FramesComposedSource(
                name="test",
                ifos=["H1"],
                frame_cache=str(cache_file),
                channel_dict={"H1": "STRAIN"},
                t0=1000,
                end=1010,
                noiseless_inj_frame_cache="/nonexistent/inj.cache",
                noiseless_inj_channel_dict={"H1": "INJ-STRAIN"},
            )

    def test_validation_segments_file_not_found(self, tmp_path):
        """Test validation fails for non-existent segments file."""
        cache_file = tmp_path / "test.cache"
        cache_file.write_text("")

        with pytest.raises(ValueError, match="Segments file does not exist"):
            FramesComposedSource(
                name="test",
                ifos=["H1"],
                frame_cache=str(cache_file),
                channel_dict={"H1": "STRAIN"},
                t0=1000,
                end=1010,
                segments_file="/nonexistent/segments.xml",
                segments_name="SCIENCE",
            )

    def test_frames_build_basic(self, tmp_path):
        """Test FramesComposedSource build without injection or segments (mocked)."""
        from unittest.mock import MagicMock, patch

        cache_file = tmp_path / "test.cache"
        cache_file.write_text("")

        mock_frame_reader = MagicMock()
        mock_frame_reader.rates = {"H1:STRAIN": 16384}

        mock_element = MagicMock()

        frames_mod = "sgnligo.sources.datasource_v2.sources.frames"
        with patch(f"{frames_mod}.FrameReader", return_value=mock_frame_reader):
            with patch(f"{frames_mod}.TSCompose") as mock_compose_cls:
                mock_compose = MagicMock()
                mock_compose.as_source.return_value = mock_element
                mock_compose_cls.return_value = mock_compose

                source = FramesComposedSource(
                    name="test",
                    ifos=["H1"],
                    frame_cache=str(cache_file),
                    channel_dict={"H1": "STRAIN"},
                    t0=1000,
                    end=1010,
                )
                result = source.element
                assert result is mock_element
                mock_compose.insert.assert_called_once_with(mock_frame_reader)

    def test_frames_build_with_injection(self, tmp_path, capsys):
        """Test FramesComposedSource build with injection (mocked)."""
        from unittest.mock import MagicMock, patch

        cache_file = tmp_path / "test.cache"
        cache_file.write_text("")
        inj_cache_file = tmp_path / "inj.cache"
        inj_cache_file.write_text("")

        mock_frame_reader = MagicMock()
        mock_frame_reader.rates = {"H1:STRAIN": 16384}

        mock_inj_reader = MagicMock()

        call_count = [0]

        def mock_frame_reader_factory(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_frame_reader
            return mock_inj_reader

        mock_element = MagicMock()

        frames_mod = "sgnligo.sources.datasource_v2.sources.frames"
        with patch(f"{frames_mod}.FrameReader", side_effect=mock_frame_reader_factory):
            with patch(f"{frames_mod}.TSCompose") as mock_compose_cls:
                mock_compose = MagicMock()
                mock_compose.as_source.return_value = mock_element
                mock_compose_cls.return_value = mock_compose

                source = FramesComposedSource(
                    name="test",
                    ifos=["H1"],
                    frame_cache=str(cache_file),
                    channel_dict={"H1": "STRAIN"},
                    t0=1000,
                    end=1010,
                    noiseless_inj_frame_cache=str(inj_cache_file),
                    noiseless_inj_channel_dict={"H1": "INJ-STRAIN"},
                    verbose=True,
                )
                result = source.element
                assert result is mock_element

        captured = capsys.readouterr()
        assert "Added injection for H1" in captured.out

    def test_frames_build_with_segments(self, tmp_path, capsys):
        """Test FramesComposedSource build with segment gating (mocked)."""
        from unittest.mock import MagicMock, patch

        import igwn_segments as segments
        from lal import LIGOTimeGPS

        cache_file = tmp_path / "test.cache"
        cache_file.write_text("")
        segments_file = tmp_path / "segments.xml"
        segments_file.write_text("<dummy/>")

        mock_frame_reader = MagicMock()
        mock_frame_reader.rates = {"H1:STRAIN": 16384}

        # Create mock segment list
        mock_seglist = segments.segmentlistdict()
        mock_seglist["H1"] = segments.segmentlist(
            [segments.segment(LIGOTimeGPS(1000), LIGOTimeGPS(1010))]
        )

        mock_result = MagicMock()
        mock_result.coalesce.return_value = mock_seglist

        mock_element = MagicMock()

        frames_mod = "sgnligo.sources.datasource_v2.sources.frames"
        with patch(f"{frames_mod}.FrameReader", return_value=mock_frame_reader):
            with patch(f"{frames_mod}.TSCompose") as mock_compose_cls:
                mock_compose = MagicMock()
                mock_compose.as_source.return_value = mock_element
                mock_compose_cls.return_value = mock_compose

                with patch(f"{frames_mod}.ligolw_utils.load_filename"):
                    with patch(
                        f"{frames_mod}.ligolw_segments.segmenttable_get_by_name",
                        return_value=mock_result,
                    ):
                        source = FramesComposedSource(
                            name="test",
                            ifos=["H1"],
                            frame_cache=str(cache_file),
                            channel_dict={"H1": "STRAIN"},
                            t0=1000,
                            end=1010,
                            segments_file=str(segments_file),
                            segments_name="SCIENCE",
                            verbose=True,
                        )
                        result = source.element
                        assert result is mock_element

        captured = capsys.readouterr()
        assert "Added segment gating for H1" in captured.out


# =============================================================================
# Tests for DevShm source (mark xfail for missing shared memory)
# =============================================================================


class TestDevShmComposedSourceBuild:
    """Tests for DevShmComposedSource build method with mocking."""

    def test_devshm_build_with_state_gating(self, capsys):
        """Test DevShmComposedSource build with state vector gating (mocked)."""
        from unittest.mock import MagicMock, patch

        mock_devshm = MagicMock()
        mock_element = MagicMock()

        devshm_mod = "sgnligo.sources.datasource_v2.sources.devshm"
        with patch(f"{devshm_mod}.DevShmSource", return_value=mock_devshm):
            with patch(f"{devshm_mod}.TSCompose") as mock_compose_cls:
                mock_compose = MagicMock()
                mock_compose.as_source.return_value = mock_element
                mock_compose_cls.return_value = mock_compose

                source = DevShmComposedSource(
                    name="test",
                    ifos=["H1"],
                    channel_dict={"H1": "STRAIN"},
                    shared_memory_dict={"H1": "/dev/shm/H1"},  # noqa: S108
                    state_channel_dict={"H1": "STATE"},
                    state_vector_on_dict={"H1": 3},
                    verbose=True,
                )
                result = source.element
                assert result is mock_element

        captured = capsys.readouterr()
        assert "Added state vector gating for H1" in captured.out


# =============================================================================
# Tests for Arrakis source build (mark xfail for missing Kafka)
# =============================================================================


@pytest.mark.skipif(not HAS_ARRAKIS, reason="sgn_arrakis not available")
class TestArrakisComposedSourceBuild:
    """Tests for ArrakisComposedSource build method with mocking."""

    def test_arrakis_build_without_state_gating(self):
        """Test ArrakisComposedSource build without state gating (mocked)."""
        from unittest.mock import MagicMock, patch

        mock_arrakis = MagicMock()
        mock_element = MagicMock()

        arrakis_mod = "sgnligo.sources.datasource_v2.sources.arrakis"
        with patch(f"{arrakis_mod}.ArrakisSource", return_value=mock_arrakis):
            with patch(f"{arrakis_mod}.TSCompose") as mock_compose_cls:
                mock_compose = MagicMock()
                mock_compose.as_source.return_value = mock_element
                mock_compose_cls.return_value = mock_compose

                source = ArrakisComposedSource(
                    name="test",
                    ifos=["H1"],
                    channel_dict={"H1": "STRAIN"},
                    t0=1000,
                    end=1010,
                )
                result = source.element
                assert result is mock_element

    def test_arrakis_build_with_state_gating(self, capsys):
        """Test ArrakisComposedSource build with state vector gating (mocked)."""
        from unittest.mock import MagicMock, patch

        mock_arrakis = MagicMock()
        mock_element = MagicMock()

        arrakis_mod = "sgnligo.sources.datasource_v2.sources.arrakis"
        with patch(f"{arrakis_mod}.ArrakisSource", return_value=mock_arrakis):
            with patch(f"{arrakis_mod}.TSCompose") as mock_compose_cls:
                mock_compose = MagicMock()
                mock_compose.as_source.return_value = mock_element
                mock_compose_cls.return_value = mock_compose

                source = ArrakisComposedSource(
                    name="test",
                    ifos=["H1"],
                    channel_dict={"H1": "STRAIN"},
                    state_channel_dict={"H1": "STATE"},
                    state_vector_on_dict={"H1": 3},
                    verbose=True,
                )
                result = source.element
                assert result is mock_element

        captured = capsys.readouterr()
        assert "Added state vector gating for H1" in captured.out


# =============================================================================
# Comprehensive CLI Mixin Tests for 100% Coverage
# =============================================================================


class TestCLIMixinsComprehensive:
    """Comprehensive tests for CLI mixins to achieve 100% coverage."""

    def test_gps_options_mixin_add_cli_arguments(self):
        """Test GPSOptionsMixin.add_cli_arguments."""
        import argparse

        from sgnligo.sources.datasource_v2.cli_mixins import GPSOptionsMixin

        parser = argparse.ArgumentParser()
        GPSOptionsMixin.add_cli_arguments(parser)

        # Verify arguments were added
        args = parser.parse_args(["--gps-start-time", "1000", "--gps-end-time", "1010"])
        assert args.t0 == 1000.0
        assert args.end == 1010.0

    def test_gps_options_mixin_process_cli_args(self):
        """Test GPSOptionsMixin.process_cli_args."""
        import argparse

        from sgnligo.sources.datasource_v2.cli_mixins import GPSOptionsMixin

        # Test with canonical names
        args = argparse.Namespace(t0=1000.0, end=1010.0)
        result = GPSOptionsMixin.process_cli_args(args)
        assert result["t0"] == 1000.0
        assert result["end"] == 1010.0

        # Test with legacy names
        args = argparse.Namespace(gps_start_time=2000.0, gps_end_time=2010.0)
        result = GPSOptionsMixin.process_cli_args(args)
        assert result["t0"] == 2000.0
        assert result["end"] == 2010.0

    def test_gps_options_optional_mixin_add_cli_arguments(self):
        """Test GPSOptionsOptionalMixin.add_cli_arguments."""
        import argparse

        from sgnligo.sources.datasource_v2.cli_mixins import GPSOptionsOptionalMixin

        parser = argparse.ArgumentParser()
        GPSOptionsOptionalMixin.add_cli_arguments(parser)

        # Without arguments - should be None
        args = parser.parse_args([])
        assert args.t0 is None
        assert args.end is None

    def test_gps_options_optional_mixin_process_cli_args(self):
        """Test GPSOptionsOptionalMixin.process_cli_args."""
        import argparse

        from sgnligo.sources.datasource_v2.cli_mixins import GPSOptionsOptionalMixin

        # Test with None values
        args = argparse.Namespace(t0=None, end=None)
        result = GPSOptionsOptionalMixin.process_cli_args(args)
        assert "t0" not in result
        assert "end" not in result

        # Test with values
        args = argparse.Namespace(t0=1000.0, end=1010.0)
        result = GPSOptionsOptionalMixin.process_cli_args(args)
        assert result["t0"] == 1000.0
        assert result["end"] == 1010.0

    def test_channel_options_mixin_add_cli_arguments(self):
        """Test ChannelOptionsMixin.add_cli_arguments."""
        import argparse

        from sgnligo.sources.datasource_v2.cli_mixins import ChannelOptionsMixin

        parser = argparse.ArgumentParser()
        ChannelOptionsMixin.add_cli_arguments(parser)

        args = parser.parse_args(["--channel-name", "H1=STRAIN"])
        assert args.channel_name == ["H1=STRAIN"]

    def test_channel_options_mixin_process_cli_args(self):
        """Test ChannelOptionsMixin.process_cli_args."""
        import argparse

        from sgnligo.sources.datasource_v2.cli_mixins import ChannelOptionsMixin

        # Test with channel names
        args = argparse.Namespace(channel_name=["H1=STRAIN", "L1=STRAIN"])
        result = ChannelOptionsMixin.process_cli_args(args)
        assert result["ifos"] == ["H1", "L1"]
        assert result["channel_dict"] == {"H1": "STRAIN", "L1": "STRAIN"}

        # Test with no channel names
        args = argparse.Namespace(channel_name=None)
        result = ChannelOptionsMixin.process_cli_args(args)
        assert result == {}

    def test_ifos_only_mixin_add_cli_arguments(self):
        """Test IfosOnlyMixin.add_cli_arguments."""
        import argparse

        from sgnligo.sources.datasource_v2.cli_mixins import IfosOnlyMixin

        parser = argparse.ArgumentParser()
        IfosOnlyMixin.add_cli_arguments(parser)

        args = parser.parse_args(["--ifos", "H1", "--ifos", "L1"])
        assert args.ifos == ["H1", "L1"]

    def test_ifos_only_mixin_process_cli_args(self):
        """Test IfosOnlyMixin.process_cli_args."""
        import argparse

        from sgnligo.sources.datasource_v2.cli_mixins import IfosOnlyMixin

        # Test with ifos
        args = argparse.Namespace(ifos=["H1", "L1"], channel_name=None)
        result = IfosOnlyMixin.process_cli_args(args)
        assert result["ifos"] == ["H1", "L1"]

        # Test fallback to channel_name
        args = argparse.Namespace(ifos=None, channel_name=["H1=STRAIN"])
        result = IfosOnlyMixin.process_cli_args(args)
        assert result["ifos"] == ["H1"]

        # Test with neither
        args = argparse.Namespace(ifos=None, channel_name=None)
        result = IfosOnlyMixin.process_cli_args(args)
        assert result == {}

    def test_ifos_from_channel_mixin(self):
        """Test IfosFromChannelMixin methods."""
        import argparse

        from sgnligo.sources.datasource_v2.cli_mixins import IfosFromChannelMixin

        # add_cli_arguments does nothing
        parser = argparse.ArgumentParser()
        IfosFromChannelMixin.add_cli_arguments(parser)
        args = parser.parse_args([])  # Should work with no args

        # get_cli_arg_names returns empty set
        assert IfosFromChannelMixin.get_cli_arg_names() == set()

        # process_cli_args returns empty dict
        result = IfosFromChannelMixin.process_cli_args(args)
        assert result == {}

    def test_sample_rate_mixin(self):
        """Test SampleRateOptionsMixin methods."""
        import argparse

        from sgnligo.sources.datasource_v2.cli_mixins import SampleRateOptionsMixin

        parser = argparse.ArgumentParser()
        SampleRateOptionsMixin.add_cli_arguments(parser)

        args = parser.parse_args(["--sample-rate", "4096"])
        assert args.sample_rate == 4096

        result = SampleRateOptionsMixin.process_cli_args(args)
        assert result["sample_rate"] == 4096

        # Test with no sample rate
        args = argparse.Namespace(sample_rate=None)
        result = SampleRateOptionsMixin.process_cli_args(args)
        assert result == {}

    def test_segments_mixin(self):
        """Test SegmentsOptionsMixin methods."""
        import argparse

        from sgnligo.sources.datasource_v2.cli_mixins import SegmentsOptionsMixin

        parser = argparse.ArgumentParser()
        SegmentsOptionsMixin.add_cli_arguments(parser)

        args = parser.parse_args(
            ["--segments-file", "/path/to/file", "--segments-name", "SCIENCE"]
        )
        assert args.segments_file == "/path/to/file"
        assert args.segments_name == "SCIENCE"

        result = SegmentsOptionsMixin.process_cli_args(args)
        assert result["segments_file"] == "/path/to/file"
        assert result["segments_name"] == "SCIENCE"

    def test_state_vector_options_mixin(self):
        """Test StateVectorOptionsMixin methods."""
        import argparse

        from sgnligo.sources.datasource_v2.cli_mixins import StateVectorOptionsMixin

        parser = argparse.ArgumentParser()
        StateVectorOptionsMixin.add_cli_arguments(parser)

        args = parser.parse_args(
            [
                "--state-channel-name",
                "H1=STATE",
                "--state-vector-on-bits",
                "H1=3",
                "--state-segments-file",
                "/path/to/file",
                "--state-sample-rate",
                "32",
            ]
        )

        result = StateVectorOptionsMixin.process_cli_args(args)
        assert result["state_channel_dict"] == {"H1": "STATE"}
        assert result["state_vector_on_dict"] == {"H1": 3}
        assert result["state_segments_file"] == "/path/to/file"
        assert result["state_sample_rate"] == 32

    def test_state_vector_on_dict_only_mixin(self):
        """Test StateVectorOnDictOnlyMixin methods."""
        import argparse

        from sgnligo.sources.datasource_v2.cli_mixins import StateVectorOnDictOnlyMixin

        parser = argparse.ArgumentParser()
        StateVectorOnDictOnlyMixin.add_cli_arguments(parser)

        args = parser.parse_args(
            [
                "--state-vector-on-bits",
                "H1=3",
                "--state-segments-file",
                "/path/to/file",
                "--state-sample-rate",
                "32",
            ]
        )

        result = StateVectorOnDictOnlyMixin.process_cli_args(args)
        assert result["state_vector_on_dict"] == {"H1": 3}
        assert result["state_segments_file"] == "/path/to/file"
        assert result["state_sample_rate"] == 32

    def test_injection_options_mixin(self):
        """Test InjectionOptionsMixin methods."""
        import argparse

        from sgnligo.sources.datasource_v2.cli_mixins import InjectionOptionsMixin

        parser = argparse.ArgumentParser()
        InjectionOptionsMixin.add_cli_arguments(parser)

        args = parser.parse_args(
            [
                "--noiseless-inj-frame-cache",
                "/path/to/cache",
                "--noiseless-inj-channel-name",
                "H1=INJ",
            ]
        )

        result = InjectionOptionsMixin.process_cli_args(args)
        assert result["noiseless_inj_frame_cache"] == "/path/to/cache"
        assert result["noiseless_inj_channel_dict"] == {"H1": "INJ"}

    def test_devshm_options_mixin(self):
        """Test DevShmOptionsMixin methods."""
        import argparse

        from sgnligo.sources.datasource_v2.cli_mixins import DevShmOptionsMixin

        parser = argparse.ArgumentParser()
        DevShmOptionsMixin.add_cli_arguments(parser)

        args = parser.parse_args(
            [
                "--shared-memory-dir",
                "H1=/dev/shm/H1",
                "--discont-wait-time",
                "120.0",
            ]
        )

        result = DevShmOptionsMixin.process_cli_args(args)
        assert result["shared_memory_dict"] == {"H1": "/dev/shm/H1"}  # noqa: S108
        assert result["discont_wait_time"] == 120.0

    def test_queue_timeout_mixin(self):
        """Test QueueTimeoutOptionsMixin methods."""
        import argparse

        from sgnligo.sources.datasource_v2.cli_mixins import QueueTimeoutOptionsMixin

        parser = argparse.ArgumentParser()
        QueueTimeoutOptionsMixin.add_cli_arguments(parser)

        args = parser.parse_args(["--queue-timeout", "5.0"])
        result = QueueTimeoutOptionsMixin.process_cli_args(args)
        assert result["queue_timeout"] == 5.0

        # Test with None
        args = argparse.Namespace(queue_timeout=None)
        result = QueueTimeoutOptionsMixin.process_cli_args(args)
        assert result == {}

    def test_frame_cache_mixin(self):
        """Test FrameCacheOptionsMixin methods."""
        import argparse

        from sgnligo.sources.datasource_v2.cli_mixins import FrameCacheOptionsMixin

        parser = argparse.ArgumentParser()
        FrameCacheOptionsMixin.add_cli_arguments(parser)

        args = parser.parse_args(["--frame-cache", "/path/to/cache"])
        result = FrameCacheOptionsMixin.process_cli_args(args)
        assert result["frame_cache"] == "/path/to/cache"

        # Test with None
        args = argparse.Namespace(frame_cache=None)
        result = FrameCacheOptionsMixin.process_cli_args(args)
        assert result == {}

    def test_channel_pattern_mixin(self):
        """Test ChannelPatternOptionsMixin methods."""
        import argparse

        from sgnligo.sources.datasource_v2.cli_mixins import ChannelPatternOptionsMixin

        parser = argparse.ArgumentParser()
        ChannelPatternOptionsMixin.add_cli_arguments(parser)

        args = parser.parse_args(["--channel-pattern", "{ifo}:CUSTOM"])
        result = ChannelPatternOptionsMixin.process_cli_args(args)
        assert result["channel_pattern"] == "{ifo}:CUSTOM"

        # Test with None
        args = argparse.Namespace(channel_pattern=None)
        result = ChannelPatternOptionsMixin.process_cli_args(args)
        assert result == {}

    def test_impulse_position_mixin(self):
        """Test ImpulsePositionOptionsMixin methods."""
        import argparse

        from sgnligo.sources.datasource_v2.cli_mixins import ImpulsePositionOptionsMixin

        parser = argparse.ArgumentParser()
        ImpulsePositionOptionsMixin.add_cli_arguments(parser)

        args = parser.parse_args(["--impulse-position", "100"])
        result = ImpulsePositionOptionsMixin.process_cli_args(args)
        assert result["impulse_position"] == 100

        # Test with None
        args = argparse.Namespace(impulse_position=None)
        result = ImpulsePositionOptionsMixin.process_cli_args(args)
        assert result == {}

    def test_verbose_mixin(self):
        """Test VerboseOptionsMixin methods."""
        import argparse

        from sgnligo.sources.datasource_v2.cli_mixins import VerboseOptionsMixin

        parser = argparse.ArgumentParser()
        VerboseOptionsMixin.add_cli_arguments(parser)

        args = parser.parse_args(["--verbose"])
        result = VerboseOptionsMixin.process_cli_args(args)
        assert result["verbose"] is True

        # Test with None
        args = argparse.Namespace(verbose=None)
        result = VerboseOptionsMixin.process_cli_args(args)
        assert result == {}


class TestCLIBuilderEdgeCases:
    """Tests for CLI builder edge cases."""

    def test_build_composed_cli_parser_skips_protocol_class(self):
        """Test that CLIMixinProtocol itself is skipped during CLI building."""
        from sgnligo.sources.datasource_v2.cli import build_composed_cli_parser

        # Should not raise - just verify it builds without error
        parser = build_composed_cli_parser()
        assert parser is not None

    def test_namespace_to_datasource_kwargs_skips_protocol_class(self):
        """Test that CLIMixinProtocol is skipped during kwargs conversion."""
        import argparse

        from sgnligo.sources.datasource_v2.cli import namespace_to_datasource_kwargs

        args = argparse.Namespace(
            data_source="white",
            ifos=["H1"],
            sample_rate=4096,
            t0=1000.0,
            end=1010.0,
            verbose=False,
        )

        # Should not raise - just verify it converts without error
        kwargs = namespace_to_datasource_kwargs(args)
        assert kwargs["data_source"] == "white"


class TestComposedBaseLatencyTracking:
    """Tests for ComposedSourceBase latency tracking CLI support."""

    def test_latency_interval_process_cli_args(self):
        """Test process_cli_args handles source_latency_interval."""
        import argparse

        from sgnligo.sources.composed_base import ComposedSourceBase

        # Test with latency interval set
        args = argparse.Namespace(source_latency_interval=10.0)
        result = ComposedSourceBase.process_cli_args(args)
        assert result["latency_interval"] == 10.0

        # Test with None
        args = argparse.Namespace(source_latency_interval=None)
        result = ComposedSourceBase.process_cli_args(args)
        assert "latency_interval" not in result

    def test_latency_interval_add_cli_arguments(self):
        """Test add_cli_arguments adds source-latency-interval."""
        import argparse

        from sgnligo.sources.composed_base import ComposedSourceBase

        parser = argparse.ArgumentParser()
        ComposedSourceBase.add_cli_arguments(parser)

        args = parser.parse_args(["--source-latency-interval", "5.0"])
        assert args.source_latency_interval == 5.0

    def test_latency_interval_get_cli_arg_names(self):
        """Test get_cli_arg_names returns correct names."""
        from sgnligo.sources.composed_base import ComposedSourceBase

        names = ComposedSourceBase.get_cli_arg_names()
        assert "source_latency_interval" in names


class TestDataSourceFromArgvHelp:
    """Tests for DataSource from_argv help handling."""

    def test_from_argv_list_sources_exits(self):
        """Test from_argv exits on --list-sources."""
        import pytest

        from sgnligo.sources.datasource_v2 import DataSource

        # Should raise SystemExit(0) when --list-sources is provided
        with pytest.raises(SystemExit) as exc_info:
            DataSource.from_argv(argv=["--list-sources"])
        assert exc_info.value.code == 0

    def test_from_argv_help_source_exits(self):
        """Test from_argv exits on --help-source."""
        import pytest

        from sgnligo.sources.datasource_v2 import DataSource

        with pytest.raises(SystemExit) as exc_info:
            DataSource.from_argv(argv=["--help-source", "white"])
        assert exc_info.value.code == 0

    def test_from_parser_list_sources_exits(self):
        """Test from_parser exits on --list-sources."""
        import pytest

        from sgnligo.sources.datasource_v2 import DataSource

        parser = DataSource.create_cli_parser()
        with pytest.raises(SystemExit) as exc_info:
            DataSource.from_parser(parser, argv=["--list-sources"])
        assert exc_info.value.code == 0

    def test_from_parser_help_source_exits(self):
        """Test from_parser exits on --help-source."""
        import pytest

        from sgnligo.sources.datasource_v2 import DataSource

        parser = DataSource.create_cli_parser()
        with pytest.raises(SystemExit) as exc_info:
            DataSource.from_parser(parser, argv=["--help-source", "gwdata-noise"])
        assert exc_info.value.code == 0


class TestComposedBaseValidate:
    """Tests for ComposedSourceBase _validate method."""

    def test_base_validate_does_nothing(self):
        """Test that base _validate is a no-op."""
        from dataclasses import dataclass

        from sgnts.compose import TSCompose, TSComposedSourceElement
        from sgnts.sources import FakeSeriesSource

        from sgnligo.sources.composed_base import ComposedSourceBase

        @dataclass(kw_only=True)
        class TestSource(ComposedSourceBase):
            source_type = ""
            description = "Test"

            def _build(self) -> TSComposedSourceElement:
                compose = TSCompose()
                fake = FakeSeriesSource(
                    name=f"{self.name}_fake",
                    source_pad_names=("data",),
                    rate=1024,  # Must be a valid sample rate
                    t0=0,
                    end=1,
                )
                compose.insert(fake)
                return compose.as_source(name=self.name)

        # Should work - base _validate does nothing
        source = TestSource(name="test")
        assert source is not None


class TestCLIBuildWithDuplicateMixins:
    """Tests for CLI builder handling of MRO with shared mixins."""

    def test_build_handles_shared_mixin_in_mro(self):
        """Test that shared mixins in MRO are only processed once."""
        import argparse

        from sgnligo.sources.datasource_v2.cli import namespace_to_datasource_kwargs

        # WhiteComposedSource and SinComposedSource share IfosOnlyMixin
        # Verify processing works without errors
        args = argparse.Namespace(
            data_source="white",
            ifos=["H1"],
            sample_rate=4096,
            t0=1000.0,
            end=1010.0,
            verbose=False,
        )

        kwargs = namespace_to_datasource_kwargs(args)
        assert kwargs["ifos"] == ["H1"]

    def test_namespace_kwargs_unknown_source_type(self):
        """Test namespace_to_datasource_kwargs with unknown source type."""
        import argparse

        from sgnligo.sources.datasource_v2.cli import namespace_to_datasource_kwargs

        # Create namespace with unknown source type
        args = argparse.Namespace(data_source="unknown-source-xyz")

        kwargs = namespace_to_datasource_kwargs(args)
        # Should only have data_source, no other kwargs
        assert kwargs == {"data_source": "unknown-source-xyz"}
