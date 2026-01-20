"""CLI option mixins for composed data sources.

These mixins define both dataclass fields AND their corresponding CLI arguments.
Sources inherit from the mixins they need, enabling composition of CLI options.

All mixins use `@dataclass(kw_only=True)` to avoid field ordering issues when
combining multiple mixins via multiple inheritance.

Example:
    >>> @register_composed_source
    ... @dataclass(kw_only=True)
    ... class MySource(ComposedSourceBase, GPSOptionsMixin, VerboseOptionsMixin):
    ...     # Inherits t0, end from GPSOptionsMixin
    ...     # Inherits verbose from VerboseOptionsMixin
    ...     my_field: str  # Source-specific field
    ...
    ...     source_type: ClassVar[str] = "my-source"
    ...     description: ClassVar[str] = "My custom source"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Protocol, Set

from sgnligo.base import parse_list_to_dict

# =============================================================================
# Base Protocol for CLI Mixins
# =============================================================================


class CLIMixinProtocol(Protocol):
    """Protocol defining the interface for CLI mixins.

    All mixins should implement:
    - add_cli_arguments(parser): Add CLI arguments to the parser
    - get_cli_arg_names(): Return set of CLI arg names (for duplicate detection)
    - process_cli_args(args): Convert CLI args to field values
    """

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add CLI arguments for this mixin."""
        ...  # pragma: no cover

    @classmethod
    def get_cli_arg_names(cls) -> Set[str]:
        """Return set of CLI argument names defined by this mixin."""
        ...  # pragma: no cover

    @classmethod
    def process_cli_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert CLI args to field values."""
        ...  # pragma: no cover


# =============================================================================
# GPS Time Options
# =============================================================================


@dataclass(kw_only=True)
class GPSOptionsMixin:
    """Mixin for required GPS time range options.

    Used by offline sources that require explicit start/end times.

    Fields:
        t0: GPS start time
        end: GPS end time
    """

    t0: float
    end: float

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add GPS time range CLI arguments."""
        parser.add_argument(
            "--gps-start-time",
            type=float,
            metavar="GPS",
            dest="t0",
            help="GPS start time",
        )
        parser.add_argument(
            "--gps-end-time",
            type=float,
            metavar="GPS",
            dest="end",
            help="GPS end time",
        )

    @classmethod
    def get_cli_arg_names(cls) -> Set[str]:
        return {"t0", "end"}

    @classmethod
    def process_cli_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert GPS CLI args to field values.

        Handles both canonical names (t0, end) and legacy names
        (gps_start_time, gps_end_time) for backward compatibility.
        """
        result: Dict[str, Any] = {}
        # Check canonical name first, then legacy name
        t0 = getattr(args, "t0", None) or getattr(args, "gps_start_time", None)
        if t0 is not None:
            result["t0"] = t0
        end = getattr(args, "end", None) or getattr(args, "gps_end_time", None)
        if end is not None:
            result["end"] = end
        return result


@dataclass(kw_only=True)
class GPSOptionsOptionalMixin:
    """Mixin for optional GPS time range options.

    Used by real-time sources where start/end times are optional.

    Fields:
        t0: GPS start time (optional, defaults to current time)
        end: GPS end time (optional for indefinite operation)
    """

    t0: Optional[float] = None
    end: Optional[float] = None

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add optional GPS time range CLI arguments."""
        parser.add_argument(
            "--gps-start-time",
            type=float,
            metavar="GPS",
            dest="t0",
            default=None,
            help="GPS start time (optional)",
        )
        parser.add_argument(
            "--gps-end-time",
            type=float,
            metavar="GPS",
            dest="end",
            default=None,
            help="GPS end time (optional)",
        )

    @classmethod
    def get_cli_arg_names(cls) -> Set[str]:
        return {"t0", "end"}

    @classmethod
    def process_cli_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert GPS CLI args to field values.

        Handles both canonical names (t0, end) and legacy names
        (gps_start_time, gps_end_time) for backward compatibility.
        """
        result: Dict[str, Any] = {}
        # Check canonical name first, then legacy name
        t0 = getattr(args, "t0", None) or getattr(args, "gps_start_time", None)
        if t0 is not None:
            result["t0"] = t0
        end = getattr(args, "end", None) or getattr(args, "gps_end_time", None)
        if end is not None:
            result["end"] = end
        return result


# =============================================================================
# Channel Options
# =============================================================================


@dataclass(kw_only=True)
class ChannelOptionsMixin:
    """Mixin for channel specification options.

    Provides channel_dict field. When used from CLI, ifos are derived from
    channel_dict keys.

    Fields:
        channel_dict: Dict mapping IFO to channel name
    """

    channel_dict: Dict[str, str]

    # Track that this mixin provides ifos derivation
    _derives_ifos: ClassVar[bool] = True

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add channel specification CLI arguments."""
        parser.add_argument(
            "--channel-name",
            action="append",
            metavar="IFO=CHANNEL",
            help="Channel as IFO=CHANNEL (repeatable, e.g., H1=GDS-CALIB_STRAIN)",
        )

    @classmethod
    def get_cli_arg_names(cls) -> Set[str]:
        return {"channel_name"}

    @classmethod
    def process_cli_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert --channel-name args to channel_dict and derive ifos."""
        channel_name = getattr(args, "channel_name", None)
        if channel_name:
            channel_dict = parse_list_to_dict(channel_name)
            return {
                "ifos": sorted(channel_dict.keys()),
                "channel_dict": channel_dict,
            }
        return {}


@dataclass(kw_only=True)
class IfosOnlyMixin:
    """Mixin for sources that only need IFO list (no channel dict).

    Used by fake sources that generate their own channel names.

    Fields:
        ifos: List of detector prefixes
    """

    ifos: List[str]

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add IFO list CLI argument."""
        parser.add_argument(
            "--ifos",
            action="append",
            metavar="IFO",
            help="Detector prefix (repeatable, e.g., H1 L1)",
        )

    @classmethod
    def get_cli_arg_names(cls) -> Set[str]:
        return {"ifos"}

    @classmethod
    def process_cli_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert --ifos args to ifos list.

        Also derives ifos from channel_name if --ifos not provided,
        for backward compatibility with pipelines that use --channel-name.
        """
        from sgnligo.base import parse_list_to_dict

        ifos = getattr(args, "ifos", None)
        if ifos:
            return {"ifos": ifos}

        # Fallback: derive ifos from channel_name if provided
        channel_name = getattr(args, "channel_name", None)
        if channel_name:
            channel_dict = parse_list_to_dict(channel_name)
            return {"ifos": sorted(channel_dict.keys())}

        return {}


@dataclass(kw_only=True)
class IfosFromChannelMixin:
    """Mixin that provides ifos field, derived from channel_dict when using CLI.

    This mixin should be used WITH ChannelOptionsMixin. The ifos field is
    required when instantiating directly, but derived automatically from
    channel_dict when using CLI.

    Fields:
        ifos: List of detector prefixes
    """

    ifos: List[str]

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """No additional CLI arguments - ifos derived from channel_dict."""
        pass

    @classmethod
    def get_cli_arg_names(cls) -> Set[str]:
        return set()  # No CLI args, derived from ChannelOptionsMixin

    @classmethod
    def process_cli_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """ifos are derived by ChannelOptionsMixin.process_cli_args."""
        return {}


# =============================================================================
# Sample Rate Options
# =============================================================================


@dataclass(kw_only=True)
class SampleRateOptionsMixin:
    """Mixin for sample rate option.

    Fields:
        sample_rate: Sample rate in Hz
    """

    sample_rate: int

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add sample rate CLI argument."""
        parser.add_argument(
            "--sample-rate",
            type=int,
            metavar="HZ",
            help="Sample rate in Hz",
        )

    @classmethod
    def get_cli_arg_names(cls) -> Set[str]:
        return {"sample_rate"}

    @classmethod
    def process_cli_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert sample_rate CLI arg to field value."""
        sample_rate = getattr(args, "sample_rate", None)
        if sample_rate is not None:
            return {"sample_rate": sample_rate}
        return {}


# =============================================================================
# Segment Gating Options
# =============================================================================


@dataclass(kw_only=True)
class SegmentsOptionsMixin:
    """Mixin for segment gating options.

    Fields:
        segments_file: Path to LIGO XML segments file
        segments_name: Name of segments to extract from XML
    """

    segments_file: Optional[str] = None
    segments_name: Optional[str] = None

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add segment gating CLI arguments."""
        parser.add_argument(
            "--segments-file",
            metavar="FILE",
            help="Path to LIGO XML segments file",
        )
        parser.add_argument(
            "--segments-name",
            metavar="NAME",
            help="Segment name in XML file",
        )

    @classmethod
    def get_cli_arg_names(cls) -> Set[str]:
        return {"segments_file", "segments_name"}

    @classmethod
    def process_cli_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert segment CLI args to field values."""
        result: Dict[str, Any] = {}
        segments_file = getattr(args, "segments_file", None)
        if segments_file is not None:
            result["segments_file"] = segments_file
        segments_name = getattr(args, "segments_name", None)
        if segments_name is not None:
            result["segments_name"] = segments_name
        return result


# =============================================================================
# State Vector Options
# =============================================================================


@dataclass(kw_only=True)
class StateVectorOptionsMixin:
    """Mixin for state vector gating options.

    Fields:
        state_channel_dict: Dict mapping IFO to state vector channel
        state_vector_on_dict: Dict mapping IFO to bitmask for valid data
        state_segments_file: Path to file with state segments
        state_sample_rate: Sample rate for state vector (default: 16 Hz)
    """

    state_channel_dict: Optional[Dict[str, str]] = None
    state_vector_on_dict: Optional[Dict[str, int]] = None
    state_segments_file: Optional[str] = None
    state_sample_rate: int = 16

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add state vector gating CLI arguments."""
        parser.add_argument(
            "--state-channel-name",
            action="append",
            metavar="IFO=CHANNEL",
            help="State vector channel as IFO=CHANNEL (repeatable)",
        )
        parser.add_argument(
            "--state-vector-on-bits",
            action="append",
            metavar="IFO=BITS",
            help="State vector bitmask as IFO=BITS (repeatable, e.g., H1=3)",
        )
        parser.add_argument(
            "--state-segments-file",
            metavar="FILE",
            help="Path to state segments file",
        )
        parser.add_argument(
            "--state-sample-rate",
            type=int,
            default=16,
            metavar="HZ",
            help="State vector sample rate (default: 16)",
        )

    @classmethod
    def get_cli_arg_names(cls) -> Set[str]:
        return {
            "state_channel_name",
            "state_vector_on_bits",
            "state_segments_file",
            "state_sample_rate",
        }

    @classmethod
    def process_cli_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert state vector CLI args to field values."""
        result: Dict[str, Any] = {}

        state_channel_name = getattr(args, "state_channel_name", None)
        if state_channel_name:
            result["state_channel_dict"] = parse_list_to_dict(state_channel_name)

        state_vector_on_bits = getattr(args, "state_vector_on_bits", None)
        if state_vector_on_bits:
            result["state_vector_on_dict"] = parse_list_to_dict(
                state_vector_on_bits, value_transform=int
            )

        state_segments_file = getattr(args, "state_segments_file", None)
        if state_segments_file:
            result["state_segments_file"] = state_segments_file

        state_sample_rate = getattr(args, "state_sample_rate", None)
        if state_sample_rate is not None:
            result["state_sample_rate"] = state_sample_rate

        return result


@dataclass(kw_only=True)
class StateVectorOnDictOnlyMixin:
    """Mixin for state vector bitmask without state channel (uses SegmentSource).

    Used by GWDataNoise sources that use SegmentSource for state vector instead
    of reading from a real state vector channel.

    Fields:
        state_vector_on_dict: Dict mapping IFO to bitmask for valid data
        state_segments_file: Path to file with state segments
        state_sample_rate: Sample rate for state vector (default: 16 Hz)
    """

    state_vector_on_dict: Optional[Dict[str, int]] = None
    state_segments_file: Optional[str] = None
    state_sample_rate: int = 16

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add state vector bitmask CLI arguments."""
        parser.add_argument(
            "--state-vector-on-bits",
            action="append",
            metavar="IFO=BITS",
            help="State vector bitmask as IFO=BITS (repeatable, e.g., H1=3)",
        )
        parser.add_argument(
            "--state-segments-file",
            metavar="FILE",
            help="Path to state segments file",
        )
        parser.add_argument(
            "--state-sample-rate",
            type=int,
            default=16,
            metavar="HZ",
            help="State vector sample rate (default: 16)",
        )

    @classmethod
    def get_cli_arg_names(cls) -> Set[str]:
        return {"state_vector_on_bits", "state_segments_file", "state_sample_rate"}

    @classmethod
    def process_cli_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert state vector CLI args to field values."""
        result: Dict[str, Any] = {}

        state_vector_on_bits = getattr(args, "state_vector_on_bits", None)
        if state_vector_on_bits:
            result["state_vector_on_dict"] = parse_list_to_dict(
                state_vector_on_bits, value_transform=int
            )

        state_segments_file = getattr(args, "state_segments_file", None)
        if state_segments_file:
            result["state_segments_file"] = state_segments_file

        state_sample_rate = getattr(args, "state_sample_rate", None)
        if state_sample_rate is not None:
            result["state_sample_rate"] = state_sample_rate

        return result


# =============================================================================
# Injection Options
# =============================================================================


@dataclass(kw_only=True)
class InjectionOptionsMixin:
    """Mixin for noiseless injection options.

    Fields:
        noiseless_inj_frame_cache: Path to injection frame cache
        noiseless_inj_channel_dict: Dict mapping IFO to injection channel
    """

    noiseless_inj_frame_cache: Optional[str] = None
    noiseless_inj_channel_dict: Optional[Dict[str, str]] = None

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add injection CLI arguments."""
        parser.add_argument(
            "--noiseless-inj-frame-cache",
            metavar="FILE",
            help="Path to injection frame cache",
        )
        parser.add_argument(
            "--noiseless-inj-channel-name",
            action="append",
            metavar="IFO=CHANNEL",
            help="Injection channel as IFO=CHANNEL (repeatable)",
        )

    @classmethod
    def get_cli_arg_names(cls) -> Set[str]:
        return {"noiseless_inj_frame_cache", "noiseless_inj_channel_name"}

    @classmethod
    def process_cli_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert injection CLI args to field values."""
        result: Dict[str, Any] = {}

        noiseless_inj_frame_cache = getattr(args, "noiseless_inj_frame_cache", None)
        if noiseless_inj_frame_cache:
            result["noiseless_inj_frame_cache"] = noiseless_inj_frame_cache

        noiseless_inj_channel_name = getattr(args, "noiseless_inj_channel_name", None)
        if noiseless_inj_channel_name:
            result["noiseless_inj_channel_dict"] = parse_list_to_dict(
                noiseless_inj_channel_name
            )

        return result


# =============================================================================
# DevShm Options
# =============================================================================


@dataclass(kw_only=True)
class DevShmOptionsMixin:
    """Mixin for shared memory (devshm) options.

    Fields:
        shared_memory_dict: Dict mapping IFO to shared memory directory
        discont_wait_time: Time to wait before dropping data (default: 60s)
    """

    shared_memory_dict: Dict[str, str]
    discont_wait_time: float = 60.0

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add shared memory CLI arguments."""
        parser.add_argument(
            "--shared-memory-dir",
            action="append",
            metavar="IFO=PATH",
            help="Shared memory directory as IFO=PATH (repeatable)",
        )
        parser.add_argument(
            "--discont-wait-time",
            type=float,
            default=60.0,
            help="Discontinuity wait time in seconds (default: 60)",
        )

    @classmethod
    def get_cli_arg_names(cls) -> Set[str]:
        return {"shared_memory_dir", "discont_wait_time"}

    @classmethod
    def process_cli_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert shared memory CLI args to field values."""
        result: Dict[str, Any] = {}

        shared_memory_dir = getattr(args, "shared_memory_dir", None)
        if shared_memory_dir:
            result["shared_memory_dict"] = parse_list_to_dict(shared_memory_dir)

        discont_wait_time = getattr(args, "discont_wait_time", None)
        if discont_wait_time is not None:
            result["discont_wait_time"] = discont_wait_time

        return result


# =============================================================================
# Queue Timeout Options
# =============================================================================


@dataclass(kw_only=True)
class QueueTimeoutOptionsMixin:
    """Mixin for queue timeout option.

    Shared by DevShm and Arrakis sources.

    Fields:
        queue_timeout: Queue timeout in seconds (default: 1.0)
    """

    queue_timeout: float = 1.0

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add queue timeout CLI argument."""
        parser.add_argument(
            "--queue-timeout",
            type=float,
            default=1.0,
            help="Queue timeout in seconds (default: 1.0)",
        )

    @classmethod
    def get_cli_arg_names(cls) -> Set[str]:
        return {"queue_timeout"}

    @classmethod
    def process_cli_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert queue timeout CLI arg to field value."""
        queue_timeout = getattr(args, "queue_timeout", None)
        if queue_timeout is not None:
            return {"queue_timeout": queue_timeout}
        return {}


# =============================================================================
# Frame Cache Options
# =============================================================================


@dataclass(kw_only=True)
class FrameCacheOptionsMixin:
    """Mixin for frame cache option.

    Fields:
        frame_cache: Path to LAL cache file
    """

    frame_cache: str

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add frame cache CLI argument."""
        parser.add_argument(
            "--frame-cache",
            metavar="FILE",
            help="Path to LAL cache file",
        )

    @classmethod
    def get_cli_arg_names(cls) -> Set[str]:
        return {"frame_cache"}

    @classmethod
    def process_cli_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert frame cache CLI arg to field value."""
        frame_cache = getattr(args, "frame_cache", None)
        if frame_cache is not None:
            return {"frame_cache": frame_cache}
        return {}


# =============================================================================
# Channel Pattern Options
# =============================================================================


@dataclass(kw_only=True)
class ChannelPatternOptionsMixin:
    """Mixin for channel pattern option.

    Used by sources that generate channel names from a pattern.

    Fields:
        channel_pattern: Pattern for generating channel names
            (default: "{ifo}:FAKE-STRAIN")
    """

    channel_pattern: str = "{ifo}:FAKE-STRAIN"

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add channel pattern CLI argument."""
        parser.add_argument(
            "--channel-pattern",
            default="{ifo}:FAKE-STRAIN",
            metavar="PATTERN",
            help="Channel naming pattern (default: {ifo}:FAKE-STRAIN)",
        )

    @classmethod
    def get_cli_arg_names(cls) -> Set[str]:
        return {"channel_pattern"}

    @classmethod
    def process_cli_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert channel pattern CLI arg to field value."""
        channel_pattern = getattr(args, "channel_pattern", None)
        if channel_pattern is not None:
            return {"channel_pattern": channel_pattern}
        return {}


# =============================================================================
# Impulse Position Options
# =============================================================================


@dataclass(kw_only=True)
class ImpulsePositionOptionsMixin:
    """Mixin for impulse position option.

    Fields:
        impulse_position: Sample index for impulse (-1 for random)
    """

    impulse_position: int = -1

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add impulse position CLI argument."""
        parser.add_argument(
            "--impulse-position",
            type=int,
            default=-1,
            metavar="INDEX",
            help="Impulse sample index (-1 for random, default: -1)",
        )

    @classmethod
    def get_cli_arg_names(cls) -> Set[str]:
        return {"impulse_position"}

    @classmethod
    def process_cli_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert impulse position CLI arg to field value."""
        impulse_position = getattr(args, "impulse_position", None)
        if impulse_position is not None:
            return {"impulse_position": impulse_position}
        return {}


# =============================================================================
# Verbose Options
# =============================================================================


@dataclass(kw_only=True)
class VerboseOptionsMixin:
    """Mixin for verbose output option.

    Fields:
        verbose: Enable verbose output (default: False)
    """

    verbose: bool = False

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add verbose CLI argument."""
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose output",
        )

    @classmethod
    def get_cli_arg_names(cls) -> Set[str]:
        return {"verbose"}

    @classmethod
    def process_cli_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert verbose CLI arg to field value."""
        verbose = getattr(args, "verbose", None)
        if verbose is not None:
            return {"verbose": verbose}
        return {}


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Protocol
    "CLIMixinProtocol",
    # GPS time
    "GPSOptionsMixin",
    "GPSOptionsOptionalMixin",
    # Channels
    "ChannelOptionsMixin",
    "IfosOnlyMixin",
    "IfosFromChannelMixin",
    "ChannelPatternOptionsMixin",
    # Sample rate
    "SampleRateOptionsMixin",
    # Segments
    "SegmentsOptionsMixin",
    # State vector
    "StateVectorOptionsMixin",
    "StateVectorOnDictOnlyMixin",
    # Injection
    "InjectionOptionsMixin",
    # DevShm
    "DevShmOptionsMixin",
    "QueueTimeoutOptionsMixin",
    # Frames
    "FrameCacheOptionsMixin",
    # Impulse
    "ImpulsePositionOptionsMixin",
    # Verbose
    "VerboseOptionsMixin",
]
