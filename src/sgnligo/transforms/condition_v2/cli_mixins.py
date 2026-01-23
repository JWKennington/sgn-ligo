"""CLI option mixins for composed condition transforms.

These mixins define both dataclass fields AND their corresponding CLI arguments.
Transforms inherit from the mixins they need, enabling composition of CLI options.

All mixins use `@dataclass(kw_only=True)` to avoid field ordering issues when
combining multiple mixins via multiple inheritance.

Example:
    >>> @register_composed_transform
    ... @dataclass(kw_only=True)
    ... class StandardCondition(
    ...     ComposedTransformBase,
    ...     PSDOptionsMixin,
    ...     GatingOptionsMixin,
    ... ):
    ...     # Inherits psd_fft_length, reference_psd, track_psd from PSDOptionsMixin
    ...     # Inherits ht_gate_threshold from GatingOptionsMixin
    ...     my_field: str  # Transform-specific field
    ...
    ...     transform_type: ClassVar[str] = "standard"
    ...     description: ClassVar[str] = "Standard whitening"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Set

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
# Input Sample Rate Options
# =============================================================================


@dataclass(kw_only=True)
class InputSampleRateOptionsMixin:
    """Mixin for input sample rate option.

    Fields:
        input_sample_rate: Sample rate of input data in Hz
    """

    input_sample_rate: int

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add input sample rate CLI argument."""
        parser.add_argument(
            "--input-sample-rate",
            type=int,
            metavar="HZ",
            help="Input sample rate in Hz",
        )

    @classmethod
    def get_cli_arg_names(cls) -> Set[str]:
        return {"input_sample_rate"}

    @classmethod
    def process_cli_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert input sample rate CLI arg to field value."""
        input_sample_rate = getattr(args, "input_sample_rate", None)
        if input_sample_rate is not None:
            return {"input_sample_rate": input_sample_rate}
        return {}


# =============================================================================
# Whitening Sample Rate Options
# =============================================================================


@dataclass(kw_only=True)
class WhitenSampleRateOptionsMixin:
    """Mixin for whitening sample rate option.

    Fields:
        whiten_sample_rate: Sample rate for whitening (Hz)
    """

    whiten_sample_rate: int = 2048

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add whitening sample rate CLI argument."""
        parser.add_argument(
            "--whiten-sample-rate",
            type=int,
            default=2048,
            metavar="HZ",
            help="Sample rate for whitening (default: 2048)",
        )

    @classmethod
    def get_cli_arg_names(cls) -> Set[str]:
        return {"whiten_sample_rate"}

    @classmethod
    def process_cli_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert whitening sample rate CLI arg to field value."""
        whiten_sample_rate = getattr(args, "whiten_sample_rate", None)
        if whiten_sample_rate is not None:
            return {"whiten_sample_rate": whiten_sample_rate}
        return {}


# =============================================================================
# PSD Options
# =============================================================================


@dataclass(kw_only=True)
class PSDOptionsMixin:
    """Mixin for PSD estimation options.

    Fields:
        psd_fft_length: FFT length for PSD estimation in seconds
        reference_psd: Path to reference PSD XML file (optional)
        track_psd: Enable dynamic PSD tracking
    """

    psd_fft_length: int = 8
    reference_psd: Optional[str] = None
    track_psd: bool = True

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add PSD options CLI arguments."""
        group = parser.add_argument_group("PSD Options")
        group.add_argument(
            "--psd-fft-length",
            type=int,
            default=8,
            metavar="SECONDS",
            help="FFT length for PSD estimation in seconds (default: 8)",
        )
        group.add_argument(
            "--reference-psd",
            metavar="FILE",
            help="Path to reference PSD XML file",
        )
        group.add_argument(
            "--track-psd",
            action="store_true",
            default=True,
            help="Enable dynamic PSD tracking (default: True)",
        )
        group.add_argument(
            "--no-track-psd",
            action="store_false",
            dest="track_psd",
            help="Disable dynamic PSD tracking",
        )

    @classmethod
    def get_cli_arg_names(cls) -> Set[str]:
        return {"psd_fft_length", "reference_psd", "track_psd"}

    @classmethod
    def process_cli_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert PSD CLI args to field values."""
        result: Dict[str, Any] = {}
        if hasattr(args, "psd_fft_length") and args.psd_fft_length is not None:
            result["psd_fft_length"] = args.psd_fft_length
        if hasattr(args, "reference_psd") and args.reference_psd is not None:
            result["reference_psd"] = args.reference_psd
        if hasattr(args, "track_psd"):
            result["track_psd"] = args.track_psd
        return result


# =============================================================================
# Gating Options
# =============================================================================


@dataclass(kw_only=True)
class GatingOptionsMixin:
    """Mixin for data gating/threshold options.

    Fields:
        ht_gate_threshold: Threshold for gating (inf = disabled)
    """

    ht_gate_threshold: float = float("+inf")

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add gating CLI arguments."""
        group = parser.add_argument_group("Data Quality")
        group.add_argument(
            "--ht-gate-threshold",
            type=float,
            default=float("+inf"),
            metavar="VALUE",
            help="Gating threshold; data above this is gated (default: inf = disabled)",
        )

    @classmethod
    def get_cli_arg_names(cls) -> Set[str]:
        return {"ht_gate_threshold"}

    @classmethod
    def process_cli_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert gating CLI args to field values."""
        if hasattr(args, "ht_gate_threshold") and args.ht_gate_threshold is not None:
            return {"ht_gate_threshold": args.ht_gate_threshold}
        return {}


# =============================================================================
# Latency Tracking Options
# =============================================================================


@dataclass(kw_only=True)
class LatencyTrackingOptionsMixin:
    """Mixin for latency tracking options.

    Fields:
        whiten_latency: Enable final whitening latency tracking
        detailed_latency: Enable per-stage latency tracking
    """

    whiten_latency: bool = False
    detailed_latency: bool = False

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add latency tracking CLI arguments."""
        group = parser.add_argument_group("Latency Tracking")
        group.add_argument(
            "--whiten-latency",
            action="store_true",
            default=False,
            help="Enable whitening latency tracking",
        )
        group.add_argument(
            "--detailed-latency",
            action="store_true",
            default=False,
            help="Enable per-stage detailed latency tracking",
        )

    @classmethod
    def get_cli_arg_names(cls) -> Set[str]:
        return {"whiten_latency", "detailed_latency"}

    @classmethod
    def process_cli_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert latency tracking CLI args to field values."""
        result: Dict[str, Any] = {}
        if hasattr(args, "whiten_latency"):
            result["whiten_latency"] = args.whiten_latency
        if hasattr(args, "detailed_latency"):
            result["detailed_latency"] = args.detailed_latency
        return result


# =============================================================================
# Zero Latency Options
# =============================================================================


@dataclass(kw_only=True)
class ZeroLatencyOptionsMixin:
    """Mixin for zero-latency AFIR path options.

    Fields:
        drift_correction: Enable drift correction stage
    """

    drift_correction: bool = True

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add zero-latency options CLI arguments."""
        parser.add_argument(
            "--no-drift-correction",
            action="store_false",
            dest="drift_correction",
            default=True,
            help="Disable drift correction in zero-latency mode",
        )

    @classmethod
    def get_cli_arg_names(cls) -> Set[str]:
        return {"drift_correction"}

    @classmethod
    def process_cli_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert zero-latency CLI args to field values."""
        result: Dict[str, Any] = {}
        if hasattr(args, "drift_correction"):
            result["drift_correction"] = args.drift_correction
        return result


# =============================================================================
# Highpass Filter Options
# =============================================================================


@dataclass(kw_only=True)
class HighpassFilterOptionsMixin:
    """Mixin for highpass filter option.

    Fields:
        highpass_filter: Enable 8Hz Butterworth highpass filter
    """

    highpass_filter: bool = False

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add highpass filter CLI argument."""
        parser.add_argument(
            "--highpass-filter",
            action="store_true",
            default=False,
            help="Enable 8Hz Butterworth highpass filter before whitening",
        )

    @classmethod
    def get_cli_arg_names(cls) -> Set[str]:
        return {"highpass_filter"}

    @classmethod
    def process_cli_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert highpass filter CLI arg to field value."""
        if hasattr(args, "highpass_filter"):
            return {"highpass_filter": args.highpass_filter}
        return {}


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Protocol
    "CLIMixinProtocol",
    # Sample rates
    "InputSampleRateOptionsMixin",
    "WhitenSampleRateOptionsMixin",
    # PSD
    "PSDOptionsMixin",
    # Gating
    "GatingOptionsMixin",
    # Latency
    "LatencyTrackingOptionsMixin",
    # Zero-latency
    "ZeroLatencyOptionsMixin",
    # Highpass
    "HighpassFilterOptionsMixin",
]
