"""Tests for condition_v2 module.

These tests verify the composed transform base class, registry,
CLI parsing, and concrete condition implementations.
"""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import ClassVar

import pytest
from sgnts.compose import TSCompose, TSComposedTransformElement

from sgnligo.transforms.condition_v2 import (
    Condition,
    StandardCondition,
    ZeroLatencyCondition,
    get_composed_transform_class,
    get_composed_transform_registry,
    list_composed_transform_types,
)
from sgnligo.transforms.condition_v2.cli import (
    add_condition_options_to_parser,
    build_condition_cli_parser,
    check_condition_help_options,
    format_condition_help,
    format_condition_list,
    get_transform_optional_fields,
    namespace_to_condition_kwargs,
)
from sgnligo.transforms.condition_v2.composed_base import ComposedTransformBase

PATH_DATA = pathlib.Path(__file__).parent / "data"
PATH_PSD = PATH_DATA / "H1L1-GSTLAL-MEDIAN.xml.gz"

# --- Registry Tests ---


class TestComposedRegistry:
    """Tests for the composed transform registry."""

    def test_list_composed_transform_types(self):
        """Test listing all registered transform types."""
        types = list_composed_transform_types()
        assert "standard" in types
        assert "zero-latency" in types

    def test_get_composed_transform_class_standard(self):
        """Test getting the standard transform class."""
        cls = get_composed_transform_class("standard")
        assert cls is StandardCondition

    def test_get_composed_transform_class_zero_latency(self):
        """Test getting the zero-latency transform class."""
        cls = get_composed_transform_class("zero-latency")
        assert cls is ZeroLatencyCondition

    def test_get_composed_transform_class_unknown(self):
        """Test that unknown type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown transform type"):
            get_composed_transform_class("nonexistent")


# --- StandardCondition Tests ---


class TestStandardCondition:
    """Tests for StandardCondition."""

    def test_basic_instantiation(self):
        """Test basic instantiation with required parameters."""
        cond = StandardCondition(
            name="test",
            ifos=["H1"],
            input_sample_rate=16384,
            whiten_sample_rate=2048,
        )
        assert cond.name == "test"
        assert cond.ifos == ["H1"]
        assert cond.input_sample_rate == 16384
        assert cond.whiten_sample_rate == 2048

    def test_snks_property(self):
        """Test that sink pads are created for each IFO."""
        cond = StandardCondition(
            name="test",
            ifos=["H1", "L1"],
            input_sample_rate=16384,
        )
        assert "H1" in cond.snks
        assert "L1" in cond.snks
        assert len(cond.snks) == 2

    def test_srcs_property(self):
        """Test that source pads include strain and spectrum for each IFO."""
        cond = StandardCondition(
            name="test",
            ifos=["H1", "L1"],
            input_sample_rate=16384,
        )
        # Should have strain and spectrum for each IFO
        assert "H1" in cond.srcs
        assert "L1" in cond.srcs
        assert "spectrum_H1" in cond.srcs
        assert "spectrum_L1" in cond.srcs

    def test_element_property(self):
        """Test that .element returns TSComposedTransformElement."""
        cond = StandardCondition(
            name="test",
            ifos=["H1"],
            input_sample_rate=16384,
        )
        assert isinstance(cond.element, TSComposedTransformElement)
        assert cond.element is cond._composed

    def test_class_metadata(self):
        """Test transform_type and description class variables."""
        assert StandardCondition.transform_type == "standard"
        assert "whitening" in StandardCondition.description.lower()

    def test_validation_empty_ifos(self):
        """Test validation catches empty IFOs list."""
        with pytest.raises(ValueError, match="at least one IFO"):
            StandardCondition(
                name="test",
                ifos=[],
                input_sample_rate=16384,
            )

    def test_default_whiten_sample_rate(self):
        """Test default whiten_sample_rate is 2048."""
        cond = StandardCondition(
            name="test",
            ifos=["H1"],
            input_sample_rate=16384,
        )
        assert cond.whiten_sample_rate == 2048


# --- ZeroLatencyCondition Tests ---


class TestZeroLatencyCondition:
    """Tests for ZeroLatencyCondition."""

    def test_basic_instantiation(self):
        """Test basic instantiation without drift correction."""
        cond = ZeroLatencyCondition(
            name="test",
            ifos=["H1"],
            input_sample_rate=16384,
            whiten_sample_rate=2048,
            drift_correction=False,
        )
        assert cond.name == "test"
        assert cond.ifos == ["H1"]
        assert cond.drift_correction is False

    def test_snks_property(self):
        """Test that sink pads are created for each IFO."""
        cond = ZeroLatencyCondition(
            name="test",
            ifos=["H1", "L1"],
            input_sample_rate=16384,
            drift_correction=False,
        )
        assert "H1" in cond.snks
        assert "L1" in cond.snks

    def test_srcs_include_spectrum(self):
        """Test that source pads include spectrum (multilink pattern)."""
        cond = ZeroLatencyCondition(
            name="test",
            ifos=["H1"],
            input_sample_rate=16384,
            drift_correction=False,
        )
        # Should have both strain and spectrum
        assert "H1" in cond.srcs
        assert "spectrum_H1" in cond.srcs

    def test_class_metadata(self):
        """Test transform_type and description class variables."""
        assert ZeroLatencyCondition.transform_type == "zero-latency"
        assert "afir" in ZeroLatencyCondition.description.lower()

    def test_validation_empty_ifos(self):
        """Test validation catches empty IFOs list."""
        with pytest.raises(ValueError, match="at least one IFO"):
            ZeroLatencyCondition(
                name="test",
                ifos=[],
                input_sample_rate=16384,
                drift_correction=False,
            )

    def test_validation_upsampling_not_allowed(self):
        """Test validation catches upsampling attempt."""
        with pytest.raises(ValueError, match="requires downsampling"):
            ZeroLatencyCondition(
                name="test",
                ifos=["H1"],
                input_sample_rate=2048,
                whiten_sample_rate=4096,  # Upsampling!
                drift_correction=False,
            )


# --- Condition Dispatcher Tests ---


class TestConditionDispatcher:
    """Tests for the Condition dispatcher class."""

    def test_dispatch_to_standard(self):
        """Test dispatching to StandardCondition."""
        cond = Condition(
            condition_type="standard",
            name="test",
            ifos=["H1"],
            input_sample_rate=16384,
        )
        assert cond.condition_type == "standard"
        assert isinstance(cond._inner, StandardCondition)

    def test_dispatch_to_zero_latency(self):
        """Test dispatching to ZeroLatencyCondition."""
        cond = Condition(
            condition_type="zero-latency",
            name="test",
            ifos=["H1"],
            input_sample_rate=16384,
            drift_correction=False,
        )
        assert cond.condition_type == "zero-latency"
        assert isinstance(cond._inner, ZeroLatencyCondition)

    def test_element_delegates_to_inner(self):
        """Test that .element returns inner element."""
        cond = Condition(
            condition_type="standard",
            name="test",
            ifos=["H1"],
            input_sample_rate=16384,
        )
        assert cond.element is cond._inner.element

    def test_srcs_delegates_to_inner(self):
        """Test that .srcs returns inner srcs."""
        cond = Condition(
            condition_type="standard",
            name="test",
            ifos=["H1"],
            input_sample_rate=16384,
        )
        assert cond.srcs == cond._inner.srcs

    def test_snks_delegates_to_inner(self):
        """Test that .snks returns inner snks."""
        cond = Condition(
            condition_type="standard",
            name="test",
            ifos=["H1"],
            input_sample_rate=16384,
        )
        assert cond.snks == cond._inner.snks

    def test_list_conditions(self):
        """Test static list_conditions method."""
        conditions = Condition.list_conditions()
        assert "standard" in conditions
        assert "zero-latency" in conditions

    def test_get_condition_class(self):
        """Test static get_condition_class method."""
        cls = Condition.get_condition_class("standard")
        assert cls is StandardCondition


# --- CLI Tests ---


class TestCLI:
    """Tests for CLI parsing."""

    def test_build_condition_cli_parser(self):
        """Test that CLI parser is built successfully."""
        parser = build_condition_cli_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_parser_has_condition_type(self):
        """Test that parser has --condition-type argument."""
        parser = build_condition_cli_parser()
        # Parse with required args
        args = parser.parse_args(
            [
                "--condition-type",
                "standard",
                "--ifos",
                "H1",
                "--input-sample-rate",
                "16384",
            ]
        )
        assert args.condition_type == "standard"

    def test_namespace_to_condition_kwargs(self):
        """Test converting namespace to kwargs."""
        parser = build_condition_cli_parser()
        args = parser.parse_args(
            [
                "--condition-type",
                "standard",
                "--ifos",
                "H1",
                "--input-sample-rate",
                "16384",
                "--whiten-sample-rate",
                "4096",
            ]
        )
        kwargs = namespace_to_condition_kwargs(args)
        assert kwargs["condition_type"] == "standard"
        assert kwargs["ifos"] == ["H1"]
        assert kwargs["input_sample_rate"] == 16384
        assert kwargs["whiten_sample_rate"] == 4096

    def test_from_argv_standard(self):
        """Test creating Condition from argv."""
        cond = Condition.from_argv(
            name="test",
            argv=[
                "--condition-type",
                "standard",
                "--ifos",
                "H1",
                "--input-sample-rate",
                "16384",
            ],
        )
        assert cond.condition_type == "standard"
        assert "H1" in cond.snks

    def test_from_argv_zero_latency(self):
        """Test creating zero-latency Condition from argv."""
        cond = Condition.from_argv(
            name="test",
            argv=[
                "--condition-type",
                "zero-latency",
                "--ifos",
                "H1",
                "--input-sample-rate",
                "16384",
                "--no-drift-correction",
            ],
        )
        assert cond.condition_type == "zero-latency"

    def test_format_condition_list(self):
        """Test format_condition_list output."""
        output = format_condition_list()
        assert "standard" in output
        assert "zero-latency" in output
        assert "Available condition types" in output

    def test_format_condition_help(self):
        """Test format_condition_help output."""
        output = format_condition_help("standard")
        assert "standard" in output
        assert "--input-sample-rate" in output
        assert "--ifos" in output

    def test_check_condition_help_options_list(self):
        """Test --list-conditions handling."""
        # Should return True and print list
        result = check_condition_help_options(["--list-conditions"])
        assert result is True

    def test_check_condition_help_options_help(self):
        """Test --help-condition handling."""
        result = check_condition_help_options(["--help-condition", "standard"])
        assert result is True

    def test_check_condition_help_options_none(self):
        """Test no help options."""
        result = check_condition_help_options(["--condition-type", "standard"])
        assert result is False


# --- ComposedTransformBase Tests ---


class TestComposedTransformBase:
    """Tests for ComposedTransformBase functionality."""

    def test_private_attrs_raise_attribute_error(self):
        """Test that private attributes don't delegate."""
        cond = StandardCondition(
            name="test",
            ifos=["H1"],
            input_sample_rate=16384,
        )
        with pytest.raises(AttributeError, match="no attribute '_nonexistent'"):
            _ = cond._nonexistent

    def test_getattr_delegates_to_composed(self):
        """Test that unknown attributes are delegated to _composed."""
        cond = StandardCondition(
            name="test",
            ifos=["H1"],
            input_sample_rate=16384,
        )
        # TSComposedTransformElement has these attributes
        assert hasattr(cond, "internal_elements")
        assert hasattr(cond, "internal_links")

    def test_repr_excludes_composed(self):
        """Test that _composed is excluded from repr."""
        cond = StandardCondition(
            name="test",
            ifos=["H1"],
            input_sample_rate=16384,
        )
        repr_str = repr(cond)
        assert "name='test'" in repr_str
        assert "_composed" not in repr_str


# --- Additional Registry Tests ---


class TestComposedRegistryExtended:
    """Additional tests for composed_registry.py coverage."""

    def test_get_composed_transform_registry(self):
        """Test get_composed_transform_registry returns a copy."""
        registry = get_composed_transform_registry()
        assert "standard" in registry
        assert "zero-latency" in registry
        # Should be a copy, not the original
        assert registry is not get_composed_transform_registry()

    def test_register_empty_transform_type_raises(self) -> None:
        """Test that registering class with empty transform_type raises."""
        from sgnligo.transforms.condition_v2.composed_registry import (
            register_composed_transform,
        )

        @dataclass(kw_only=True)
        class BadTransform(ComposedTransformBase):
            transform_type: ClassVar[str] = ""  # Empty!
            description: ClassVar[str] = "Bad"

            def _build(self):
                return TSCompose().as_transform(name=self.name)

        with pytest.raises(ValueError, match="must define transform_type"):
            register_composed_transform(BadTransform)

    def test_register_duplicate_transform_type_raises(self) -> None:
        """Test that registering duplicate transform_type raises."""
        from sgnligo.transforms.condition_v2.composed_registry import (
            register_composed_transform,
        )

        @dataclass(kw_only=True)
        class DuplicateTransform(ComposedTransformBase):
            transform_type: ClassVar[str] = "standard"  # Already registered!
            description: ClassVar[str] = "Duplicate"

            def _build(self):
                return TSCompose().as_transform(name=self.name)

        with pytest.raises(ValueError, match="already registered"):
            register_composed_transform(DuplicateTransform)


# --- Additional CLI Tests ---


class TestCLIExtended:
    """Additional tests for cli.py coverage."""

    def test_get_transform_optional_fields_with_default_factory(self) -> None:
        """Test get_transform_optional_fields with a field using default_factory."""
        from dataclasses import field as dataclass_field

        # Create a test class with default_factory
        @dataclass(kw_only=True)
        class TestTransformWithFactory(ComposedTransformBase):
            transform_type: ClassVar[str] = ""
            description: ClassVar[str] = "Test"
            tags: list = dataclass_field(default_factory=list)

            def _build(self):
                return TSCompose().as_transform(name=self.name)

        optional = get_transform_optional_fields(TestTransformWithFactory)
        # Should have called default_factory() for 'tags' field
        assert "tags" in optional
        assert optional["tags"] == []  # default_factory=list returns []

    def test_check_help_options_unknown_condition(self):
        """Test --help-condition with unknown condition type."""
        result = check_condition_help_options(["--help-condition", "nonexistent"])
        assert result is True

    def test_check_help_options_missing_argument(self):
        """Test --help-condition without argument."""
        result = check_condition_help_options(["--help-condition"])
        assert result is True

    def test_check_help_options_default_argv(self, monkeypatch):
        """Test check_condition_help_options with default argv (None)."""
        import sys

        # Mock sys.argv to not include help options
        monkeypatch.setattr(sys, "argv", ["prog", "--condition-type", "standard"])
        result = check_condition_help_options(None)
        assert result is False

    def test_get_transform_optional_fields_with_factory(self):
        """Test get_transform_optional_fields with default_factory fields."""
        # StandardCondition has fields with default_factory (like list fields)
        # This tests line 76 in cli.py
        optional = get_transform_optional_fields(ZeroLatencyCondition)
        # Should include fields with defaults
        assert "whiten_sample_rate" in optional
        assert "drift_correction" in optional

    def test_add_condition_options_to_parser(self):
        """Test add_condition_options_to_parser function."""
        parser = argparse.ArgumentParser()
        add_condition_options_to_parser(parser)
        # Should be able to parse condition options
        args = parser.parse_args(["--ifos", "H1", "--input-sample-rate", "16384"])
        assert args.ifos == ["H1"]
        assert args.input_sample_rate == 16384

    def test_add_condition_options_with_condition_type(self):
        """Test add_condition_options_to_parser with include_condition_type."""
        parser = argparse.ArgumentParser()
        add_condition_options_to_parser(parser, include_condition_type=True)
        args = parser.parse_args(
            [
                "--condition-type",
                "standard",
                "--ifos",
                "H1",
                "--input-sample-rate",
                "16384",
            ]
        )
        assert args.condition_type == "standard"

    def test_build_parser_with_overlapping_mixins(self) -> None:
        """Test that overlapping mixins are handled correctly.

        This tests the 'continue' branch when a mixin's args overlap
        with already-added args (lines 286, 368 in cli.py).
        """
        from sgnligo.transforms.condition_v2.cli_mixins import (
            InputSampleRateOptionsMixin,
        )
        from sgnligo.transforms.condition_v2.composed_registry import (
            _COMPOSED_TRANSFORM_REGISTRY,
        )

        # Create a mixin that overlaps with InputSampleRateOptionsMixin
        @dataclass(kw_only=True)
        class OverlappingMixin:
            input_sample_rate: int  # Same field as InputSampleRateOptionsMixin

            @classmethod
            def add_cli_arguments(cls, parser):
                parser.add_argument("--input-sample-rate", type=int)

            @classmethod
            def get_cli_arg_names(cls):
                return {"input_sample_rate"}  # Overlaps!

            @classmethod
            def process_cli_args(cls, args):
                return {}

        # Create a transform using both mixins (overlap)
        @dataclass(kw_only=True)
        class OverlapTransform(
            ComposedTransformBase,
            InputSampleRateOptionsMixin,
            OverlappingMixin,
        ):
            transform_type: ClassVar[str] = "_test_overlap"
            description: ClassVar[str] = "Test overlap"

            def _build(self):
                return TSCompose().as_transform(name=self.name)

        # Temporarily register
        _COMPOSED_TRANSFORM_REGISTRY["_test_overlap"] = OverlapTransform
        try:
            # Build parser - should skip the overlapping mixin
            parser = build_condition_cli_parser()
            # Should not raise, and should have the arg only once
            args = parser.parse_args(
                [
                    "--condition-type",
                    "standard",
                    "--ifos",
                    "H1",
                    "--input-sample-rate",
                    "16384",
                ]
            )
            assert args.input_sample_rate == 16384

            # Also test add_condition_options_to_parser
            parser2 = argparse.ArgumentParser()
            add_condition_options_to_parser(parser2)
            args2 = parser2.parse_args(["--ifos", "H1", "--input-sample-rate", "16384"])
            assert args2.input_sample_rate == 16384
        finally:
            # Clean up
            del _COMPOSED_TRANSFORM_REGISTRY["_test_overlap"]

    def test_parser_skips_incomplete_mixin(self) -> None:
        """Test that mixins without get_cli_arg_names are skipped.

        This tests line 354 in cli.py where we skip bases that have
        add_cli_arguments but not get_cli_arg_names in __dict__.
        """
        from sgnligo.transforms.condition_v2.composed_registry import (
            _COMPOSED_TRANSFORM_REGISTRY,
        )

        # Create a mixin with add_cli_arguments but NOT get_cli_arg_names
        @dataclass(kw_only=True)
        class IncompleteMixin:
            @classmethod
            def add_cli_arguments(cls, parser):
                pass

            # Note: no get_cli_arg_names method!

        # Create a transform using this incomplete mixin
        @dataclass(kw_only=True)
        class IncompleteTransform(ComposedTransformBase, IncompleteMixin):
            transform_type: ClassVar[str] = "_test_incomplete"
            description: ClassVar[str] = "Test incomplete"
            ifos: list

            def _build(self):
                from sgnts.sources import FakeSeriesSource

                compose = TSCompose()
                for ifo in self.ifos:
                    src = FakeSeriesSource(
                        name=f"{self.name}_{ifo}",
                        source_pad_names=(ifo,),
                        sample_rate=16384,
                        t0=0,
                        end=1,
                    )
                    compose.insert(src)
                return compose.as_source(name=self.name)

        # Temporarily register
        _COMPOSED_TRANSFORM_REGISTRY["_test_incomplete"] = IncompleteTransform
        try:
            # Build parser - should skip IncompleteMixin (no get_cli_arg_names)
            parser = build_condition_cli_parser()
            # Should not raise
            assert parser is not None

            # Also test add_condition_options_to_parser
            parser2 = argparse.ArgumentParser()
            add_condition_options_to_parser(parser2)
            assert parser2 is not None
        finally:
            # Clean up
            del _COMPOSED_TRANSFORM_REGISTRY["_test_incomplete"]

    def test_namespace_to_condition_kwargs_with_override(self):
        """Test namespace_to_condition_kwargs with condition_type override."""
        parser = build_condition_cli_parser()
        args = parser.parse_args(
            [
                "--condition-type",
                "standard",
                "--ifos",
                "H1",
                "--input-sample-rate",
                "16384",
            ]
        )
        # Override condition type
        kwargs = namespace_to_condition_kwargs(args, condition_type="zero-latency")
        assert kwargs["condition_type"] == "zero-latency"

    def test_namespace_to_condition_kwargs_default_type(self):
        """Test namespace_to_condition_kwargs defaults to standard."""
        # Create namespace without condition_type
        args = argparse.Namespace(ifos=["H1"], input_sample_rate=16384)
        kwargs = namespace_to_condition_kwargs(args)
        assert kwargs["condition_type"] == "standard"


# --- Additional Condition Dispatcher Tests ---


class TestConditionDispatcherExtended:
    """Additional tests for condition.py coverage."""

    def test_getattr_delegates_to_inner(self):
        """Test __getattr__ delegates to inner transform."""
        cond = Condition(
            condition_type="standard",
            name="test",
            ifos=["H1"],
            input_sample_rate=16384,
        )
        # Access attribute that exists on inner
        assert hasattr(cond, "internal_elements")

    def test_getattr_raises_for_private(self):
        """Test __getattr__ raises for private attributes."""
        cond = Condition(
            condition_type="standard",
            name="test",
            ifos=["H1"],
            input_sample_rate=16384,
        )
        with pytest.raises(AttributeError):
            _ = cond._nonexistent

    def test_create_cli_parser(self):
        """Test Condition.create_cli_parser class method."""
        parser = Condition.create_cli_parser()
        assert isinstance(parser, argparse.ArgumentParser)
        # Should have condition-type argument
        args = parser.parse_args(
            [
                "--condition-type",
                "standard",
                "--ifos",
                "H1",
                "--input-sample-rate",
                "16384",
            ]
        )
        assert args.condition_type == "standard"

    def test_create_cli_parser_with_prog(self):
        """Test Condition.create_cli_parser with custom prog."""
        parser = Condition.create_cli_parser(prog="myapp", description="My app")
        assert parser.prog == "myapp"

    def test_from_parser(self):
        """Test Condition.from_parser class method."""
        parser = Condition.create_cli_parser()
        parser.add_argument("--extra", default="value")
        cond, args = Condition.from_parser(
            parser,
            name="test",
            argv=[
                "--condition-type",
                "standard",
                "--ifos",
                "H1",
                "--input-sample-rate",
                "16384",
            ],
        )
        assert cond.condition_type == "standard"
        assert args.extra == "value"

    def test_from_argv_with_help_exits(self, monkeypatch):
        """Test from_argv exits when help option is present."""

        class ExitCalled(Exception):
            pass

        def fake_exit(code):
            raise ExitCalled(code)

        monkeypatch.setattr("sys.exit", fake_exit)
        with pytest.raises(ExitCalled) as exc_info:
            Condition.from_argv(name="test", argv=["--list-conditions"])
        assert exc_info.value.args[0] == 0

    def test_from_parser_with_help_exits(self, monkeypatch):
        """Test from_parser exits when help option is present."""

        class ExitCalled(Exception):
            pass

        def fake_exit(code):
            raise ExitCalled(code)

        monkeypatch.setattr("sys.exit", fake_exit)
        parser = Condition.create_cli_parser()
        with pytest.raises(ExitCalled) as exc_info:
            Condition.from_parser(parser, name="test", argv=["--list-conditions"])
        assert exc_info.value.args[0] == 0


# --- StandardCondition Extended Tests ---


class TestStandardConditionExtended:
    """Additional tests for StandardCondition coverage."""

    def test_validation_no_psd_no_tracking(self):
        """Test validation when no reference_psd and track_psd=False."""
        with pytest.raises(ValueError, match="track_psd"):
            StandardCondition(
                name="test",
                ifos=["H1"],
                input_sample_rate=16384,
                track_psd=False,
                reference_psd=None,
            )

    def test_with_gating(self):
        """Test StandardCondition with gating enabled."""
        cond = StandardCondition(
            name="test",
            ifos=["H1"],
            input_sample_rate=16384,
            ht_gate_threshold=50.0,
        )
        # Should have threshold element in internal elements
        element_names = [e.name for e in cond.internal_elements]
        assert any("threshold" in name for name in element_names)

    @pytest.mark.xfail(reason="Latency is TransformElement, not TSTransform")
    def test_with_latency_tracking(self):
        """Test StandardCondition with latency tracking enabled."""
        cond = StandardCondition(
            name="test",
            ifos=["H1"],
            input_sample_rate=16384,
            whiten_latency=True,
        )
        # Should have latency element and latency output pad
        assert "H1_latency" in cond.srcs
        element_names = [e.name for e in cond.internal_elements]
        assert any("latency" in name for name in element_names)

    def test_with_highpass_filter(self):
        """Test StandardCondition with highpass filter."""
        cond = StandardCondition(
            name="test",
            ifos=["H1"],
            input_sample_rate=16384,
            highpass_filter=True,
        )
        assert cond.highpass_filter is True


# --- ZeroLatencyCondition Extended Tests ---


class TestZeroLatencyConditionExtended:
    """Additional tests for ZeroLatencyCondition coverage."""

    def test_validation_no_psd_no_tracking(self):
        """Test validation when no reference_psd and track_psd=False."""
        with pytest.raises(ValueError, match="track_psd"):
            ZeroLatencyCondition(
                name="test",
                ifos=["H1"],
                input_sample_rate=16384,
                track_psd=False,
                reference_psd=None,
                drift_correction=False,
            )

    def test_same_sample_rate_no_resampling(self):
        """Test ZeroLatencyCondition without resampling."""
        cond = ZeroLatencyCondition(
            name="test",
            ifos=["H1"],
            input_sample_rate=2048,
            whiten_sample_rate=2048,
            drift_correction=False,
        )
        # Should not have resampler
        element_names = [e.name for e in cond.internal_elements]
        assert not any("resamp" in name for name in element_names)

    def test_with_resampling(self):
        """Test ZeroLatencyCondition with resampling."""
        cond = ZeroLatencyCondition(
            name="test",
            ifos=["H1"],
            input_sample_rate=16384,
            whiten_sample_rate=2048,
            drift_correction=False,
        )
        # Should have resampler
        element_names = [e.name for e in cond.internal_elements]
        assert any("resamp" in name for name in element_names)

    def test_with_gating(self):
        """Test ZeroLatencyCondition with gating enabled."""
        cond = ZeroLatencyCondition(
            name="test",
            ifos=["H1"],
            input_sample_rate=16384,
            drift_correction=False,
            ht_gate_threshold=50.0,
        )
        element_names = [e.name for e in cond.internal_elements]
        assert any("threshold" in name for name in element_names)

    @pytest.mark.xfail(reason="Latency is TransformElement, not TSTransform")
    def test_with_whiten_latency(self):
        """Test ZeroLatencyCondition with whiten_latency tracking."""
        cond = ZeroLatencyCondition(
            name="test",
            ifos=["H1"],
            input_sample_rate=16384,
            drift_correction=False,
            whiten_latency=True,
        )
        assert "H1_latency" in cond.srcs

    @pytest.mark.xfail(reason="Latency is TransformElement, not TSTransform")
    def test_with_detailed_latency(self):
        """Test ZeroLatencyCondition with detailed_latency tracking."""
        cond = ZeroLatencyCondition(
            name="test",
            ifos=["H1"],
            input_sample_rate=16384,
            drift_correction=False,
            detailed_latency=True,
        )
        # Should have detailed latency pads
        # Note: whiten_latency has different pad names than detailed_latency
        element_names = [e.name for e in cond.internal_elements]
        assert any("lat" in name for name in element_names)

    def test_multiple_ifos(self):
        """Test ZeroLatencyCondition with multiple IFOs."""
        cond = ZeroLatencyCondition(
            name="test",
            ifos=["H1", "L1"],
            input_sample_rate=16384,
            drift_correction=False,
        )
        assert "H1" in cond.snks
        assert "L1" in cond.snks
        assert "H1" in cond.srcs
        assert "L1" in cond.srcs
        assert "spectrum_H1" in cond.srcs
        assert "spectrum_L1" in cond.srcs

    def test_drift_correction_with_invalid_psd_file(self, capsys):
        """Test drift correction with non-existent PSD file prints warning.

        Note: The warning is printed but the Whiten element also uses the
        reference_psd and fails to load it, causing overall creation to fail.
        """
        with pytest.raises(FileNotFoundError):
            ZeroLatencyCondition(
                name="test",
                ifos=["H1"],
                input_sample_rate=16384,
                drift_correction=True,
                reference_psd="/nonexistent/path/to/psd.xml",
            )
        # Warning should have been printed before the failure
        captured = capsys.readouterr()
        assert "Warning" in captured.out

    def test_load_reference_psds_caching(self):
        """Test that _load_reference_psds caches results."""
        cond = ZeroLatencyCondition(
            name="test",
            ifos=["H1"],
            input_sample_rate=16384,
            drift_correction=False,
        )
        # First call populates cache
        result1 = cond._load_reference_psds()
        # Second call should return cached result
        result2 = cond._load_reference_psds()
        assert result1 is result2

    def test_with_drift_correction(self):
        """Test ZeroLatencyCondition with drift correction enabled."""
        cond = ZeroLatencyCondition(
            name="test",
            ifos=["H1"],
            input_sample_rate=16384,
            whiten_sample_rate=2048,
            drift_correction=True,
            reference_psd=PATH_PSD.as_posix(),
        )
        # Should have drift correction elements (DriftCorrectionKernel + AFIR)
        assert "H1" in cond.snks
        assert "H1" in cond.srcs
        assert "spectrum_H1" in cond.srcs

    @pytest.mark.xfail(reason="Latency is TransformElement, not TSTransform")
    def test_with_drift_correction_and_detailed_latency(self):
        """Test ZeroLatencyCondition with drift correction and detailed latency.

        This covers lines 315-331 (detailed latency after drift correction).
        """
        cond = ZeroLatencyCondition(
            name="test",
            ifos=["H1"],
            input_sample_rate=16384,
            whiten_sample_rate=2048,
            drift_correction=True,
            reference_psd=PATH_PSD.as_posix(),
            detailed_latency=True,
            whiten_latency=False,  # Required to hit lines 315-331
        )
        # Should have drift latency output
        assert "H1" in cond.snks
        assert "H1" in cond.srcs
        # Should have detailed latency pads
        assert "H1_drift_latency" in cond.srcs


# --- ComposedTransformBase Extended Tests ---


class TestComposedTransformBaseExtended:
    """Additional tests for ComposedTransformBase coverage."""

    def test_add_latency_tracking_with_interval(self) -> None:
        """Test _add_latency_tracking helper method."""
        from sgnts.sources import FakeSeriesSource

        # Create a custom transform that uses _add_latency_tracking
        @dataclass(kw_only=True)
        class LatencyTrackingTransform(ComposedTransformBase):
            transform_type: ClassVar[str] = ""
            description: ClassVar[str] = "Test latency tracking"
            ifos: list

            def _build(self) -> TSComposedTransformElement:
                compose = TSCompose()
                for ifo in self.ifos:
                    src = FakeSeriesSource(
                        name=f"{self.name}_{ifo}",
                        source_pad_names=(ifo,),
                        rate=16384,
                        t0=0,
                        end=10,
                        signal_type="white",
                    )
                    compose.insert(src)
                    # Call _add_latency_tracking to cover lines 167-189
                    self._add_latency_tracking(compose, ifo, src, ifo)
                return compose.as_source(
                    name=self.name,
                    also_expose_source_pads=self._also_expose_pads,
                )

        # Create with latency_interval set (not None) to exercise the code
        transform = LatencyTrackingTransform(
            name="test_latency",
            ifos=["H1"],
            latency_interval=1.0,
        )

        # Verify the latency pad was created
        assert "H1_latency" in transform.srcs
        # Verify original pad is also exposed (multilink)
        assert "H1" in transform.srcs
        # Verify _also_expose_pads was populated
        assert len(transform._also_expose_pads) == 1

    def test_add_latency_tracking_without_interval(self) -> None:
        """Test _add_latency_tracking with latency_interval=None (early return)."""
        from sgnts.sources import FakeSeriesSource

        # Create a custom transform that uses _add_latency_tracking
        @dataclass(kw_only=True)
        class NoLatencyTransform(ComposedTransformBase):
            transform_type: ClassVar[str] = ""
            description: ClassVar[str] = "Test no latency tracking"
            ifos: list

            def _build(self) -> TSComposedTransformElement:
                compose = TSCompose()
                for ifo in self.ifos:
                    src = FakeSeriesSource(
                        name=f"{self.name}_{ifo}",
                        source_pad_names=(ifo,),
                        rate=16384,
                        t0=0,
                        end=10,
                        signal_type="white",
                    )
                    compose.insert(src)
                    # Call _add_latency_tracking - should early return (line 168)
                    self._add_latency_tracking(compose, ifo, src, ifo)
                return compose.as_source(name=self.name)

        # Create WITHOUT latency_interval (default is None) - hits line 168
        transform = NoLatencyTransform(
            name="test_no_latency",
            ifos=["H1"],
        )

        # Verify no latency pad was created (early return)
        assert "H1_latency" not in transform.srcs
        # Original pad should still exist
        assert "H1" in transform.srcs
        # _also_expose_pads should be empty
        assert len(transform._also_expose_pads) == 0

    def test_process_cli_args_with_latency_interval(self):
        """Test ComposedTransformBase.process_cli_args with latency_interval."""
        args = argparse.Namespace(condition_latency_interval=1.5)
        result = ComposedTransformBase.process_cli_args(args)
        assert result["latency_interval"] == 1.5

    def test_process_cli_args_without_latency_interval(self):
        """Test ComposedTransformBase.process_cli_args without latency_interval."""
        args = argparse.Namespace()  # No condition_latency_interval
        result = ComposedTransformBase.process_cli_args(args)
        assert result == {}

    def test_base_validate_is_noop(self) -> None:
        """Test that base _validate is a no-op (line 130 coverage)."""
        from sgnts.sources import FakeSeriesSource

        # Create a minimal subclass that doesn't override _validate
        @dataclass(kw_only=True)
        class MinimalTransform(ComposedTransformBase):
            transform_type: ClassVar[str] = ""  # Empty to avoid registration
            description: ClassVar[str] = "Minimal"
            ifos: list

            def _build(self) -> TSComposedTransformElement:
                compose = TSCompose()
                for ifo in self.ifos:
                    src = FakeSeriesSource(
                        name=f"{self.name}_{ifo}",
                        source_pad_names=(ifo,),
                        rate=16384,
                        t0=0,
                        end=10,
                        signal_type="white",
                    )
                    compose.insert(src)
                return compose.as_source(name=self.name)

        # Instantiate to trigger __post_init__ which calls _validate()
        # Should not raise - base _validate() is a pass (line 130)
        transform = MinimalTransform(name="test", ifos=["H1"])
        assert transform.srcs is not None


# --- CLI Mixins Tests ---


class TestCLIMixins:
    """Tests for cli_mixins.py coverage."""

    def test_input_sample_rate_process_cli_args_none(self):
        """Test InputSampleRateOptionsMixin returns empty when arg is None."""
        from sgnligo.transforms.condition_v2.cli_mixins import (
            InputSampleRateOptionsMixin,
        )

        args = argparse.Namespace()  # No input_sample_rate attribute
        result = InputSampleRateOptionsMixin.process_cli_args(args)
        assert result == {}

    def test_whiten_sample_rate_process_cli_args_none(self):
        """Test WhitenSampleRateOptionsMixin returns empty when arg is None."""
        from sgnligo.transforms.condition_v2.cli_mixins import (
            WhitenSampleRateOptionsMixin,
        )

        args = argparse.Namespace()  # No whiten_sample_rate attribute
        result = WhitenSampleRateOptionsMixin.process_cli_args(args)
        assert result == {}

    def test_psd_options_process_cli_args_partial(self):
        """Test PSDOptionsMixin process_cli_args with partial args."""
        from sgnligo.transforms.condition_v2.cli_mixins import PSDOptionsMixin

        args = argparse.Namespace(psd_fft_length=16, track_psd=True)
        # No reference_psd
        result = PSDOptionsMixin.process_cli_args(args)
        assert result["psd_fft_length"] == 16
        assert result["track_psd"] is True
        assert "reference_psd" not in result

    def test_psd_options_process_cli_args_with_reference_psd(self):
        """Test PSDOptionsMixin process_cli_args with reference_psd."""
        from sgnligo.transforms.condition_v2.cli_mixins import PSDOptionsMixin

        args = argparse.Namespace(
            psd_fft_length=16, track_psd=True, reference_psd="/path/to/psd.xml"
        )
        result = PSDOptionsMixin.process_cli_args(args)
        assert result["reference_psd"] == "/path/to/psd.xml"

    def test_gating_options_process_cli_args_none(self):
        """Test GatingOptionsMixin returns empty when arg is None."""
        from sgnligo.transforms.condition_v2.cli_mixins import GatingOptionsMixin

        args = argparse.Namespace()
        result = GatingOptionsMixin.process_cli_args(args)
        assert result == {}

    def test_highpass_options_process_cli_args_none(self):
        """Test HighpassFilterOptionsMixin returns empty when not present."""
        from sgnligo.transforms.condition_v2.cli_mixins import (
            HighpassFilterOptionsMixin,
        )

        args = argparse.Namespace()
        result = HighpassFilterOptionsMixin.process_cli_args(args)
        assert result == {}
