"""Tests for condition_v2 module.

These tests verify the composed transform base class, registry,
CLI parsing, and concrete condition implementations.
"""

from __future__ import annotations

import argparse

import pytest
from sgnts.compose import TSComposedTransformElement

from sgnligo.transforms.condition_v2 import (
    Condition,
    ComposedTransformBase,
    StandardCondition,
    ZeroLatencyCondition,
    get_composed_transform_class,
    list_composed_transform_types,
    register_composed_transform,
)
from sgnligo.transforms.condition_v2.cli import (
    build_condition_cli_parser,
    check_condition_help_options,
    format_condition_help,
    format_condition_list,
    namespace_to_condition_kwargs,
)


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
        args = parser.parse_args([
            "--condition-type", "standard",
            "--ifos", "H1",
            "--input-sample-rate", "16384",
        ])
        assert args.condition_type == "standard"

    def test_namespace_to_condition_kwargs(self):
        """Test converting namespace to kwargs."""
        parser = build_condition_cli_parser()
        args = parser.parse_args([
            "--condition-type", "standard",
            "--ifos", "H1",
            "--input-sample-rate", "16384",
            "--whiten-sample-rate", "4096",
        ])
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
                "--condition-type", "standard",
                "--ifos", "H1",
                "--input-sample-rate", "16384",
            ]
        )
        assert cond.condition_type == "standard"
        assert "H1" in cond.snks

    def test_from_argv_zero_latency(self):
        """Test creating zero-latency Condition from argv."""
        cond = Condition.from_argv(
            name="test",
            argv=[
                "--condition-type", "zero-latency",
                "--ifos", "H1",
                "--input-sample-rate", "16384",
                "--no-drift-correction",
            ]
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
