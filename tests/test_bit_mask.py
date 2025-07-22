#!/usr/bin/env python3
"""Test coverage for sgnligo.transforms.bit_mask module."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from sgnts.base import SeriesBuffer, TSFrame

from sgnligo.transforms.bit_mask import BitMask


class TestBitMask:
    """Test cases for BitMask transform."""

    def test_init_defaults(self):
        """Test initialization with bit mask."""
        bit_mask = BitMask(
            name="TestBitMask",
            bit_mask=0x3,  # Binary: 11
            source_pad_names=("output",),
            sink_pad_names=("input",),
        )
        assert bit_mask.bit_mask == 0x3

    def test_init_no_bit_mask(self):
        """Test initialization without bit mask raises assertion."""
        with pytest.raises(AssertionError):
            BitMask(
                name="TestBitMask",
                bit_mask=None,
                source_pad_names=("output",),
                sink_pad_names=("input",),
            )

    @patch("sgnligo.transforms.bit_mask.state_vector_on_off_bits")
    def test_init_calls_state_vector_on_off_bits(self, mock_state_vector):
        """Test that initialization calls state_vector_on_off_bits."""
        mock_state_vector.return_value = 0x5
        bit_mask = BitMask(
            name="TestBitMask",
            bit_mask=0x3,
            source_pad_names=("output",),
            sink_pad_names=("input",),
        )
        mock_state_vector.assert_called_once_with(0x3)
        assert bit_mask.bit_mask == 0x5

    def create_mock_buffer(self, data, offset=0, sample_rate=16, is_gap=False):
        """Create a mock SeriesBuffer."""
        if data is not None:
            # Create a real SeriesBuffer for non-gap buffers
            buffer = SeriesBuffer(
                offset=offset,
                sample_rate=sample_rate,
                data=np.array(data),
                shape=(len(data),),
            )
        else:
            # Create a gap buffer
            buffer = SeriesBuffer(
                offset=offset,
                sample_rate=sample_rate,
                data=None,
                shape=(1,),
            )
        return buffer

    def create_mock_frame(self, buffers, metadata=None, EOS=False):
        """Create a mock TSFrame."""
        frame = Mock(spec=TSFrame)
        frame.__iter__ = Mock(return_value=iter(buffers))
        frame.metadata = metadata or {}
        frame.EOS = EOS
        return frame

    def test_new_all_pass(self):
        """Test new() when all data passes the bit mask."""
        # Create bit mask requiring bits 0 and 1
        bit_mask = BitMask(
            name="TestBitMask",
            bit_mask=0x3,  # Binary: 11
            source_pad_names=("output",),
            sink_pad_names=("input",),
        )

        # Create mock sink pad
        sink_pad = Mock()
        bit_mask.sink_pads = [sink_pad]

        # Create mock frame with data that passes (has bits 0 and 1 set)
        buffer = self.create_mock_buffer([3, 7, 15])  # All have bits 0 and 1 set
        frame = self.create_mock_frame([buffer], metadata={"test": "metadata"})

        # Mock preparedframes
        bit_mask.preparedframes = {sink_pad: frame}

        # Create source pad
        source_pad = Mock()

        # Process frame
        result = bit_mask.new(source_pad)

        # Check result
        assert isinstance(result, TSFrame)
        assert len(result.buffers) == 1
        assert result.buffers[0] is buffer  # Should pass through unchanged
        assert result.metadata == {"test": "metadata"}
        assert result.EOS is False

    def test_new_all_fail(self):
        """Test new() when no data passes the bit mask."""
        # Create bit mask requiring bits 0 and 1
        bit_mask = BitMask(
            name="TestBitMask",
            bit_mask=0x3,  # Binary: 11
            source_pad_names=("output",),
            sink_pad_names=("input",),
        )

        # Create mock sink pad
        sink_pad = Mock()
        bit_mask.sink_pads = [sink_pad]

        # Create mock frame with data that fails (doesn't have both bits set)
        buffer = self.create_mock_buffer([0, 1, 2])  # None have both bits 0 and 1
        frame = self.create_mock_frame([buffer])

        # Mock preparedframes
        bit_mask.preparedframes = {sink_pad: frame}

        # Create source pad
        source_pad = Mock()

        # Process frame
        result = bit_mask.new(source_pad)

        # Check result
        assert len(result.buffers) == 1
        # Buffer should be converted to gap
        assert result.buffers[0].is_gap is True
        assert result.buffers[0].data is None

    def test_new_mixed_pass_fail(self):
        """Test new() when some data passes and some fails."""
        # Create bit mask requiring bit 0
        bit_mask = BitMask(
            name="TestBitMask",
            bit_mask=0x1,  # Binary: 1
            source_pad_names=("output",),
            sink_pad_names=("input",),
        )

        # Create mock sink pad
        sink_pad = Mock()
        bit_mask.sink_pads = [sink_pad]

        # Create mock frame with mixed data
        buffer = self.create_mock_buffer([1, 0, 3, 2])  # 1 and 3 pass, 0 and 2 fail
        frame = self.create_mock_frame([buffer])

        # Mock preparedframes
        bit_mask.preparedframes = {sink_pad: frame}

        # Create source pad
        source_pad = Mock()

        # Process frame
        result = bit_mask.new(source_pad)

        # Check result - should have 4 separate buffers
        assert len(result.buffers) == 4

        # Check each buffer
        # Buffer 0: data=1 (pass)
        assert result.buffers[0].data is not None
        assert result.buffers[0].data[0] == 1
        assert result.buffers[0].offset == 0

        # Buffer 1: data=0 (fail - gap)
        assert result.buffers[1].data is None
        assert result.buffers[1].offset == result.buffers[0].end_offset

        # Buffer 2: data=3 (pass)
        assert result.buffers[2].data is not None
        assert result.buffers[2].data[0] == 3
        assert result.buffers[2].offset == result.buffers[1].end_offset

        # Buffer 3: data=2 (fail - gap)
        assert result.buffers[3].data is None
        assert result.buffers[3].offset == result.buffers[2].end_offset

    def test_new_with_gap_buffer(self):
        """Test new() with gap buffer input."""
        bit_mask = BitMask(
            name="TestBitMask",
            bit_mask=0x1,
            source_pad_names=("output",),
            sink_pad_names=("input",),
        )

        # Create mock sink pad
        sink_pad = Mock()
        bit_mask.sink_pads = [sink_pad]

        # Create mock frame with gap buffer
        gap_buffer = self.create_mock_buffer(None, is_gap=True)
        frame = self.create_mock_frame([gap_buffer])

        # Mock preparedframes
        bit_mask.preparedframes = {sink_pad: frame}

        # Create source pad
        source_pad = Mock()

        # Process frame
        result = bit_mask.new(source_pad)

        # Check result - gap should pass through
        assert len(result.buffers) == 1
        assert result.buffers[0] is gap_buffer

    def test_new_with_eos(self):
        """Test new() with End-of-Stream."""
        bit_mask = BitMask(
            name="TestBitMask",
            bit_mask=0x1,
            source_pad_names=("output",),
            sink_pad_names=("input",),
        )

        # Create mock sink pad
        sink_pad = Mock()
        bit_mask.sink_pads = [sink_pad]

        # Create mock frame with EOS
        buffer = self.create_mock_buffer([1])
        frame = self.create_mock_frame([buffer], EOS=True)

        # Mock preparedframes
        bit_mask.preparedframes = {sink_pad: frame}

        # Create source pad
        source_pad = Mock()

        # Process frame
        result = bit_mask.new(source_pad)

        # Check that EOS is propagated
        assert result.EOS is True

    def test_new_complex_bit_mask(self):
        """Test new() with complex bit mask pattern."""
        # Create bit mask requiring bits 2, 4, and 5
        bit_mask = BitMask(
            name="TestBitMask",
            bit_mask=0x34,  # Binary: 110100
            source_pad_names=("output",),
            sink_pad_names=("input",),
        )

        # Create mock sink pad
        sink_pad = Mock()
        bit_mask.sink_pads = [sink_pad]

        # Create mock frame with various values
        # 52 = 110100 (pass - exact match)
        # 60 = 111100 (pass - has all required bits)
        # 20 = 010100 (fail - missing bit 5)
        # 48 = 110000 (fail - missing bit 2)
        buffer = self.create_mock_buffer([52, 60, 20, 48])
        frame = self.create_mock_frame([buffer])

        # Mock preparedframes
        bit_mask.preparedframes = {sink_pad: frame}

        # Create source pad
        source_pad = Mock()

        # Process frame
        result = bit_mask.new(source_pad)

        # Check result - should have 4 separate buffers
        assert len(result.buffers) == 4

        # Buffer 0: data=52 (pass)
        assert result.buffers[0].data is not None
        assert result.buffers[0].data[0] == 52

        # Buffer 1: data=60 (pass)
        assert result.buffers[1].data is not None
        assert result.buffers[1].data[0] == 60

        # Buffer 2: data=20 (fail - gap)
        assert result.buffers[2].data is None

        # Buffer 3: data=48 (fail - gap)
        assert result.buffers[3].data is None

    def test_new_multiple_buffers(self):
        """Test new() with multiple buffers in frame."""
        bit_mask = BitMask(
            name="TestBitMask",
            bit_mask=0x1,  # Binary: 1
            source_pad_names=("output",),
            sink_pad_names=("input",),
        )

        # Create mock sink pad
        sink_pad = Mock()
        bit_mask.sink_pads = [sink_pad]

        # Create mock frame with multiple buffers
        # Need to create buffers with proper offsets to avoid discontinuities
        buffer1 = self.create_mock_buffer([1, 3, 5], offset=0)  # All pass
        buffer2 = self.create_mock_buffer(
            [0, 2, 4], offset=buffer1.end_offset
        )  # All fail
        gap_buffer = self.create_mock_buffer(
            None, offset=buffer2.end_offset, is_gap=True
        )
        buffer3 = self.create_mock_buffer([1, 0], offset=gap_buffer.end_offset)  # Mixed

        frame = self.create_mock_frame([buffer1, buffer2, gap_buffer, buffer3])

        # Mock preparedframes
        bit_mask.preparedframes = {sink_pad: frame}

        # Create source pad
        source_pad = Mock()

        # Process frame
        result = bit_mask.new(source_pad)

        # Check result
        # buffer1: passes through as-is (1 buffer)
        # buffer2: converted to gap (1 buffer)
        # gap_buffer: passes through (1 buffer)
        # buffer3: split into 2 buffers
        assert len(result.buffers) == 5

    def test_new_empty_data(self):
        """Test new() with empty data array."""
        bit_mask = BitMask(
            name="TestBitMask",
            bit_mask=0x1,
            source_pad_names=("output",),
            sink_pad_names=("input",),
        )

        # Create mock sink pad
        sink_pad = Mock()
        bit_mask.sink_pads = [sink_pad]

        # Create mock frame with empty buffer
        buffer = self.create_mock_buffer([])
        frame = self.create_mock_frame([buffer])

        # Mock preparedframes
        bit_mask.preparedframes = {sink_pad: frame}

        # Create source pad
        source_pad = Mock()

        # Process frame
        result = bit_mask.new(source_pad)

        # Check result - should pass through
        assert len(result.buffers) == 1
        assert result.buffers[0] is buffer


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
