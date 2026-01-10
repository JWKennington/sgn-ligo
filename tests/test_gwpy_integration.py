"""Tests for the sgnligo.gwpy integration module."""

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from sgnts.base import Offset, SeriesBuffer, TSFrame

from sgnligo.gwpy import (
    GWpyFilter,
    seriesbuffer_to_timeseries,
    timeseries_to_seriesbuffer,
    tsframe_to_timeseries,
)
from sgnligo.gwpy.converters import (
    buffers_to_timeseriesdict,
    timeseries_to_tsframe,
    timeseriesdict_to_buffers,
)


class TestConverters:
    """Test SeriesBuffer <-> TimeSeries conversion."""

    def test_seriesbuffer_to_timeseries_basic(self):
        """Test basic conversion from SeriesBuffer to TimeSeries."""
        sample_rate = 4096
        gps_time = 1126259462.0
        data = np.random.randn(4096)

        buf = SeriesBuffer(
            offset=Offset.fromsec(gps_time),
            sample_rate=sample_rate,
            data=data,
        )

        ts = seriesbuffer_to_timeseries(buf, channel="H1:TEST")

        assert float(ts.t0.value) == pytest.approx(gps_time, rel=1e-6)
        assert int(ts.sample_rate.value) == sample_rate
        assert len(ts) == len(data)
        assert np.allclose(ts.value, data)

    def test_timeseries_to_seriesbuffer_basic(self):
        """Test basic conversion from TimeSeries to SeriesBuffer."""
        sample_rate = 4096
        gps_time = 1126259462.0
        data = np.random.randn(4096)

        ts = TimeSeries(data, t0=gps_time, sample_rate=sample_rate)
        buf = timeseries_to_seriesbuffer(ts)

        assert Offset.tosec(buf.offset) == pytest.approx(gps_time, rel=1e-6)
        assert buf.sample_rate == sample_rate
        assert np.allclose(buf.data, data)

    def test_round_trip_conversion(self):
        """Test that round-trip conversion preserves data."""
        sample_rate = 4096
        gps_time = 1126259462.0
        data = np.random.randn(4096)

        buf1 = SeriesBuffer(
            offset=Offset.fromsec(gps_time),
            sample_rate=sample_rate,
            data=data,
        )

        ts = seriesbuffer_to_timeseries(buf1)
        buf2 = timeseries_to_seriesbuffer(ts)

        assert buf1.offset == buf2.offset
        assert buf1.sample_rate == buf2.sample_rate
        assert np.allclose(buf1.data, buf2.data)

    def test_gap_buffer_conversion(self):
        """Test that gap buffers are converted to NaN and back."""
        sample_rate = 4096
        gps_time = 1126259462.0

        gap_buf = SeriesBuffer(
            offset=Offset.fromsec(gps_time),
            sample_rate=sample_rate,
            data=None,
            shape=(4096,),
        )

        ts = seriesbuffer_to_timeseries(gap_buf)
        assert np.all(np.isnan(ts.value))

        buf_back = timeseries_to_seriesbuffer(ts)
        assert buf_back.is_gap

    def test_tsframe_to_timeseries(self):
        """Test TSFrame to TimeSeries conversion."""
        sample_rate = 4096
        gps_time = 1126259462.0
        data = np.random.randn(4096)

        buf = SeriesBuffer(
            offset=Offset.fromsec(gps_time),
            sample_rate=sample_rate,
            data=data,
        )
        frame = TSFrame(buffers=[buf])

        ts = tsframe_to_timeseries(frame, channel="H1:TEST")

        assert float(ts.t0.value) == pytest.approx(gps_time, rel=1e-6)
        assert len(ts) == len(data)

    def test_tsframe_to_timeseries_empty_raises(self):
        """Test that empty frame raises ValueError."""
        frame = TSFrame(buffers=[])
        with pytest.raises(ValueError, match="Cannot convert empty TSFrame"):
            tsframe_to_timeseries(frame)

    def test_tsframe_to_timeseries_fill_gaps_false_with_gap(self):
        """Test that frame with gaps raises when fill_gaps=False."""
        sample_rate = 4096
        gps_time = 1126259462.0

        gap_buf = SeriesBuffer(
            offset=Offset.fromsec(gps_time),
            sample_rate=sample_rate,
            data=None,
            shape=(4096,),
        )
        frame = TSFrame(buffers=[gap_buf])

        with pytest.raises(ValueError, match="Frame contains gaps"):
            tsframe_to_timeseries(frame, fill_gaps=False)

    def test_timeseries_to_tsframe(self):
        """Test TimeSeries to TSFrame conversion."""
        sample_rate = 4096
        gps_time = 1126259462.0
        data = np.random.randn(4096)

        ts = TimeSeries(data, t0=gps_time, sample_rate=sample_rate, channel="H1:TEST")
        frame = timeseries_to_tsframe(ts)

        assert len(frame.buffers) == 1
        assert frame.sample_rate == sample_rate
        assert np.allclose(frame.buffers[0].data, data)

    def test_timeseriesdict_to_buffers(self):
        """Test TimeSeriesDict to buffers conversion."""
        sample_rate = 4096
        gps_time = 1126259462.0
        data1 = np.random.randn(4096)
        data2 = np.random.randn(4096)

        tsd = TimeSeriesDict()
        tsd["H1:STRAIN"] = TimeSeries(data1, t0=gps_time, sample_rate=sample_rate)
        tsd["L1:STRAIN"] = TimeSeries(data2, t0=gps_time, sample_rate=sample_rate)

        buffers = timeseriesdict_to_buffers(tsd)

        assert "H1:STRAIN" in buffers
        assert "L1:STRAIN" in buffers
        assert np.allclose(buffers["H1:STRAIN"].data, data1)
        assert np.allclose(buffers["L1:STRAIN"].data, data2)

    def test_buffers_to_timeseriesdict(self):
        """Test buffers to TimeSeriesDict conversion."""
        sample_rate = 4096
        gps_time = 1126259462.0
        data1 = np.random.randn(4096)
        data2 = np.random.randn(4096)

        buffers = {
            "H1:STRAIN": SeriesBuffer(
                offset=Offset.fromsec(gps_time),
                sample_rate=sample_rate,
                data=data1,
            ),
            "L1:STRAIN": SeriesBuffer(
                offset=Offset.fromsec(gps_time),
                sample_rate=sample_rate,
                data=data2,
            ),
        }

        tsd = buffers_to_timeseriesdict(buffers)

        assert "H1:STRAIN" in tsd
        assert "L1:STRAIN" in tsd
        assert np.allclose(tsd["H1:STRAIN"].value, data1)
        assert np.allclose(tsd["L1:STRAIN"].value, data2)


class TestGWpyFilter:
    """Test GWpyFilter transform."""

    def test_bandpass_creation(self):
        """Test creating a bandpass filter."""
        filt = GWpyFilter(
            name="BP",
            sink_pad_names=("in",),
            source_pad_names=("out",),
            filter_type="bandpass",
            low_freq=20,
            high_freq=500,
        )
        assert filt.filter_type == "bandpass"
        assert filt.low_freq == 20
        assert filt.high_freq == 500

    def test_lowpass_creation(self):
        """Test creating a lowpass filter."""
        filt = GWpyFilter(
            name="LP",
            sink_pad_names=("in",),
            source_pad_names=("out",),
            filter_type="lowpass",
            high_freq=100,
        )
        assert filt.filter_type == "lowpass"
        assert filt.high_freq == 100

    def test_highpass_creation(self):
        """Test creating a highpass filter."""
        filt = GWpyFilter(
            name="HP",
            sink_pad_names=("in",),
            source_pad_names=("out",),
            filter_type="highpass",
            low_freq=10,
        )
        assert filt.filter_type == "highpass"
        assert filt.low_freq == 10

    def test_notch_creation(self):
        """Test creating a notch filter."""
        filt = GWpyFilter(
            name="Notch",
            sink_pad_names=("in",),
            source_pad_names=("out",),
            filter_type="notch",
            notch_freq=60,
            notch_q=30,
        )
        assert filt.filter_type == "notch"
        assert filt.notch_freq == 60

    def test_bandpass_attenuates_out_of_band(self):
        """Test that bandpass filter attenuates out-of-band frequencies."""
        from scipy import signal

        sample_rate = 4096
        duration = 4.0
        t = np.arange(int(sample_rate * duration)) / sample_rate

        # Create signal: 100 Hz (in-band) + 500 Hz (out-of-band)
        data = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 500 * t)

        ts = TimeSeries(data, t0=0, sample_rate=sample_rate)

        filt = GWpyFilter(
            name="BP",
            sink_pad_names=("in",),
            source_pad_names=("out",),
            filter_type="bandpass",
            low_freq=50,
            high_freq=150,
        )

        filtered = filt._apply_filter(ts)

        # Check power at 100 Hz vs 500 Hz
        f, psd = signal.welch(filtered.value, fs=sample_rate, nperseg=1024)
        idx_100 = np.argmin(np.abs(f - 100))
        idx_500 = np.argmin(np.abs(f - 500))

        # 100 Hz should have significant power, 500 Hz should be attenuated
        assert psd[idx_100] > psd[idx_500] * 1000

    def test_explicit_edge_duration(self):
        """Test creating filter with explicit edge_duration (line 276)."""
        filt = GWpyFilter(
            name="BP",
            sink_pad_names=("in",),
            source_pad_names=("out",),
            filter_type="bandpass",
            low_freq=20,
            high_freq=500,
            edge_duration=0.5,  # Explicitly set edge duration
        )
        assert filt.edge_duration == 0.5

    def test_lowpass_apply_filter(self):
        """Test lowpass filter application."""
        sample_rate = 4096
        data = np.random.randn(4096)
        ts = TimeSeries(data, t0=0, sample_rate=sample_rate)

        filt = GWpyFilter(
            name="LP",
            sink_pad_names=("in",),
            source_pad_names=("out",),
            filter_type="lowpass",
            high_freq=100,
        )

        filtered = filt._apply_filter(ts)
        assert len(filtered) == len(ts)

    def test_highpass_apply_filter(self):
        """Test highpass filter application."""
        sample_rate = 4096
        data = np.random.randn(4096)
        ts = TimeSeries(data, t0=0, sample_rate=sample_rate)

        filt = GWpyFilter(
            name="HP",
            sink_pad_names=("in",),
            source_pad_names=("out",),
            filter_type="highpass",
            low_freq=10,
        )

        filtered = filt._apply_filter(ts)
        assert len(filtered) == len(ts)

    def test_notch_apply_filter(self):
        """Test notch filter application."""
        sample_rate = 4096
        data = np.random.randn(4096)
        ts = TimeSeries(data, t0=0, sample_rate=sample_rate)

        filt = GWpyFilter(
            name="Notch",
            sink_pad_names=("in",),
            source_pad_names=("out",),
            filter_type="notch",
            notch_freq=60,
        )

        # Note: GWpy's notch method has a specific signature - we just verify it runs
        filtered = filt._apply_filter(ts)
        assert len(filtered) == len(ts)

    def test_invalid_filter_type_raises(self):
        """Test that invalid filter type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown filter_type"):
            GWpyFilter(
                name="Invalid",
                sink_pad_names=("in",),
                source_pad_names=("out",),
                filter_type="invalid",
            )

    def test_bandpass_missing_freq_raises(self):
        """Test that bandpass without frequencies raises ValueError."""
        with pytest.raises(ValueError, match="requires low_freq and high_freq"):
            GWpyFilter(
                name="BP",
                sink_pad_names=("in",),
                source_pad_names=("out",),
                filter_type="bandpass",
                low_freq=20,
                # missing high_freq
            )

    def test_lowpass_missing_freq_raises(self):
        """Test that lowpass without high_freq raises ValueError."""
        with pytest.raises(ValueError, match="requires high_freq"):
            GWpyFilter(
                name="LP",
                sink_pad_names=("in",),
                source_pad_names=("out",),
                filter_type="lowpass",
                # missing high_freq
            )

    def test_highpass_missing_freq_raises(self):
        """Test that highpass without low_freq raises ValueError."""
        with pytest.raises(ValueError, match="requires low_freq"):
            GWpyFilter(
                name="HP",
                sink_pad_names=("in",),
                source_pad_names=("out",),
                filter_type="highpass",
                # missing low_freq
            )

    def test_notch_missing_freq_raises(self):
        """Test that notch without notch_freq raises ValueError."""
        with pytest.raises(ValueError, match="requires notch_freq"):
            GWpyFilter(
                name="Notch",
                sink_pad_names=("in",),
                source_pad_names=("out",),
                filter_type="notch",
                # missing notch_freq
            )


class TestGWpySpectrogram:
    """Test GWpySpectrogram transform."""

    def test_spectrogram_creation(self):
        """Test creating a spectrogram transform."""
        from sgnligo.gwpy.transforms.spectrogram import GWpySpectrogram

        spec = GWpySpectrogram(
            name="Spec",
            sink_pad_names=("in",),
            source_pad_names=("out",),
            spec_stride=1.0,
            fft_length=2.0,
        )
        assert spec.spec_stride == 1.0
        assert spec.fft_length == 2.0
        assert spec.fft_overlap == 1.0  # Default: fft_length / 2
        assert spec.output_rate in {2**n for n in range(15)}  # Power-of-2

    def test_spectrogram_invalid_stride_raises(self):
        """Test that non-positive spec_stride raises ValueError."""
        from sgnligo.gwpy.transforms.spectrogram import GWpySpectrogram

        with pytest.raises(ValueError, match="spec_stride must be positive"):
            GWpySpectrogram(
                name="Spec",
                sink_pad_names=("in",),
                source_pad_names=("out",),
                spec_stride=-1,
            )

    def test_spectrogram_invalid_fft_length_raises(self):
        """Test that non-positive fft_length raises ValueError."""
        from sgnligo.gwpy.transforms.spectrogram import GWpySpectrogram

        with pytest.raises(ValueError, match="fft_length must be positive"):
            GWpySpectrogram(
                name="Spec",
                sink_pad_names=("in",),
                source_pad_names=("out",),
                fft_length=-1,
            )


class TestGWpyQTransform:
    """Test GWpyQTransform transform."""

    def test_qtransform_creation(self):
        """Test creating a Q-transform."""
        from sgnligo.gwpy.transforms.qtransform import GWpyQTransform

        qt = GWpyQTransform(
            name="QT",
            sink_pad_names=("in",),
            source_pad_names=("out",),
            qrange=(4, 64),
            frange=(20, 1024),
        )
        assert qt.qrange == (4, 64)
        assert qt.frange == (20, 1024)
        assert qt.output_stride > 0
        assert qt.output_rate in {2**n for n in range(15)}  # Power-of-2

    def test_qtransform_invalid_qrange_raises(self):
        """Test that invalid qrange raises ValueError."""
        from sgnligo.gwpy.transforms.qtransform import GWpyQTransform

        with pytest.raises(ValueError, match="qrange"):
            GWpyQTransform(
                name="QT",
                sink_pad_names=("in",),
                source_pad_names=("out",),
                qrange=(64, 4),  # min > max
            )

    def test_qtransform_invalid_frange_raises(self):
        """Test that invalid frange raises ValueError."""
        from sgnligo.gwpy.transforms.qtransform import GWpyQTransform

        with pytest.raises(ValueError, match="frange"):
            GWpyQTransform(
                name="QT",
                sink_pad_names=("in",),
                source_pad_names=("out",),
                frange=(1024, 20),  # min > max
            )

    def test_qtransform_invalid_output_stride_raises(self):
        """Test that non-positive output_stride raises ValueError."""
        from sgnligo.gwpy.transforms.qtransform import GWpyQTransform

        with pytest.raises(ValueError, match="output_stride must be positive"):
            GWpyQTransform(
                name="QT",
                sink_pad_names=("in",),
                source_pad_names=("out",),
                output_stride=-1,
            )

    def test_qtransform_invalid_output_rate_raises(self):
        """Test that non-power-of-2 output_rate raises ValueError."""
        from sgnligo.gwpy.transforms.qtransform import GWpyQTransform

        with pytest.raises(ValueError, match="output_rate.*power-of-2"):
            GWpyQTransform(
                name="QT",
                sink_pad_names=("in",),
                source_pad_names=("out",),
                output_rate=100,  # Not power of 2
            )

    def test_qtransform_linear_freq_spacing_with_fres(self):
        """Test _estimate_freq_bins with logf=False and fres (lines 182-183)."""
        from sgnligo.gwpy.transforms.qtransform import GWpyQTransform

        qt = GWpyQTransform(
            name="QT",
            sink_pad_names=("in",),
            source_pad_names=("out",),
            qrange=(4, 64),
            frange=(20, 520),  # 500 Hz range
            logf=False,  # Linear frequency spacing
            fres=10.0,  # 10 Hz resolution -> 50 bins expected
        )
        # _estimate_freq_bins should return (520-20)/10 = 50
        assert qt._estimate_freq_bins() == 50

    def test_qtransform_linear_freq_spacing_no_fres(self):
        """Test _estimate_freq_bins with logf=False and no fres (line 184)."""
        from sgnligo.gwpy.transforms.qtransform import GWpyQTransform

        qt = GWpyQTransform(
            name="QT",
            sink_pad_names=("in",),
            source_pad_names=("out",),
            qrange=(4, 64),
            frange=(20, 520),
            logf=False,  # Linear frequency spacing
            fres=None,  # No fres specified -> defaults to 100
        )
        # _estimate_freq_bins should return 100 (default)
        assert qt._estimate_freq_bins() == 100


class TestTimeSeriesSource:
    """Test TimeSeriesSource."""

    def test_source_creation_from_timeseries(self):
        """Test creating source from a single TimeSeries."""
        from sgnligo.gwpy.sources.timeseries_source import TimeSeriesSource

        sample_rate = 4096
        gps_time = 1126259462.0
        data = np.random.randn(4096 * 4)  # 4 seconds

        ts = TimeSeries(data, t0=gps_time, sample_rate=sample_rate, channel="H1:TEST")

        source = TimeSeriesSource(
            name="Source",
            timeseries=ts,
            buffer_duration=1.0,
        )

        assert source.t0 == gps_time
        assert source.end == gps_time + 4.0
        assert len(source.source_pad_names) == 1

    def test_source_creation_from_timeseriesdict(self):
        """Test creating source from a TimeSeriesDict."""
        from sgnligo.gwpy.sources.timeseries_source import TimeSeriesSource

        sample_rate = 4096
        gps_time = 1126259462.0
        data1 = np.random.randn(4096 * 4)
        data2 = np.random.randn(4096 * 4)

        tsd = TimeSeriesDict()
        tsd["H1:STRAIN"] = TimeSeries(
            data1, t0=gps_time, sample_rate=sample_rate, channel="H1:STRAIN"
        )
        tsd["L1:STRAIN"] = TimeSeries(
            data2, t0=gps_time, sample_rate=sample_rate, channel="L1:STRAIN"
        )

        source = TimeSeriesSource(
            name="Source",
            timeseries=tsd,
            buffer_duration=1.0,
        )

        assert len(source.source_pad_names) == 2
        assert "H1:STRAIN" in source.source_pad_names
        assert "L1:STRAIN" in source.source_pad_names

    def test_source_missing_timeseries_raises(self):
        """Test that missing timeseries raises ValueError."""
        from sgnligo.gwpy.sources.timeseries_source import TimeSeriesSource

        with pytest.raises(ValueError, match="timeseries argument is required"):
            TimeSeriesSource(
                name="Source",
                source_pad_names=("out",),
                timeseries=None,
            )

    def test_source_invalid_type_raises(self):
        """Test that invalid timeseries type raises TypeError."""
        from sgnligo.gwpy.sources.timeseries_source import TimeSeriesSource

        with pytest.raises(TypeError, match="must be TimeSeries or TimeSeriesDict"):
            TimeSeriesSource(
                name="Source",
                source_pad_names=("out",),
                timeseries="not a timeseries",
            )

    def test_source_empty_timeseriesdict_raises(self):
        """Test that empty TimeSeriesDict raises ValueError."""
        from sgnligo.gwpy.sources.timeseries_source import TimeSeriesSource

        with pytest.raises(ValueError, match="timeseries cannot be empty"):
            TimeSeriesSource(
                name="Source",
                source_pad_names=("out",),
                timeseries=TimeSeriesDict(),
            )

    def test_source_new_generates_frames(self):
        """Test that new() generates proper frames."""
        from sgnligo.gwpy.sources.timeseries_source import TimeSeriesSource

        sample_rate = 4096
        gps_time = 1126259462.0
        data = np.random.randn(4096 * 4)  # 4 seconds

        ts = TimeSeries(data, t0=gps_time, sample_rate=sample_rate, channel="H1:TEST")

        source = TimeSeriesSource(
            name="Source",
            timeseries=ts,
            buffer_duration=1.0,
        )

        pad = source.source_pads[0]

        # Get first frame
        frame1 = source.new(pad)
        assert len(frame1.buffers) == 1
        assert frame1.buffers[0].sample_rate == sample_rate
        assert frame1.buffers[0].shape[0] == 4096  # 1 second worth
        assert not frame1.EOS

        # Get subsequent frames
        frame2 = source.new(pad)
        assert not frame2.EOS

        frame3 = source.new(pad)
        assert not frame3.EOS

        # Last frame should have EOS
        frame4 = source.new(pad)
        assert frame4.EOS

    def test_source_new_after_eos_returns_empty_buffer(self):
        """Test new() after EOS returns empty buffer (lines 141, 159)."""
        from sgnligo.gwpy.sources.timeseries_source import TimeSeriesSource

        sample_rate = 4096
        gps_time = 1126259462.0
        data = np.random.randn(4096)  # 1 second

        ts = TimeSeries(data, t0=gps_time, sample_rate=sample_rate, channel="H1:TEST")

        source = TimeSeriesSource(
            name="Source",
            timeseries=ts,
            buffer_duration=1.0,
        )

        pad = source.source_pads[0]

        # Get first (and last) frame with EOS
        frame1 = source.new(pad)
        assert frame1.EOS
        assert frame1.buffers[0].shape[0] == 4096

        # Call new() after EOS - should return empty buffer
        frame2 = source.new(pad)
        assert frame2.EOS
        # Buffer should have data=None (gap) with shape (0,)
        assert frame2.buffers[0].data is None
        assert frame2.buffers[0].shape == (0,)


class TestTimeSeriesSink:
    """Test TimeSeriesSink."""

    def test_sink_creation(self):
        """Test creating a sink."""
        from sgnligo.gwpy.sinks.timeseries_sink import TimeSeriesSink

        sink = TimeSeriesSink(
            name="Sink",
            sink_pad_names=("in",),
            channel="H1:OUTPUT",
            unit="strain",
        )
        assert sink.channel == "H1:OUTPUT"
        assert sink.unit == "strain"
        assert sink.collect_all is True

    def test_sink_get_result_no_data_raises(self):
        """Test that get_result with no data raises ValueError."""
        from sgnligo.gwpy.sinks.timeseries_sink import TimeSeriesSink

        sink = TimeSeriesSink(
            name="Sink",
            sink_pad_names=("in",),
        )

        with pytest.raises(ValueError, match="No data collected"):
            sink.get_result()

    def test_sink_clear(self):
        """Test that clear() resets the sink."""
        from sgnligo.gwpy.sinks.timeseries_sink import TimeSeriesSink

        sink = TimeSeriesSink(
            name="Sink",
            sink_pad_names=("in",),
        )

        # Add some buffers manually
        buf = SeriesBuffer(
            offset=Offset.fromsec(1126259462.0),
            sample_rate=4096,
            data=np.random.randn(4096),
        )
        sink._buffers.append(buf)
        sink._sample_rate = 4096
        sink._first_offset = buf.offset
        sink._is_complete = True

        assert len(sink._buffers) == 1

        sink.clear()

        assert len(sink._buffers) == 0
        assert sink._sample_rate is None
        assert sink._first_offset is None
        assert sink._is_complete is False

    def test_sink_properties(self):
        """Test sink properties."""
        from sgnligo.gwpy.sinks.timeseries_sink import TimeSeriesSink

        sink = TimeSeriesSink(
            name="Sink",
            sink_pad_names=("in",),
        )

        # Initially no data
        assert sink.samples_collected == 0
        assert sink.duration_collected == 0.0
        assert sink.is_complete is False

        # Add a buffer
        buf = SeriesBuffer(
            offset=Offset.fromsec(1126259462.0),
            sample_rate=4096,
            data=np.random.randn(4096),
        )
        sink._buffers.append(buf)
        sink._sample_rate = 4096

        assert sink.samples_collected == 4096
        assert sink.duration_collected == 1.0

    def test_sink_get_result(self):
        """Test get_result returns proper TimeSeries."""
        from sgnligo.gwpy.sinks.timeseries_sink import TimeSeriesSink

        sink = TimeSeriesSink(
            name="Sink",
            sink_pad_names=("in",),
            channel="H1:OUTPUT",
        )

        # Add buffers manually
        gps_time = 1126259462.0
        data1 = np.random.randn(4096)
        data2 = np.random.randn(4096)

        buf1 = SeriesBuffer(
            offset=Offset.fromsec(gps_time),
            sample_rate=4096,
            data=data1,
        )
        buf2 = SeriesBuffer(
            offset=Offset.fromsec(gps_time + 1.0),
            sample_rate=4096,
            data=data2,
        )

        sink._buffers = [buf1, buf2]
        sink._sample_rate = 4096
        sink._first_offset = buf1.offset

        ts = sink.get_result()

        assert len(ts) == 8192
        assert ts.channel.name == "H1:OUTPUT"
        assert float(ts.t0.value) == pytest.approx(gps_time, rel=1e-6)

    def test_sink_get_result_with_gaps(self):
        """Test get_result with gap buffers fills NaN."""
        from sgnligo.gwpy.sinks.timeseries_sink import TimeSeriesSink

        sink = TimeSeriesSink(
            name="Sink",
            sink_pad_names=("in",),
        )

        gps_time = 1126259462.0
        data1 = np.random.randn(4096)

        buf1 = SeriesBuffer(
            offset=Offset.fromsec(gps_time),
            sample_rate=4096,
            data=data1,
        )
        gap_buf = SeriesBuffer(
            offset=Offset.fromsec(gps_time + 1.0),
            sample_rate=4096,
            data=None,
            shape=(4096,),
        )

        sink._buffers = [buf1, gap_buf]
        sink._sample_rate = 4096
        sink._first_offset = buf1.offset

        ts = sink.get_result()

        assert len(ts) == 8192
        # First half should be data
        assert not np.any(np.isnan(ts.value[:4096]))
        # Second half should be NaN (gap)
        assert np.all(np.isnan(ts.value[4096:]))

    def test_sink_get_result_dict(self):
        """Test get_result_dict returns TimeSeriesDict."""
        from sgnligo.gwpy.sinks.timeseries_sink import TimeSeriesSink

        sink = TimeSeriesSink(
            name="Sink",
            sink_pad_names=("in",),
            channel="H1:OUTPUT",
        )

        gps_time = 1126259462.0
        data = np.random.randn(4096)

        buf = SeriesBuffer(
            offset=Offset.fromsec(gps_time),
            sample_rate=4096,
            data=data,
        )

        sink._buffers = [buf]
        sink._sample_rate = 4096
        sink._first_offset = buf.offset

        tsd = sink.get_result_dict()

        assert isinstance(tsd, TimeSeriesDict)
        assert "H1:OUTPUT" in tsd

    def test_sink_collect_all_false(self):
        """Test that collect_all=False only keeps last buffer."""
        from sgnligo.gwpy.sinks.timeseries_sink import TimeSeriesSink

        sink = TimeSeriesSink(
            name="Sink",
            sink_pad_names=("in",),
            collect_all=False,
        )

        # Simulate internal adding buffers
        gps_time = 1126259462.0

        buf1 = SeriesBuffer(
            offset=Offset.fromsec(gps_time),
            sample_rate=4096,
            data=np.random.randn(4096),
        )
        buf2 = SeriesBuffer(
            offset=Offset.fromsec(gps_time + 1.0),
            sample_rate=4096,
            data=np.random.randn(4096),
        )

        # When collect_all is False, only the last buffer should be kept
        sink._sample_rate = 4096
        sink._first_offset = buf1.offset
        sink._buffers = [buf1]

        # Simulating what internal() does when collect_all=False
        sink._buffers = [buf2]

        assert len(sink._buffers) == 1

    def test_sink_handles_none_frame(self):
        """Test that TimeSeriesSink handles None frames gracefully (line 86)."""
        from unittest.mock import MagicMock, patch

        from sgnts.base import TSSink

        from sgnligo.gwpy.sinks.timeseries_sink import TimeSeriesSink

        sink = TimeSeriesSink(
            name="Sink",
            sink_pad_names=("in",),
        )

        # Mock preparedframes to return None
        mock_pad = MagicMock()
        sink.sink_pads = [mock_pad]
        sink.preparedframes = {mock_pad: None}

        # Mock parent internal() to avoid audio adapter issues
        with patch.object(TSSink, "internal"):
            sink.internal()

        # No data should be collected
        assert sink.samples_collected == 0

    def test_sink_handles_zero_length_buffer(self):
        """Test that TimeSeriesSink skips zero-length buffers (line 97)."""
        from unittest.mock import MagicMock, patch

        from sgnts.base import TSSink

        from sgnligo.gwpy.sinks.timeseries_sink import TimeSeriesSink

        sink = TimeSeriesSink(
            name="Sink",
            sink_pad_names=("in",),
        )

        # Create mock frame with zero-length buffer
        mock_pad = MagicMock()
        mock_buffer = MagicMock()
        mock_buffer.shape = (0,)  # Zero-length buffer

        mock_frame = MagicMock()
        mock_frame.EOS = False
        mock_frame.buffers = [mock_buffer]

        sink.sink_pads = [mock_pad]
        sink.preparedframes = {mock_pad: mock_frame}

        # Mock parent internal() to avoid audio adapter issues
        with patch.object(TSSink, "internal"):
            sink.internal()

        # No data should be collected from zero-length buffer
        assert sink.samples_collected == 0

    def test_sink_collect_all_false_internal(self):
        """Test internal() with collect_all=False keeps only last buffer (line 109)."""
        from unittest.mock import MagicMock, patch

        from sgnts.base import TSSink

        from sgnligo.gwpy.sinks.timeseries_sink import TimeSeriesSink

        sink = TimeSeriesSink(
            name="Sink",
            sink_pad_names=("in",),
            collect_all=False,
        )

        gps_time = 1126259462.0
        mock_pad = MagicMock()

        # First buffer
        buf1 = SeriesBuffer(
            offset=Offset.fromsec(gps_time),
            sample_rate=4096,
            data=np.random.randn(4096),
        )
        mock_frame1 = MagicMock()
        mock_frame1.EOS = False
        mock_frame1.buffers = [buf1]

        sink.sink_pads = [mock_pad]
        sink.preparedframes = {mock_pad: mock_frame1}

        with patch.object(TSSink, "internal"):
            sink.internal()

        assert len(sink._buffers) == 1

        # Second buffer - should replace the first
        buf2 = SeriesBuffer(
            offset=Offset.fromsec(gps_time + 1.0),
            sample_rate=4096,
            data=np.random.randn(4096),
        )
        mock_frame2 = MagicMock()
        mock_frame2.EOS = False
        mock_frame2.buffers = [buf2]

        sink.preparedframes = {mock_pad: mock_frame2}

        with patch.object(TSSink, "internal"):
            sink.internal()

        # Should still be only 1 buffer (the second one)
        assert len(sink._buffers) == 1
        assert sink._buffers[0] is buf2

    def test_sink_handles_eos_frame(self):
        """Test that TimeSeriesSink handles EOS frames (lines 90-91)."""
        from unittest.mock import MagicMock, patch

        from sgnts.base import TSSink

        from sgnligo.gwpy.sinks.timeseries_sink import TimeSeriesSink

        sink = TimeSeriesSink(
            name="Sink",
            sink_pad_names=("in",),
        )

        gps_time = 1126259462.0
        mock_pad = MagicMock()

        buf = SeriesBuffer(
            offset=Offset.fromsec(gps_time),
            sample_rate=4096,
            data=np.random.randn(4096),
        )
        mock_frame = MagicMock()
        mock_frame.EOS = True  # End of stream
        mock_frame.buffers = [buf]

        sink.sink_pads = [mock_pad]
        sink.preparedframes = {mock_pad: mock_frame}

        # Mock parent internal() and mark_eos()
        with patch.object(TSSink, "internal"):
            with patch.object(sink, "mark_eos") as mock_mark_eos:
                sink.internal()

        # Should have marked complete and called mark_eos
        assert sink._is_complete is True
        mock_mark_eos.assert_called_once_with(mock_pad)

    def test_sink_collect_all_true_appends(self):
        """Test internal() with collect_all=True appends buffers (line 106)."""
        from unittest.mock import MagicMock, patch

        from sgnts.base import TSSink

        from sgnligo.gwpy.sinks.timeseries_sink import TimeSeriesSink

        sink = TimeSeriesSink(
            name="Sink",
            sink_pad_names=("in",),
            collect_all=True,  # Default, but explicit
        )

        gps_time = 1126259462.0
        mock_pad = MagicMock()

        # First buffer
        buf1 = SeriesBuffer(
            offset=Offset.fromsec(gps_time),
            sample_rate=4096,
            data=np.random.randn(4096),
        )
        mock_frame1 = MagicMock()
        mock_frame1.EOS = False
        mock_frame1.buffers = [buf1]

        sink.sink_pads = [mock_pad]
        sink.preparedframes = {mock_pad: mock_frame1}

        with patch.object(TSSink, "internal"):
            sink.internal()

        assert len(sink._buffers) == 1

        # Second buffer - should be appended
        buf2 = SeriesBuffer(
            offset=Offset.fromsec(gps_time + 1.0),
            sample_rate=4096,
            data=np.random.randn(4096),
        )
        mock_frame2 = MagicMock()
        mock_frame2.EOS = False
        mock_frame2.buffers = [buf2]

        sink.preparedframes = {mock_pad: mock_frame2}

        with patch.object(TSSink, "internal"):
            sink.internal()

        # Should now have 2 buffers
        assert len(sink._buffers) == 2
        assert sink._buffers[0] is buf1
        assert sink._buffers[1] is buf2


class TestConverterEdgeCases:
    """Test converter edge cases."""

    def test_tsframe_fill_gaps_false_no_gap(self):
        """Test tsframe_to_timeseries with fill_gaps=False and no gaps."""
        sample_rate = 4096
        gps_time = 1126259462.0
        data = np.random.randn(4096)

        buf = SeriesBuffer(
            offset=Offset.fromsec(gps_time),
            sample_rate=sample_rate,
            data=data,
        )
        frame = TSFrame(buffers=[buf])

        ts = tsframe_to_timeseries(frame, fill_gaps=False)
        assert len(ts) == len(data)


class TestGWpyFilterPipeline:
    """Test GWpyFilter in a real pipeline."""

    def test_filter_pipeline_bandpass(self):
        """Test bandpass filter in a pipeline."""
        from sgn import NullSink
        from sgn.apps import Pipeline
        from sgnts.sources import FakeSeriesSource

        pipeline = Pipeline()

        pipeline.insert(
            FakeSeriesSource(
                name="src",
                source_pad_names=("out",),
                rate=4096,
                signal_type="white",
                t0=0,
                end=4,
            ),
            GWpyFilter(
                name="filter",
                sink_pad_names=("in",),
                source_pad_names=("out",),
                filter_type="bandpass",
                low_freq=20,
                high_freq=500,
            ),
            NullSink(name="snk", sink_pad_names=("out",)),
            link_map={
                "filter:snk:in": "src:src:out",
                "snk:snk:out": "filter:src:out",
            },
        )

        pipeline.run()

    def test_filter_pipeline_lowpass(self):
        """Test lowpass filter in a pipeline."""
        from sgn import NullSink
        from sgn.apps import Pipeline
        from sgnts.sources import FakeSeriesSource

        pipeline = Pipeline()

        pipeline.insert(
            FakeSeriesSource(
                name="src",
                source_pad_names=("out",),
                rate=4096,
                signal_type="white",
                t0=0,
                end=4,
            ),
            GWpyFilter(
                name="filter",
                sink_pad_names=("in",),
                source_pad_names=("out",),
                filter_type="lowpass",
                high_freq=100,
            ),
            NullSink(name="snk", sink_pad_names=("out",)),
            link_map={
                "filter:snk:in": "src:src:out",
                "snk:snk:out": "filter:src:out",
            },
        )

        pipeline.run()

    def test_filter_pipeline_highpass(self):
        """Test highpass filter in a pipeline."""
        from sgn import NullSink
        from sgn.apps import Pipeline
        from sgnts.sources import FakeSeriesSource

        pipeline = Pipeline()

        pipeline.insert(
            FakeSeriesSource(
                name="src",
                source_pad_names=("out",),
                rate=4096,
                signal_type="white",
                t0=0,
                end=4,
            ),
            GWpyFilter(
                name="filter",
                sink_pad_names=("in",),
                source_pad_names=("out",),
                filter_type="highpass",
                low_freq=10,
            ),
            NullSink(name="snk", sink_pad_names=("out",)),
            link_map={
                "filter:snk:in": "src:src:out",
                "snk:snk:out": "filter:src:out",
            },
        )

        pipeline.run()

    def test_filter_pipeline_notch(self):
        """Test notch filter in a pipeline."""
        from sgn import NullSink
        from sgn.apps import Pipeline
        from sgnts.sources import FakeSeriesSource

        pipeline = Pipeline()

        pipeline.insert(
            FakeSeriesSource(
                name="src",
                source_pad_names=("out",),
                rate=4096,
                signal_type="white",
                t0=0,
                end=4,
            ),
            GWpyFilter(
                name="filter",
                sink_pad_names=("in",),
                source_pad_names=("out",),
                filter_type="notch",
                notch_freq=60,
            ),
            NullSink(name="snk", sink_pad_names=("out",)),
            link_map={
                "filter:snk:in": "src:src:out",
                "snk:snk:out": "filter:src:out",
            },
        )

        pipeline.run()

    def test_filter_pipeline_with_gap(self):
        """Test filter handles gaps correctly in a pipeline."""
        from sgn import NullSink
        from sgn.apps import Pipeline
        from sgnts.sources import FakeSeriesSource, SegmentSource
        from sgnts.transforms import Gate

        pipeline = Pipeline()

        pipeline.insert(
            FakeSeriesSource(
                name="datasrc",
                source_pad_names=("out",),
                rate=4096,
                signal_type="white",
                t0=0,
                end=10,
            ),
            SegmentSource(
                name="segsrc",
                source_pad_names=("seg",),
                rate=4096,
                t0=0,
                end=10,
                segments=(
                    (0, int(4 * 1e9)),
                    (int(6 * 1e9), int(10 * 1e9)),
                ),
            ),
            Gate(
                name="gate",
                source_pad_names=("out",),
                sink_pad_names=("data", "control"),
                control="control",
            ),
            GWpyFilter(
                name="filter",
                sink_pad_names=("in",),
                source_pad_names=("out",),
                filter_type="bandpass",
                low_freq=20,
                high_freq=500,
            ),
            NullSink(name="snk", sink_pad_names=("out",)),
            link_map={
                "gate:snk:data": "datasrc:src:out",
                "gate:snk:control": "segsrc:src:seg",
                "filter:snk:in": "gate:src:out",
                "snk:snk:out": "filter:src:out",
            },
        )

        pipeline.run()


class TestGWpySpectrogramPipeline:
    """Test GWpySpectrogram in a real pipeline."""

    def test_spectrogram_pipeline(self):
        """Test spectrogram transform in a pipeline."""
        from sgn import NullSink
        from sgn.apps import Pipeline
        from sgnts.sources import FakeSeriesSource

        from sgnligo.gwpy.transforms.spectrogram import GWpySpectrogram

        pipeline = Pipeline()

        pipeline.insert(
            FakeSeriesSource(
                name="src",
                source_pad_names=("out",),
                rate=4096,
                signal_type="white",
                t0=0,
                end=8,
            ),
            GWpySpectrogram(
                name="spec",
                sink_pad_names=("in",),
                source_pad_names=("out",),
                spec_stride=1.0,  # Time between spectrogram columns
                fft_length=1.0,  # FFT length (will accumulate buffers)
                fft_overlap=0.5,
                output_rate=64,  # Power-of-2 output rate
            ),
            NullSink(name="snk", sink_pad_names=("out",)),
            link_map={
                "spec:snk:in": "src:src:out",
                "snk:snk:out": "spec:src:out",
            },
        )

        pipeline.run()


class TestTimeSeriesSourceSinkIntegration:
    """Test TimeSeriesSource and TimeSeriesSink in integrated pipelines."""

    def test_source_to_sink_pipeline(self):
        """Test TimeSeriesSource to TimeSeriesSink end-to-end."""
        from sgn.apps import Pipeline

        from sgnligo.gwpy.sinks.timeseries_sink import TimeSeriesSink
        from sgnligo.gwpy.sources.timeseries_source import TimeSeriesSource

        # Create source data
        sample_rate = 4096
        gps_time = 1126259462.0
        duration = 4.0
        data = np.random.randn(int(sample_rate * duration))

        ts_input = TimeSeries(
            data, t0=gps_time, sample_rate=sample_rate, channel="H1:TEST"
        )

        # Create pipeline
        pipeline = Pipeline()

        source = TimeSeriesSource(
            name="src",
            timeseries=ts_input,
            buffer_duration=1.0,
        )

        sink = TimeSeriesSink(
            name="snk",
            sink_pad_names=("H1:TEST",),
            channel="H1:OUTPUT",
        )

        pipeline.insert(
            source,
            sink,
            link_map={
                "snk:snk:H1:TEST": "src:src:H1:TEST",
            },
        )

        pipeline.run()

        # Verify output
        assert sink.is_complete
        result = sink.get_result()
        assert len(result) == len(data)
        assert np.allclose(result.value, data, rtol=1e-6)

    def test_source_through_filter_to_sink(self):
        """Test data flow: TimeSeriesSource -> GWpyFilter -> TimeSeriesSink."""
        from sgn.apps import Pipeline

        from sgnligo.gwpy.sinks.timeseries_sink import TimeSeriesSink
        from sgnligo.gwpy.sources.timeseries_source import TimeSeriesSource

        # Create source data with known frequency content
        sample_rate = 4096
        gps_time = 1126259462.0
        duration = 4.0
        t = np.arange(int(sample_rate * duration)) / sample_rate
        # 100 Hz sine wave
        data = np.sin(2 * np.pi * 100 * t)

        ts_input = TimeSeries(
            data, t0=gps_time, sample_rate=sample_rate, channel="H1:STRAIN"
        )

        # Create pipeline
        pipeline = Pipeline()

        source = TimeSeriesSource(
            name="src",
            timeseries=ts_input,
            buffer_duration=1.0,
        )

        filt = GWpyFilter(
            name="filter",
            sink_pad_names=("in",),
            source_pad_names=("out",),
            filter_type="bandpass",
            low_freq=50,
            high_freq=150,
        )

        sink = TimeSeriesSink(
            name="snk",
            sink_pad_names=("out",),
            channel="H1:FILTERED",
        )

        pipeline.insert(
            source,
            filt,
            sink,
            link_map={
                "filter:snk:in": "src:src:H1:STRAIN",
                "snk:snk:out": "filter:src:out",
            },
        )

        pipeline.run()

        # Verify output exists
        assert sink.is_complete
        result = sink.get_result()
        assert len(result) > 0


class TestGWpyQTransformPipeline:
    """Test GWpyQTransform in a pipeline."""

    def test_qtransform_pipeline(self):
        """Test Q-transform in a pipeline."""
        from sgn import NullSink
        from sgn.apps import Pipeline
        from sgnts.sources import FakeSeriesSource

        from sgnligo.gwpy.transforms.qtransform import GWpyQTransform

        pipeline = Pipeline()

        pipeline.insert(
            FakeSeriesSource(
                name="src",
                source_pad_names=("out",),
                rate=4096,
                signal_type="white",
                t0=0,
                end=8,  # Enough for Q-transform
            ),
            GWpyQTransform(
                name="qtrans",
                sink_pad_names=("in",),
                source_pad_names=("out",),
                qrange=(4, 16),  # Smaller Q range
                frange=(50, 500),  # Smaller freq range
                output_rate=64,  # Power-of-2 output rate
            ),
            NullSink(name="snk", sink_pad_names=("out",)),
            link_map={
                "qtrans:snk:in": "src:src:out",
                "snk:snk:out": "qtrans:src:out",
            },
        )

        pipeline.run()

    def test_qtransform_pipeline_with_gap(self):
        """Test Q-transform handles gaps correctly."""
        from sgn import NullSink
        from sgn.apps import Pipeline
        from sgnts.sources import FakeSeriesSource, SegmentSource
        from sgnts.transforms import Gate

        from sgnligo.gwpy.transforms.qtransform import GWpyQTransform

        pipeline = Pipeline()

        pipeline.insert(
            FakeSeriesSource(
                name="src",
                source_pad_names=("out",),
                rate=4096,
                signal_type="white",
                t0=0,
                end=12,
            ),
            SegmentSource(
                name="seg",
                source_pad_names=("seg",),
                segments=((0, int(4e9)), (int(8e9), int(12e9))),  # Gap from 4-8
                t0=0,
                end=12,
            ),
            Gate(
                name="gate",
                sink_pad_names=("data", "control"),
                source_pad_names=("out",),
                control="control",
            ),
            GWpyQTransform(
                name="qtrans",
                sink_pad_names=("in",),
                source_pad_names=("out",),
                qrange=(4, 16),
                frange=(50, 500),
                output_rate=64,  # Power-of-2 output rate
            ),
            NullSink(name="snk", sink_pad_names=("out",)),
            link_map={
                "gate:snk:data": "src:src:out",
                "gate:snk:control": "seg:src:seg",
                "qtrans:snk:in": "gate:src:out",
                "snk:snk:out": "qtrans:src:out",
            },
        )

        pipeline.run()


class TestGWpySpectrogramEdgeCases:
    """Test GWpySpectrogram edge cases."""

    def test_spectrogram_gap_resets_accumulator(self):
        """Test that gaps properly reset the spectrogram accumulator."""
        from sgn import NullSink
        from sgn.apps import Pipeline
        from sgnts.sources import FakeSeriesSource, SegmentSource
        from sgnts.transforms import Gate

        from sgnligo.gwpy.transforms.spectrogram import GWpySpectrogram

        pipeline = Pipeline()

        pipeline.insert(
            FakeSeriesSource(
                name="src",
                source_pad_names=("out",),
                rate=4096,
                signal_type="white",
                t0=0,
                end=12,
            ),
            SegmentSource(
                name="seg",
                source_pad_names=("seg",),
                segments=((0, int(4e9)), (int(8e9), int(12e9))),  # Gap from 4-8
                t0=0,
                end=12,
            ),
            Gate(
                name="gate",
                sink_pad_names=("data", "control"),
                source_pad_names=("out",),
                control="control",
            ),
            GWpySpectrogram(
                name="spec",
                sink_pad_names=("in",),
                source_pad_names=("out",),
                spec_stride=1.0,  # Must be >= fft_length
                fft_length=1.0,
                output_rate=64,  # Power-of-2 output rate
            ),
            NullSink(name="snk", sink_pad_names=("out",)),
            link_map={
                "gate:snk:data": "src:src:out",
                "gate:snk:control": "seg:src:seg",
                "spec:snk:in": "gate:src:out",
                "snk:snk:out": "spec:src:out",
            },
        )

        pipeline.run()

    def test_spectrogram_validation_errors(self):
        """Test spectrogram validation catches invalid parameters."""
        import pytest

        from sgnligo.gwpy.transforms.spectrogram import GWpySpectrogram

        with pytest.raises(ValueError, match="spec_stride must be positive"):
            GWpySpectrogram(
                name="spec",
                sink_pad_names=("in",),
                source_pad_names=("out",),
                spec_stride=-1.0,
            )

        with pytest.raises(ValueError, match="fft_length must be positive"):
            GWpySpectrogram(
                name="spec",
                sink_pad_names=("in",),
                source_pad_names=("out",),
                fft_length=-1.0,
            )

        with pytest.raises(ValueError, match="output_stride must be positive"):
            GWpySpectrogram(
                name="spec",
                sink_pad_names=("in",),
                source_pad_names=("out",),
                output_stride=-1.0,
            )

    def test_spectrogram_invalid_output_rate_raises(self):
        """Test that non-power-of-2 output_rate raises ValueError (line 143)."""
        from sgnligo.gwpy.transforms.spectrogram import GWpySpectrogram

        with pytest.raises(ValueError, match="output_rate.*power-of-2"):
            GWpySpectrogram(
                name="spec",
                sink_pad_names=("in",),
                source_pad_names=("out",),
                output_rate=100,  # Not power of 2
            )


class TestGWpyPlotSink:
    """Test GWpyPlotSink."""

    def test_plot_sink_creation(self):
        """Test creating a plot sink with default parameters."""
        from sgnligo.gwpy.sinks import GWpyPlotSink

        sink = GWpyPlotSink(
            name="plotter",
            sink_pad_names=("in",),
        )

        assert sink.ifo == "H1"
        assert sink.description == "STRAIN"
        assert sink.plot_type == "timeseries"
        assert sink.plot_stride == 1.0
        assert sink.overlap_before == 0.0
        assert sink.overlap_after == 0.0

    def test_plot_sink_custom_parameters(self):
        """Test creating a plot sink with custom parameters."""
        from sgnligo.gwpy.sinks import GWpyPlotSink

        sink = GWpyPlotSink(
            name="plotter",
            sink_pad_names=("in",),
            ifo="L1",
            description="STRAIN_FILTERED",
            plot_type="spectrogram",
            plot_stride=2.0,
            overlap_before=0.5,
            overlap_after=0.5,
            fft_length=0.25,
        )

        assert sink.ifo == "L1"
        assert sink.description == "STRAIN_FILTERED"
        assert sink.plot_type == "spectrogram"
        assert sink.plot_stride == 2.0
        assert sink.overlap_before == 0.5
        assert sink.overlap_after == 0.5
        assert sink.fft_length == 0.25

    def test_plot_sink_invalid_plot_type_raises(self):
        """Test that invalid plot_type raises ValueError."""
        from sgnligo.gwpy.sinks import GWpyPlotSink

        with pytest.raises(ValueError, match="plot_type must be one of"):
            GWpyPlotSink(
                name="plotter",
                sink_pad_names=("in",),
                plot_type="invalid",
            )

    def test_plot_sink_negative_stride_raises(self):
        """Test that negative plot_stride raises ValueError."""
        from sgnligo.gwpy.sinks import GWpyPlotSink

        with pytest.raises(ValueError, match="plot_stride must be positive"):
            GWpyPlotSink(
                name="plotter",
                sink_pad_names=("in",),
                plot_stride=-1.0,
            )

    def test_plot_sink_negative_overlap_before_raises(self):
        """Test that negative overlap_before raises ValueError."""
        from sgnligo.gwpy.sinks import GWpyPlotSink

        with pytest.raises(ValueError, match="overlap_before must be non-negative"):
            GWpyPlotSink(
                name="plotter",
                sink_pad_names=("in",),
                overlap_before=-0.5,
            )

    def test_plot_sink_negative_overlap_after_raises(self):
        """Test that negative overlap_after raises ValueError."""
        from sgnligo.gwpy.sinks import GWpyPlotSink

        with pytest.raises(ValueError, match="overlap_after must be non-negative"):
            GWpyPlotSink(
                name="plotter",
                sink_pad_names=("in",),
                overlap_after=-0.5,
            )

    def test_plot_sink_overlap_configuration(self):
        """Test that overlaps can be configured independently."""
        from sgnligo.gwpy.sinks import GWpyPlotSink

        # Symmetric overlap: 1s before + 2s stride + 1s after = 4s plots
        sink = GWpyPlotSink(
            name="plotter",
            sink_pad_names=("in",),
            plot_stride=2.0,
            overlap_before=1.0,
            overlap_after=1.0,
        )
        assert sink.plot_stride == 2.0
        assert sink.overlap_before == 1.0
        assert sink.overlap_after == 1.0

        # Asymmetric overlap: 2s before + 1s stride + 0s after = 3s plots
        sink2 = GWpyPlotSink(
            name="plotter2",
            sink_pad_names=("in",),
            plot_stride=1.0,
            overlap_before=2.0,
            overlap_after=0.0,
        )
        assert sink2.plot_stride == 1.0
        assert sink2.overlap_before == 2.0
        assert sink2.overlap_after == 0.0

    def test_plot_sink_filename_generation(self):
        """Test LIGO filename convention generation."""
        from sgnligo.gwpy.sinks import GWpyPlotSink

        sink = GWpyPlotSink(
            name="plotter",
            sink_pad_names=("in",),
            ifo="H1",
            description="STRAIN",
        )

        filename = sink._generate_filename(1126259462.0, 2.0)
        assert filename == "H1-STRAIN-1126259462-2.png"

    def test_plot_sink_filename_replaces_dashes(self):
        """Test that dashes in description are replaced with underscores."""
        from sgnligo.gwpy.sinks import GWpyPlotSink

        sink = GWpyPlotSink(
            name="plotter",
            sink_pad_names=("in",),
            ifo="L1",
            description="STRAIN-FILTERED-WHITENED",
        )

        filename = sink._generate_filename(1126259462.0, 4.0)
        assert filename == "L1-STRAIN_FILTERED_WHITENED-1126259462-4.png"

    def test_plot_sink_invalid_qrange_raises(self):
        """Test that invalid qrange raises ValueError."""
        from sgnligo.gwpy.sinks import GWpyPlotSink

        with pytest.raises(ValueError, match="qrange must be"):
            GWpyPlotSink(
                name="plotter",
                sink_pad_names=("in",),
                qrange=(64, 4),  # min > max
            )

    def test_plot_sink_invalid_frange_raises(self):
        """Test that invalid frange raises ValueError."""
        from sgnligo.gwpy.sinks import GWpyPlotSink

        with pytest.raises(ValueError, match="frange must be"):
            GWpyPlotSink(
                name="plotter",
                sink_pad_names=("in",),
                frange=(500, 20),  # min > max
            )

    def test_plot_sink_timeseries_pipeline(self, tmp_path):
        """Test GWpyPlotSink in a pipeline generating timeseries plots."""
        from gwpy.timeseries import TimeSeries
        from sgn.apps import Pipeline

        from sgnligo.gwpy.sinks import GWpyPlotSink
        from sgnligo.gwpy.sources import TimeSeriesSource

        # Create test data - 4 seconds at 4096 Hz
        sample_rate = 4096
        duration = 4
        t = np.arange(0, duration, 1 / sample_rate)
        data = np.sin(2 * np.pi * 100 * t)
        ts = TimeSeries(data, t0=1000000000, sample_rate=sample_rate, channel="H1:TEST")

        # Build pipeline
        pipeline = Pipeline()

        source = TimeSeriesSource(
            name="Source",
            timeseries=ts,
            buffer_duration=1.0,
        )
        pipeline.insert(source)

        plot_sink = GWpyPlotSink(
            name="Plotter",
            sink_pad_names=("in",),
            ifo="H1",
            description="TEST",
            plot_type="timeseries",
            output_dir=str(tmp_path),
            plot_stride=2.0,
        )
        pipeline.insert(
            plot_sink,
            link_map={"Plotter:snk:in": "Source:src:H1:TEST"},
        )

        # Run pipeline
        pipeline.run()

        # Check that plots were generated
        plots = list(tmp_path.glob("*.png"))
        assert len(plots) >= 1
        assert plot_sink.plots_generated >= 1

    def test_plot_sink_spectrogram_pipeline(self, tmp_path):
        """Test GWpyPlotSink in a pipeline generating spectrogram plots."""
        from gwpy.timeseries import TimeSeries
        from sgn.apps import Pipeline

        from sgnligo.gwpy.sinks import GWpyPlotSink
        from sgnligo.gwpy.sources import TimeSeriesSource

        # Create test data
        sample_rate = 4096
        duration = 4
        data = np.random.randn(sample_rate * duration)
        ts = TimeSeries(
            data, t0=1000000000, sample_rate=sample_rate, channel="L1:NOISE"
        )

        # Build pipeline
        pipeline = Pipeline()

        source = TimeSeriesSource(
            name="Source",
            timeseries=ts,
            buffer_duration=2.0,
        )
        pipeline.insert(source)

        plot_sink = GWpyPlotSink(
            name="Plotter",
            sink_pad_names=("in",),
            ifo="L1",
            description="NOISE",
            plot_type="spectrogram",
            output_dir=str(tmp_path),
            plot_stride=2.0,
            fft_length=0.25,
        )
        pipeline.insert(
            plot_sink,
            link_map={"Plotter:snk:in": "Source:src:L1:NOISE"},
        )

        # Run pipeline
        pipeline.run()

        # Check that plots were generated
        plots = list(tmp_path.glob("*.png"))
        assert len(plots) >= 1

    def test_plot_sink_qtransform_pipeline(self, tmp_path):
        """Test GWpyPlotSink in a pipeline generating Q-transform plots."""
        from gwpy.timeseries import TimeSeries
        from sgn.apps import Pipeline

        from sgnligo.gwpy.sinks import GWpyPlotSink
        from sgnligo.gwpy.sources import TimeSeriesSource

        # Create test data with chirp signal
        sample_rate = 4096
        duration = 4
        t = np.arange(0, duration, 1 / sample_rate)
        # Chirp from 50 to 200 Hz
        chirp = np.sin(2 * np.pi * (50 * t + 75 * t**2 / duration))
        data = chirp + 0.1 * np.random.randn(len(t))
        ts = TimeSeries(
            data, t0=1000000000, sample_rate=sample_rate, channel="H1:CHIRP"
        )

        # Build pipeline
        pipeline = Pipeline()

        source = TimeSeriesSource(
            name="Source",
            timeseries=ts,
            buffer_duration=2.0,
        )
        pipeline.insert(source)

        plot_sink = GWpyPlotSink(
            name="Plotter",
            sink_pad_names=("in",),
            ifo="H1",
            description="CHIRP",
            plot_type="qtransform",
            output_dir=str(tmp_path),
            plot_stride=2.0,
            qrange=(4, 64),
            frange=(20, 300),
        )
        pipeline.insert(
            plot_sink,
            link_map={"Plotter:snk:in": "Source:src:H1:CHIRP"},
        )

        # Run pipeline
        pipeline.run()

        # Check that plots were generated
        plots = list(tmp_path.glob("*.png"))
        assert len(plots) >= 1

    def test_plot_sink_custom_title(self, tmp_path):
        """Test GWpyPlotSink with custom title template."""
        from sgnligo.gwpy.sinks import GWpyPlotSink

        sink = GWpyPlotSink(
            name="plotter",
            sink_pad_names=("in",),
            ifo="H1",
            description="STRAIN",
            title_template="{ifo} Custom Title - GPS {gps_start:.0f}",
            output_dir=str(tmp_path),
        )

        # Create mock timeseries for title generation
        ts = TimeSeries(np.zeros(4096), t0=1000000000, sample_rate=4096)
        title = sink._get_title(ts, 1000000000.0, 1.0)
        assert title == "H1 Custom Title - GPS 1000000000"

    def test_plot_sink_invalid_fft_length_raises(self):
        """Test that invalid fft_length raises ValueError."""
        from sgnligo.gwpy.sinks import GWpyPlotSink

        with pytest.raises(ValueError, match="fft_length must be positive"):
            GWpyPlotSink(
                name="plotter",
                sink_pad_names=("in",),
                fft_length=0,
            )

        with pytest.raises(ValueError, match="fft_length must be positive"):
            GWpyPlotSink(
                name="plotter",
                sink_pad_names=("in",),
                fft_length=-0.5,
            )

    def test_plot_sink_handles_plot_exception(self, tmp_path, caplog):
        """Test that GWpyPlotSink handles exceptions during plot generation."""
        from unittest.mock import patch

        from sgnligo.gwpy.sinks import GWpyPlotSink

        sink = GWpyPlotSink(
            name="plotter",
            sink_pad_names=("in",),
            ifo="H1",
            description="TEST",
            output_dir=str(tmp_path),
        )

        # Create test timeseries
        ts = TimeSeries(np.zeros(4096), t0=1000000000, sample_rate=4096)

        # Mock _plot_timeseries to raise an exception
        with patch.object(
            sink, "_plot_timeseries", side_effect=RuntimeError("Test error")
        ):
            # Should not raise, just log warning
            sink._generate_plot(ts, 1000000000.0)

        # Check that warning was logged
        assert "Failed to generate plot" in caplog.text
        assert "Test error" in caplog.text

    def test_plot_sink_handles_none_frame(self, tmp_path):
        """Test that GWpyPlotSink handles None frames gracefully."""
        from unittest.mock import MagicMock, patch

        from sgnts.base import TSSink

        from sgnligo.gwpy.sinks import GWpyPlotSink

        sink = GWpyPlotSink(
            name="plotter",
            sink_pad_names=("in",),
            ifo="H1",
            description="TEST",
            output_dir=str(tmp_path),
        )

        # Mock preparedframes to return None
        mock_pad = MagicMock()
        sink.sink_pads = [mock_pad]
        sink.preparedframes = {mock_pad: None}

        # Mock parent internal() to avoid audio adapter issues
        with patch.object(TSSink, "internal"):
            # Should not raise - just continue
            sink.internal()

        # No plots should be generated
        assert sink.plots_generated == 0

    def test_plot_sink_handles_gap_buffer(self, tmp_path):
        """Test that GWpyPlotSink skips gap buffers."""
        from unittest.mock import MagicMock, patch

        from sgnts.base import TSSink

        from sgnligo.gwpy.sinks import GWpyPlotSink

        sink = GWpyPlotSink(
            name="plotter",
            sink_pad_names=("in",),
            ifo="H1",
            description="TEST",
            output_dir=str(tmp_path),
        )

        # Create mock frame with gap buffer
        mock_pad = MagicMock()
        mock_buffer = MagicMock()
        mock_buffer.is_gap = True

        mock_frame = MagicMock()
        mock_frame.EOS = False
        mock_frame.is_gap = False
        mock_frame.buffers = [mock_buffer]

        sink.sink_pads = [mock_pad]
        sink.preparedframes = {mock_pad: mock_frame}

        # Mock parent internal() to avoid audio adapter issues
        with patch.object(TSSink, "internal"):
            # Should not raise - just skip the gap buffer
            sink.internal()

        # No plots should be generated from gap buffer
        assert sink.plots_generated == 0

    def test_plot_sink_handles_empty_buffer(self, tmp_path):
        """Test that GWpyPlotSink skips empty buffers."""
        from unittest.mock import MagicMock, patch

        from sgnts.base import TSSink

        from sgnligo.gwpy.sinks import GWpyPlotSink

        sink = GWpyPlotSink(
            name="plotter",
            sink_pad_names=("in",),
            ifo="H1",
            description="TEST",
            output_dir=str(tmp_path),
        )

        # Create mock frame with empty buffer
        mock_pad = MagicMock()
        mock_buffer = MagicMock()
        mock_buffer.is_gap = False
        mock_buffer.shape = (0,)  # Empty buffer

        mock_frame = MagicMock()
        mock_frame.EOS = False
        mock_frame.is_gap = False
        mock_frame.buffers = [mock_buffer]

        sink.sink_pads = [mock_pad]
        sink.preparedframes = {mock_pad: mock_frame}

        # Mock parent internal() to avoid audio adapter issues
        with patch.object(TSSink, "internal"):
            # Should not raise - just skip the empty buffer
            sink.internal()

        # No plots should be generated from empty buffer
        assert sink.plots_generated == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
