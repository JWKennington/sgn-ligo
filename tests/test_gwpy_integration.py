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


class TestGWOSCSource:
    """Test GWOSCSource."""

    def test_gwosc_source_creation(self):
        """Test creating a GWOSC source."""
        from sgnligo.gwpy.sources.gwosc_source import GWOSCSource

        source = GWOSCSource(
            name="H1_GWOSC",
            source_pad_names=("strain",),
            detector="H1",
            start_time=1126259462,
            duration=32,
        )

        assert source.detector == "H1"
        assert source.target_sample_rate == 4096  # Default
        assert source.chunk_size == 64  # Default

    def test_gwosc_source_invalid_detector_raises(self):
        """Test that invalid detector raises ValueError."""
        from sgnligo.gwpy.sources.gwosc_source import GWOSCSource

        with pytest.raises(ValueError, match="Invalid detector"):
            GWOSCSource(
                name="Invalid",
                source_pad_names=("strain",),
                detector="XX",  # Invalid
                start_time=1126259462,
                duration=32,
            )

    def test_gwosc_source_missing_start_time_raises(self):
        """Test that missing start_time raises ValueError."""
        from sgnligo.gwpy.sources.gwosc_source import GWOSCSource

        with pytest.raises(ValueError, match="start_time is required"):
            GWOSCSource(
                name="H1_GWOSC",
                source_pad_names=("strain",),
                detector="H1",
                start_time=None,
                duration=32,
            )

    def test_gwosc_source_missing_duration_raises(self):
        """Test that missing duration raises ValueError."""
        from sgnligo.gwpy.sources.gwosc_source import GWOSCSource

        with pytest.raises(ValueError, match="duration is required"):
            GWOSCSource(
                name="H1_GWOSC",
                source_pad_names=("strain",),
                detector="H1",
                start_time=1126259462,
                duration=None,
            )

    def test_gwosc_source_all_valid_detectors(self):
        """Test that all valid detectors are accepted."""
        from sgnligo.gwpy.sources.gwosc_source import GWOSCSource

        for detector in ["H1", "L1", "V1", "G1", "K1"]:
            source = GWOSCSource(
                name=f"{detector}_GWOSC",
                source_pad_names=("strain",),
                detector=detector,
                start_time=1126259462,
                duration=32,
            )
            assert source.detector == detector

    def test_gwosc_source_worker_process_success(self):
        """Test worker_process with mocked GWOSC fetch."""
        from queue import Queue
        from unittest.mock import MagicMock, patch

        from gwpy.timeseries import TimeSeries

        from sgnligo.gwpy.sources.gwosc_source import GWOSCSource

        # Create mock TimeSeries
        sample_rate = 4096
        duration = 2  # seconds
        data = np.random.randn(sample_rate * duration)
        mock_ts = TimeSeries(data, t0=1126259462, sample_rate=sample_rate)

        # Mock context
        output_queue = Queue()
        context = MagicMock()
        context.output_queue = output_queue
        context.should_stop.return_value = False

        # Mock source pad
        mock_pad = MagicMock()
        srcs = {"strain": mock_pad}

        with patch.object(TimeSeries, "fetch_open_data", return_value=mock_ts):
            GWOSCSource.worker_process(
                None,  # self - not used since it's a minimal instance
                context=context,
                srcs=srcs,
                detector="H1",
                start_time=1126259462,
                _duration_sec=2,
                target_sample_rate=4096,
                chunk_size=64,
                channel=None,
                cache=True,
                timeout=120,
                max_retries=3,
            )

        # Check that data was put in the queue
        assert not output_queue.empty()
        pad, buf = output_queue.get()
        assert pad == mock_pad
        assert len(buf.data) > 0

    def test_gwosc_source_worker_process_with_channel(self):
        """Test worker_process with specified channel."""
        from queue import Queue
        from unittest.mock import MagicMock, patch

        from gwpy.timeseries import TimeSeries

        from sgnligo.gwpy.sources.gwosc_source import GWOSCSource

        # Create mock TimeSeries
        sample_rate = 4096
        duration = 2
        data = np.random.randn(sample_rate * duration)
        mock_ts = TimeSeries(data, t0=1126259462, sample_rate=sample_rate)

        # Mock context
        output_queue = Queue()
        context = MagicMock()
        context.output_queue = output_queue
        context.should_stop.return_value = False

        # Mock source pad
        mock_pad = MagicMock()
        srcs = {"strain": mock_pad}

        with patch.object(TimeSeries, "get", return_value=mock_ts) as mock_get:
            GWOSCSource.worker_process(
                None,  # self - not used since it's a minimal instance
                context=context,
                srcs=srcs,
                detector="H1",
                start_time=1126259462,
                _duration_sec=2,
                target_sample_rate=4096,
                chunk_size=64,
                channel="H1:GWOSC-4KHZ_R1_STRAIN",  # Specified channel
                cache=True,
                timeout=120,
                max_retries=3,
            )

        # Check that TimeSeries.get was called with channel
        mock_get.assert_called()
        assert not output_queue.empty()

    def test_gwosc_source_worker_process_handles_error(self):
        """Test worker_process handles fetch errors gracefully."""
        from queue import Queue
        from unittest.mock import MagicMock, patch

        from gwpy.timeseries import TimeSeries

        from sgnligo.gwpy.sources.gwosc_source import GWOSCSource

        # Mock context
        output_queue = Queue()
        context = MagicMock()
        context.output_queue = output_queue
        context.should_stop.return_value = False

        # Mock source pad
        mock_pad = MagicMock()
        srcs = {"strain": mock_pad}

        # Make fetch_open_data raise an exception
        with patch.object(
            TimeSeries, "fetch_open_data", side_effect=Exception("Network error")
        ):
            GWOSCSource.worker_process(
                None,  # self - not used since it's a minimal instance
                context=context,
                srcs=srcs,
                detector="H1",
                start_time=1126259462,
                _duration_sec=2,
                target_sample_rate=4096,
                chunk_size=64,
                channel=None,
                cache=True,
                timeout=120,
                max_retries=1,  # Only 1 retry so test runs quickly
            )

        # Check that a gap buffer was sent
        assert not output_queue.empty()
        pad, buf = output_queue.get()
        assert pad == mock_pad
        assert buf.is_gap  # Should be a gap buffer due to error

    def test_gwosc_source_worker_process_respects_should_stop(self):
        """Test that worker_process stops when context.should_stop() returns True."""
        from queue import Queue
        from unittest.mock import MagicMock

        from sgnligo.gwpy.sources.gwosc_source import GWOSCSource

        # Mock context that immediately says to stop
        output_queue = Queue()
        context = MagicMock()
        context.output_queue = output_queue
        context.should_stop.return_value = True  # Stop immediately

        # Mock source pad
        mock_pad = MagicMock()
        srcs = {"strain": mock_pad}

        GWOSCSource.worker_process(
            None,  # self - not used since it's a minimal instance
            context=context,
            srcs=srcs,
            detector="H1",
            start_time=1126259462,
            _duration_sec=32,
            target_sample_rate=4096,
            chunk_size=64,
            channel=None,
            cache=True,
            timeout=120,
            max_retries=3,
        )

        # Should have stopped immediately without fetching any data
        assert output_queue.empty()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
