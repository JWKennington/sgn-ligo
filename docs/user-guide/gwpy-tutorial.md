# GWpy Integration Tutorial

## Overview

The `sgnligo.gwpy` module provides seamless integration between SGN streaming pipelines and [GWpy](https://gwpy.github.io/docs/stable/), the standard gravitational wave data analysis library. This enables you to:

- Use GWpy's powerful analysis methods within streaming pipelines
- Convert data between SGN's `SeriesBuffer` and GWpy's `TimeSeries`
- Produce output ready for further GWpy-based analysis

This tutorial covers:

1. Data conversion utilities
2. Filtering with GWpyFilter
3. Time-frequency analysis with GWpyQTransform and GWpySpectrogram
4. Data sources: TimeSeriesSource
5. Output collection with TimeSeriesSink
6. Streaming plot generation with GWpyPlotSink
7. Complete pipeline examples

!!! note "PSD and Whitening"
    For power spectral density computation and whitening, use the native SGN-LIGO transforms in `sgnligo.transforms` and `sgnligo.psd` which provide optimized streaming implementations.

## Installation Requirements

The GWpy integration requires GWpy to be installed:

```bash
pip install gwpy
```

All GWpy components are available under `sgnligo.gwpy`:

```python
from sgnligo.gwpy.converters import (
    seriesbuffer_to_timeseries,
    timeseries_to_seriesbuffer,
    tsframe_to_timeseries,
    timeseries_to_tsframe,
)
from sgnligo.gwpy.transforms import (
    GWpyFilter,
    GWpyQTransform,
    GWpySpectrogram,
)
from sgnligo.gwpy.sources import TimeSeriesSource
from sgnligo.gwpy.sinks import TimeSeriesSink, GWpyPlotSink
```

## Data Conversion Utilities

The `converters` module provides bidirectional conversion between SGN data containers and GWpy objects.

### SeriesBuffer to TimeSeries

Convert a single buffer to a GWpy TimeSeries:

```python
import numpy as np
from sgnts.base import Offset, SeriesBuffer
from sgnligo.gwpy.converters import seriesbuffer_to_timeseries

# Create a buffer with sample data
data = np.random.randn(4096)
buf = SeriesBuffer(
    offset=Offset.fromsec(1126259462),  # GW150914 GPS time
    sample_rate=4096,
    data=data,
    shape=data.shape,
)

# Convert to GWpy TimeSeries
ts = seriesbuffer_to_timeseries(buf, channel="H1:STRAIN", unit="strain")

print(f"TimeSeries t0: {ts.t0}")
print(f"TimeSeries duration: {ts.duration}")
print(f"TimeSeries sample rate: {ts.sample_rate}")
```

Output:
```
TimeSeries t0: 1126259462.0 s
TimeSeries duration: 1.0 s
TimeSeries sample rate: 4096.0 Hz
```

### TimeSeries to SeriesBuffer

Convert a GWpy TimeSeries back to a SeriesBuffer:

```python
import numpy as np
from gwpy.timeseries import TimeSeries
from sgnligo.gwpy.converters import timeseries_to_seriesbuffer
from sgnts.base import Offset

# Create a TimeSeries
ts = TimeSeries(
    np.random.randn(4096),
    t0=1126259462,
    sample_rate=4096,
    channel="H1:STRAIN",
)

# Convert to SeriesBuffer
buf = timeseries_to_seriesbuffer(ts)

print(f"Buffer offset (GPS seconds): {Offset.tosec(buf.offset)}")
print(f"Buffer sample rate: {buf.sample_rate}")
print(f"Buffer shape: {buf.shape}")
```

### TSFrame Conversion

For frames containing multiple buffers:

```python
import numpy as np
from sgnts.base import TSFrame, SeriesBuffer, Offset
from sgnligo.gwpy.converters import tsframe_to_timeseries, timeseries_to_tsframe

# Create a TSFrame with multiple buffers
buf1 = SeriesBuffer(
    offset=Offset.fromsec(1126259462),
    sample_rate=4096,
    data=np.random.randn(4096),
    shape=(4096,),
)
buf2 = SeriesBuffer(
    offset=Offset.fromsec(1126259463),
    sample_rate=4096,
    data=np.random.randn(4096),
    shape=(4096,),
)
frame = TSFrame(buffers=[buf1, buf2])

# Convert TSFrame to single concatenated TimeSeries
ts = tsframe_to_timeseries(frame, channel="H1:PROCESSED")
print(f"Concatenated TimeSeries duration: {ts.duration}")

# Convert TimeSeries to TSFrame (creates single-buffer frame)
new_frame = timeseries_to_tsframe(ts)
print(f"New frame has {len(new_frame.buffers)} buffer(s)")
```

### Gap Handling

Gap buffers (missing data) are converted to NaN values, preserving timing information:

```python
import numpy as np
from sgnts.base import SeriesBuffer, Offset
from sgnligo.gwpy.converters import seriesbuffer_to_timeseries

# Gap buffer (data=None)
gap_buf = SeriesBuffer(
    offset=Offset.fromsec(1126259462),
    sample_rate=4096,
    data=None,
    shape=(4096,),
)

ts = seriesbuffer_to_timeseries(gap_buf)
print(f"All NaN: {np.all(np.isnan(ts.value))}")  # True
```

## GWpyFilter: Streaming Filtering

Apply bandpass, lowpass, highpass, or notch filters using GWpy's filtering methods.

### Bandpass Filter

```python skip
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from sgn.apps import Pipeline
from sgnligo.gwpy.transforms import GWpyFilter
from sgnts.sources import FakeSeriesSource
from sgnts.sinks import TSPlotSink

# Create pipeline
pipeline = Pipeline()

# Generate a 100 Hz sine wave
source = FakeSeriesSource(
    name="Source",
    source_pad_names=("signal",),
    signals={
        "signal": {"signal_type": "sin", "fsin": 100, "rate": 4096},
    },
    end=4,  # 4 seconds of data
)
pipeline.insert(source)

# Add bandpass filter (50-200 Hz) - passes the 100 Hz signal
bandpass = GWpyFilter(
    name="Bandpass",
    sink_pad_names=("in",),
    source_pad_names=("out",),
    filter_type="bandpass",
    low_freq=50,
    high_freq=200,
)
pipeline.insert(
    bandpass,
    link_map={"Bandpass:snk:in": "Source:src:signal"}
)

# TSPlotSink with two pads: original and filtered
sink = TSPlotSink(
    name="Comparison",
    sink_pad_names=("original", "filtered"),
)
pipeline.insert(
    sink,
    link_map={
        "Comparison:snk:original": "Source:src:signal",
        "Comparison:snk:filtered": "Bandpass:src:out",
    }
)

# Run pipeline
pipeline.run()

# Plot both signals overlaid (default) or as subplots
fig, ax = sink.plot(time_unit="s", layout="overlay")
ax.set_title("Bandpass Filter: 100 Hz sine through 50-200 Hz filter")
ax.legend()
plt.savefig("bandpass_filter_example.png", dpi=150)
plt.show()

# Or use subplots layout for clearer comparison
fig, axes = sink.plot(time_unit="s", layout="subplots")
plt.savefig("bandpass_filter_subplots.png", dpi=150)
plt.show()
```

### Notch Filter (Remove Power Line)

```python
from sgnligo.gwpy.transforms import GWpyFilter

notch = GWpyFilter(
    name="Notch60Hz",
    sink_pad_names=("in",),
    source_pad_names=("out",),
    filter_type="notch",
    notch_freq=60,
    notch_q=30,
)
```

### Highpass Filter

```python
from sgnligo.gwpy.transforms import GWpyFilter

highpass = GWpyFilter(
    name="Highpass",
    sink_pad_names=("in",),
    source_pad_names=("out",),
    filter_type="highpass",
    low_freq=20,
)
```

### Lowpass Filter

```python
from sgnligo.gwpy.transforms import GWpyFilter

lowpass = GWpyFilter(
    name="Lowpass",
    sink_pad_names=("in",),
    source_pad_names=("out",),
    filter_type="lowpass",
    high_freq=500,
)
```

## GWpyQTransform: Time-Frequency Q-Transform

The Q-transform provides a time-frequency representation with constant-Q frequency resolution, ideal for visualizing transients like gravitational wave signals.

### Basic Q-Transform

```python
from sgnligo.gwpy.transforms import GWpyQTransform

qtrans = GWpyQTransform(
    name="QTransform",
    sink_pad_names=("in",),
    source_pad_names=("out",),
    qrange=(4, 64),         # Range of Q values
    frange=(20, 500),       # Frequency range in Hz
    output_rate=64,         # Output sample rate (power-of-2)
    output_stride=1.0,      # Output every 1 second
    input_sample_rate=4096, # Expected input rate
)
```

!!! note "Power-of-2 Output Rates"
    Output sample rates must be from `Offset.ALLOWED_RATES`: {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384}. This ensures proper offset alignment in the streaming framework.

### Q-Transform Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `qrange` | (min, max) Q values | (4, 64) |
| `frange` | (min, max) frequency in Hz | (20, 1024) |
| `mismatch` | Maximum tile mismatch | 0.2 |
| `logf` | Logarithmic frequency spacing | True |
| `tres` | Time resolution (auto if None) | None |
| `fres` | Frequency resolution (auto if None) | None |
| `output_rate` | Output sample rate (power-of-2) | 64 |
| `output_stride` | Output duration per cycle (s) | 1.0 |

### Q-Transform Output

The Q-transform produces 2D output (frequency × time):

```python
# Output buffer shape: (n_frequencies, n_times)
# Metadata available:
# - metadata["qtransform"]: Full GWpy Spectrogram object
# - metadata["q_frequencies"]: Frequency array
# - metadata["q_times"]: Time array
# - metadata["q_qrange"]: Q range used
# - metadata["q_frange"]: Frequency range used
```

### Complete Q-Transform Example

```python
from sgn.apps import Pipeline
from sgnligo.gwpy.transforms import GWpyQTransform
from sgnligo.gwpy.sources import TimeSeriesSource
from sgnligo.gwpy.sinks import TimeSeriesSink
from gwpy.timeseries import TimeSeries
import numpy as np

# Create test signal with a chirp
sample_rate = 4096
duration = 4
t = np.arange(0, duration, 1/sample_rate)
# Chirp from 50 to 200 Hz
chirp = np.sin(2 * np.pi * (50 * t + 75 * t**2 / duration))
noise = 0.1 * np.random.randn(len(t))
signal = chirp + noise

ts = TimeSeries(signal, t0=1000000000, sample_rate=sample_rate, channel="CHIRP")

pipeline = Pipeline()

source = TimeSeriesSource(name="Source", timeseries=ts, buffer_duration=1.0)
pipeline.insert(source)

qtrans = GWpyQTransform(
    name="QTrans",
    sink_pad_names=("in",),
    source_pad_names=("out",),
    qrange=(4, 64),
    frange=(20, 300),
    output_rate=64,
    output_stride=1.0,
    input_sample_rate=sample_rate,
)
pipeline.insert(qtrans, link_map={"QTrans:snk:in": "Source:src:CHIRP"})

# For 2D data, use TimeSeriesSink or custom sink
sink = TimeSeriesSink(name="Sink", sink_pad_names=("in",), channel="QTRANS")
pipeline.insert(sink, link_map={"Sink:snk:in": "QTrans:src:out"})

pipeline.run()
print("Q-transform pipeline complete")
```

## GWpySpectrogram: Time-Frequency Spectrogram

Compute standard FFT-based spectrograms for streaming data.

### Basic Spectrogram

```python
from sgnligo.gwpy.transforms import GWpySpectrogram

spec = GWpySpectrogram(
    name="Spectrogram",
    sink_pad_names=("in",),
    source_pad_names=("out",),
    spec_stride=1.0,        # Time step between columns
    fft_length=2.0,         # FFT length in seconds
    fft_overlap=1.0,        # Overlap between FFTs
    window="hann",          # Window function
    output_rate=64,         # Output sample rate (power-of-2)
    output_stride=1.0,      # Output duration per cycle
    input_sample_rate=4096,
)
```

### Spectrogram Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `spec_stride` | Time step between columns (s) | 1.0 |
| `fft_length` | FFT length in seconds | 2.0 |
| `fft_overlap` | Overlap between FFTs (s) | fft_length/2 |
| `window` | Window function | 'hann' |
| `nproc` | Parallel processes | 1 |
| `output_rate` | Output sample rate (power-of-2) | auto |
| `output_stride` | Output duration per cycle (s) | 1.0 |

### Spectrogram Output

```python
# Output buffer shape: (n_frequencies, n_times)
# Metadata available:
# - metadata["spectrogram"]: Full GWpy Spectrogram object
# - metadata["spec_frequencies"]: Frequency array
# - metadata["spec_times"]: Time array
# - metadata["spec_df"]: Frequency resolution
# - metadata["spec_dt"]: Time resolution
```

## TimeSeriesSource: GWpy Data to Pipeline

Feed an existing GWpy TimeSeries or TimeSeriesDict into a streaming pipeline.

### Single TimeSeries

```python
import numpy as np
from gwpy.timeseries import TimeSeries
from sgnligo.gwpy.sources import TimeSeriesSource

# Load data with GWpy (example with local data)
ts = TimeSeries(
    np.random.randn(4096 * 10),
    t0=1126259462,
    sample_rate=4096,
    channel="H1:STRAIN",
)

# Create source for pipeline
source = TimeSeriesSource(
    name="H1_Data",
    timeseries=ts,
    buffer_duration=1.0,  # Output 1-second buffers
)
```

### Multi-Channel TimeSeriesDict

```python
import numpy as np
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from sgnligo.gwpy.sources import TimeSeriesSource

# Create multi-channel data
tsd = TimeSeriesDict()
tsd["H1:STRAIN"] = TimeSeries(np.random.randn(4096*10), t0=1000000000, sample_rate=4096)
tsd["L1:STRAIN"] = TimeSeries(np.random.randn(4096*10), t0=1000000000, sample_rate=4096)

# Source automatically creates pads for each channel
source = TimeSeriesSource(
    name="Multi_IFO",
    timeseries=tsd,
    buffer_duration=1.0,
)

# Source pad names: ("H1:STRAIN", "L1:STRAIN")
```

## TimeSeriesSink: Collect Pipeline Output

Collect pipeline output into a GWpy TimeSeries for further analysis or plotting.

### Basic Usage

```python
import numpy as np
from gwpy.timeseries import TimeSeries
from sgn.apps import Pipeline
from sgnligo.gwpy.sources import TimeSeriesSource
from sgnligo.gwpy.sinks import TimeSeriesSink

# Create sample data
ts = TimeSeries(np.random.randn(4096), t0=1000000000, sample_rate=4096, channel="TEST")

# Build pipeline
pipeline = Pipeline()
source = TimeSeriesSource(name="Source", timeseries=ts, buffer_duration=1.0)
pipeline.insert(source)

sink = TimeSeriesSink(
    name="Collector",
    sink_pad_names=("in",),
    channel="H1:PROCESSED",
    unit="strain",
    collect_all=True,  # Concatenate all buffers
)
pipeline.insert(sink, link_map={"Collector:snk:in": "Source:src:TEST"})

# Run and collect output
pipeline.run()
result = sink.get_result()           # Single TimeSeries
result_dict = sink.get_result_dict() # TimeSeriesDict

# Check collection status
print(f"Complete: {sink.is_complete}")
print(f"Samples: {sink.samples_collected}")
print(f"Duration: {sink.duration_collected} seconds")
```

### Reusing the Sink

```python
from sgnligo.gwpy.sinks import TimeSeriesSink

sink = TimeSeriesSink(name="Sink", sink_pad_names=("in",), channel="DATA")
# After first pipeline run, clear for reuse:
sink.clear()
```

## GWpyPlotSink: Streaming Plot Generation

Generate plots at fixed intervals during pipeline execution using the audio adapter pattern.

### Basic Usage

```python
import numpy as np
from gwpy.timeseries import TimeSeries
from sgn.apps import Pipeline
from sgnligo.gwpy.sources import TimeSeriesSource
from sgnligo.gwpy.sinks import GWpyPlotSink

# Create sample data
ts = TimeSeries(
    np.random.randn(4096 * 10),
    t0=1000000000,
    sample_rate=4096,
    channel="H1:STRAIN",
)

# Build pipeline
pipeline = Pipeline()
source = TimeSeriesSource(name="Source", timeseries=ts, buffer_duration=1.0)
pipeline.insert(source)

# Create plot sink - generates one plot per stride
plot_sink = GWpyPlotSink(
    name="Plotter",
    sink_pad_names=("in",),
    ifo="H1",
    description="STRAIN",
    plot_type="timeseries",
    output_dir="./plots",
    plot_stride=2.0,  # Generate plot every 2 seconds
    overlap_before=0.0,  # No overlap before
    overlap_after=0.0,   # No overlap after
    input_sample_rate=4096,
)
pipeline.insert(plot_sink, link_map={"Plotter:snk:in": "Source:src:H1:STRAIN"})

pipeline.run()
print(f"Generated {plot_sink.plots_generated} plots")
# Generates: H1-STRAIN-1000000000-2.png, H1-STRAIN-1000000002-2.png, etc.
```

### Plot Types

GWpyPlotSink supports three plot types:

**Time Series** (default):
```{.python notest}
plot_sink = GWpyPlotSink(
    name="Plotter",
    sink_pad_names=("in",),
    plot_type="timeseries",
    ...
)
```

**Spectrogram**:
```{.python notest}
plot_sink = GWpyPlotSink(
    name="Plotter",
    sink_pad_names=("in",),
    plot_type="spectrogram",
    fft_length=0.5,  # FFT length in seconds
    ...
)
```

**Q-Transform**:
```{.python notest}
plot_sink = GWpyPlotSink(
    name="Plotter",
    sink_pad_names=("in",),
    plot_type="qtransform",
    qrange=(4, 64),      # Q range
    frange=(20, 500),    # Frequency range in Hz
    ...
)
```

### Stride and Overlap

Control plot timing using `plot_stride`, `overlap_before`, and `overlap_after`:

- `plot_stride`: Time between consecutive plot starts
- `overlap_before`: Data to include before the stride (left overlap)
- `overlap_after`: Data to include after the stride (right overlap)
- Total plot duration = `overlap_before` + `plot_stride` + `overlap_after`

```{.python notest}
# Generate 4-second plots every 2 seconds
plot_sink = GWpyPlotSink(
    name="Plotter",
    sink_pad_names=("in",),
    plot_stride=2.0,      # New plot every 2 seconds
    overlap_before=1.0,   # Include 1 second before stride
    overlap_after=1.0,    # Include 1 second after stride
    # Total duration = 1 + 2 + 1 = 4 seconds per plot
    ...
)
# Produces: GPS 999999999 (4s), GPS 1000000001 (4s), GPS 1000000003 (4s), ...
```

### File Naming Convention

Files follow LIGO convention: `<IFO>-<DESCRIPTION>-<GPS_START>-<DURATION>.png`

- Dashes are used only as delimiters
- Any dashes in `description` are replaced with underscores
- GPS start and duration are integers

Examples:
- `H1-STRAIN-1000000000-2.png`
- `L1-STRAIN_FILTERED-1126259462-4.png`

### Custom Titles

Override the default title using a template:

```{.python notest}
plot_sink = GWpyPlotSink(
    name="Plotter",
    sink_pad_names=("in",),
    title_template="{ifo} Analysis - GPS {gps_start:.0f} ({duration:.1f}s)",
    ...
)
# Produces title: "H1 Analysis - GPS 1000000000 (2.0s)"
```

Available placeholders: `{ifo}`, `{gps_start}`, `{duration}`

### GWpyPlotSink Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ifo` | Interferometer name (H1, L1, V1) | "H1" |
| `description` | Description for filename | "STRAIN" |
| `plot_type` | "timeseries", "spectrogram", or "qtransform" | "timeseries" |
| `output_dir` | Directory for output files | "." |
| `plot_stride` | Time between plot starts (seconds) | 1.0 |
| `overlap_before` | Data before stride (seconds) | 0.0 |
| `overlap_after` | Data after stride (seconds). Duration = before + stride + after | 0.0 |
| `input_sample_rate` | Expected input sample rate (Hz) | 4096 |
| `fft_length` | FFT length for spectrogram (seconds) | 0.5 |
| `qrange` | Q range for Q-transform | (4, 64) |
| `frange` | Frequency range for Q-transform (Hz) | (20, 500) |
| `figsize` | Figure size (width, height) inches | (12, 4) |
| `dpi` | Resolution for saved plots | 150 |
| `title_template` | Custom title template | None |

## Complete Pipeline Examples

### Example 1: Bandpass and Q-Transform

A complete pipeline that filters data and computes a Q-transform:

```python
from sgn.apps import Pipeline
from sgnligo.gwpy.sources import TimeSeriesSource
from sgnligo.gwpy.transforms import GWpyFilter, GWpyQTransform
from sgnligo.gwpy.sinks import TimeSeriesSink
from gwpy.timeseries import TimeSeries
import numpy as np

# Generate test data with a burst signal
sample_rate = 4096
duration = 8
t = np.arange(0, duration, 1/sample_rate)
# Gaussian-modulated sinusoid (burst)
burst_time = 4.0
sigma = 0.1
burst = np.exp(-((t - burst_time)**2) / (2 * sigma**2)) * np.sin(2 * np.pi * 150 * t)
noise = 0.1 * np.random.randn(len(t))
data = burst + noise

ts = TimeSeries(data, t0=1000000000, sample_rate=sample_rate, channel="TEST")

# Build pipeline
pipeline = Pipeline()

# Source
source = TimeSeriesSource(name="Source", timeseries=ts, buffer_duration=1.0)
pipeline.insert(source)

# Bandpass 30-300 Hz
bandpass = GWpyFilter(
    name="Bandpass",
    sink_pad_names=("in",),
    source_pad_names=("out",),
    filter_type="bandpass",
    low_freq=30,
    high_freq=300,
)
pipeline.insert(bandpass, link_map={"Bandpass:snk:in": "Source:src:TEST"})

# Q-transform
qtrans = GWpyQTransform(
    name="QTrans",
    sink_pad_names=("in",),
    source_pad_names=("out",),
    qrange=(4, 64),
    frange=(20, 400),
    output_rate=64,
    input_sample_rate=sample_rate,
)
pipeline.insert(qtrans, link_map={"QTrans:snk:in": "Bandpass:src:out"})

# Collect Q-transform output
sink = TimeSeriesSink(name="Sink", sink_pad_names=("in",), channel="QTRANS")
pipeline.insert(sink, link_map={"Sink:snk:in": "QTrans:src:out"})

pipeline.run()
print("Bandpass + Q-transform pipeline complete!")
```

## Best Practices

### 1. Match Sample Rates

Ensure input sample rates match transform expectations:

```{.python notest}
# Correct: Specify input_sample_rate matching your data
qtrans = GWpyQTransform(
    input_sample_rate=4096,  # Must match source data rate
    ...
)
```

### 2. Use Power-of-2 Output Rates

For Q-transform and spectrogram, use power-of-2 output rates:

```{.python notest}
# Good: Power-of-2 rate
qtrans = GWpyQTransform(output_rate=64, ...)

# Bad: Arbitrary rate (will raise ValueError)
qtrans = GWpyQTransform(output_rate=100, ...)  # Error!
```

### 3. Handle Startup Transients

Transforms with accumulation (PSD, Whiten) have startup delays:

```python
# PSD needs fft_length of data before producing valid output
# Whiten needs 2*fft_length before producing output
# Plan for this in your analysis
```

### 4. Buffer Duration Selection

Choose buffer duration based on your use case:

```{.python notest}
# Smaller buffers: More responsive, higher overhead
source = TimeSeriesSource(buffer_duration=0.1, ...)

# Larger buffers: More efficient, higher latency
source = TimeSeriesSource(buffer_duration=1.0, ...)
```

### 5. Check for Gaps

When using TimeSeriesSink, gaps appear as NaN:

```{.python notest}
result = sink.get_result()
if np.any(np.isnan(result.value)):
    print("Warning: Result contains gaps")
    # Use np.nanmean, np.nanstd, etc. for statistics
```

## Summary

| Component | Purpose | Key Parameters |
|-----------|---------|----------------|
| `seriesbuffer_to_timeseries` | Convert buffer to GWpy | channel, unit |
| `timeseries_to_seriesbuffer` | Convert GWpy to buffer | - |
| `GWpyFilter` | Bandpass/lowpass/highpass/notch | filter_type, frequencies |
| `GWpyQTransform` | Q-transform | qrange, frange, output_rate |
| `GWpySpectrogram` | FFT spectrogram | spec_stride, fft_length |
| `TimeSeriesSource` | GWpy data to pipeline | timeseries, buffer_duration |
| `TimeSeriesSink` | Collect to GWpy | channel, collect_all |
| `GWpyPlotSink` | Streaming plot generation | plot_type, plot_stride, output_dir |

!!! note "PSD and Whitening"
    For power spectral density computation and whitening, use the native SGN-LIGO transforms in `sgnligo.transforms` and `sgnligo.psd` which provide optimized streaming implementations.

For more information:

- [GWpy Documentation](https://gwpy.github.io/docs/stable/)
- [DataSource Tutorial](datasource-tutorial.md) - For using simulated/real detector data
- [GWDataNoiseSource Tutorial](gwdata_noise_source.md) - For realistic noise simulation
