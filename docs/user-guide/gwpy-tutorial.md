# GWpy Integration Tutorial

## Overview

The `sgnligo.gwpy` module provides seamless integration between SGN streaming pipelines and [GWpy](https://gwpy.github.io/docs/stable/), the standard gravitational wave data analysis library. This enables you to:

- Use GWpy's powerful analysis methods within streaming pipelines
- Convert data between SGN's `SeriesBuffer` and GWpy's `TimeSeries`
- Access GWOSC open data directly in pipelines
- Produce output ready for further GWpy-based analysis

This tutorial covers:

1. Data conversion utilities
2. Filtering with GWpyFilter
3. Time-frequency analysis with GWpyQTransform and GWpySpectrogram
4. Data sources: TimeSeriesSource and GWOSCSource
5. Output collection with TimeSeriesSink
6. Complete pipeline examples
7. **Visualizing Gravitational Waves** - Plot GW150914 with filtering and Q-transform

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
from sgnligo.gwpy.sources import TimeSeriesSource, GWOSCSource
from sgnligo.gwpy.sinks import TimeSeriesSink
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

```python
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

## GWOSCSource: GWOSC Open Data

Fetch data directly from the Gravitational Wave Open Science Center.

### Fetching Event Data

```python
from sgnligo.gwpy.sources import GWOSCSource
from sgn.apps import Pipeline

# Fetch GW150914 data
source = GWOSCSource(
    name="GWOSC_H1",
    source_pad_names=("strain",),
    detector="H1",
    start_time=1126259462,
    duration=32,
    target_sample_rate=4096,
    chunk_size=16,  # Fetch in 16-second chunks
)
```

!!! note "Internet Required"
    GWOSCSource requires an internet connection to fetch data from GWOSC. Data is cached locally after first fetch.

### Available Detectors

| Detector | Description |
|----------|-------------|
| `H1` | LIGO Hanford |
| `L1` | LIGO Livingston |
| `V1` | Virgo |
| `G1` | GEO600 |
| `K1` | KAGRA |

### Complete GWOSC Example

!!! note "Network Required"
    This example fetches data from GWOSC and requires an internet connection.

```{.python notest}
from sgn.apps import Pipeline
from sgnligo.gwpy.sources import GWOSCSource
from sgnligo.gwpy.transforms import GWpyFilter
from sgnligo.gwpy.sinks import TimeSeriesSink

pipeline = Pipeline()

# Fetch 32 seconds around GW150914
source = GWOSCSource(
    name="GWOSC",
    source_pad_names=("strain",),
    detector="H1",
    start_time=1126259446,  # 16s before merger
    duration=32,
    target_sample_rate=4096,
)
pipeline.insert(source)

# Bandpass filter 35-350 Hz
filt = GWpyFilter(
    name="BandPass",
    sink_pad_names=("in",),
    source_pad_names=("out",),
    filter_type="bandpass",
    low_freq=35,
    high_freq=350,
)
pipeline.insert(filt, link_map={"BandPass:snk:in": "GWOSC:src:strain"})

# Collect output
sink = TimeSeriesSink(name="Sink", sink_pad_names=("in",), channel="H1:FILTERED")
pipeline.insert(sink, link_map={"Sink:snk:in": "BandPass:src:out"})

pipeline.run()

# Get result for further analysis
result = sink.get_result()
print(f"Filtered GW150914 data: {result.duration} seconds")
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

## Visualizing Gravitational Waves: GW150914 Example

This section demonstrates a complete workflow for visualizing gravitational wave events using a pure SGN streaming pipeline. We'll use `GWOSCSource` to fetch data, process it through SGN transforms (whitening, filtering, Q-transform), collect output with `TimeSeriesSink`, and visualize using `sgnts.plotting` utilities.

### Plotting Dependencies

```bash
pip install matplotlib
pip install sgn-ts[plot]  # SGN-TS plotting utilities
```

### Complete GW150914 Pipeline (Pure SGN)

This script demonstrates a fully native SGN approach using only SGN pipeline components:

```{.python notest}
#!/usr/bin/env python3
"""Visualize GW150914 using a pure SGN streaming pipeline.

This script uses ONLY SGN pipeline components:
- GWOSCSource: Fetch data from GWOSC
- GWpyFilter: Bandpass filter
- GWpyQTransform: Compute Q-transform (streaming)
- TimeSeriesSink: Collect output
- sgnts.plotting: Visualize results

Note: For whitening, use sgnligo.transforms.Whiten which provides
optimized streaming whitening with proper edge handling.
"""

import matplotlib.pyplot as plt
import numpy as np

from sgn.apps import Pipeline
from sgnligo.gwpy.sources import GWOSCSource
from sgnligo.gwpy.transforms import GWpyFilter
from sgnligo.gwpy.sinks import TimeSeriesSink
from sgnligo.gwpy.converters import timeseries_to_seriesbuffer
from sgnts.plotting import plot_buffer

# GW150914 event parameters
EVENT_GPS = 1126259462.4
START_GPS = 1126259446  # 16s before merger
DURATION = 32           # 32 seconds total

# ============================================================
# Build Pure SGN Pipeline
# ============================================================
print("Building SGN pipeline...")
pipeline = Pipeline()

# Source: GWOSCSource fetches directly from GWOSC
source = GWOSCSource(
    name="GWOSC",
    source_pad_names=("strain",),
    detector="H1",
    start_time=START_GPS,
    duration=DURATION,
    target_sample_rate=4096,
    chunk_size=8,  # Fetch in 8-second chunks for streaming
)
pipeline.insert(source)

# Bandpass filter 30-350 Hz
bandpass = GWpyFilter(
    name="Bandpass",
    sink_pad_names=("in",),
    source_pad_names=("out",),
    filter_type="bandpass",
    low_freq=30,
    high_freq=350,
)
pipeline.insert(bandpass, link_map={"Bandpass:snk:in": "GWOSC:src:strain"})

# Collect filtered strain
strain_sink = TimeSeriesSink(
    name="StrainSink",
    sink_pad_names=("in",),
    channel="H1:FILTERED",
)
pipeline.insert(strain_sink, link_map={"StrainSink:snk:in": "Bandpass:src:out"})

# Run the pipeline
print("Running pipeline (fetching from GWOSC and processing)...")
pipeline.run()

# ============================================================
# Get Results as SGN Buffers
# ============================================================
print("Extracting results...")

# Get the collected TimeSeries and convert to buffer for plotting
filtered_ts = strain_sink.get_result()
filtered_buf = timeseries_to_seriesbuffer(filtered_ts)

print(f"  Collected {filtered_buf.samples} samples")
print(f"  Duration: {filtered_buf.samples / filtered_buf.sample_rate:.1f}s")

# ============================================================
# Visualize Using SGN-TS Plotting
# ============================================================
print("Creating visualization...")

fig, ax = plt.subplots(figsize=(12, 4))

# Plot filtered strain using SGN's plot_buffer
plot_buffer(filtered_buf, ax=ax, time_unit='gps', color='#ee0000', linewidth=0.8)
ax.set_xlim(EVENT_GPS - 0.5, EVENT_GPS + 0.2)
ax.set_ylabel('Filtered Strain')
ax.set_xlabel('GPS Time (s)')
ax.set_title('H1 GW150914 - Bandpass Filtered (Pure SGN Pipeline)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gw150914_sgn_filtered.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot to gw150914_sgn_filtered.png")
```

### Full Streaming Pipeline with Q-Transform

For a complete analysis including the Q-transform computed within the streaming pipeline:

```{.python notest}
#!/usr/bin/env python3
"""Complete GW150914 analysis with streaming Q-transform."""

import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field

from sgn.apps import Pipeline
from sgnligo.gwpy.sources import GWOSCSource
from sgnligo.gwpy.transforms import GWpyFilter, GWpyQTransform
from sgnligo.gwpy.sinks import TimeSeriesSink
from sgnligo.gwpy.converters import timeseries_to_seriesbuffer
from sgnts.base import TSSink
from sgnts.plotting import plot_buffer

EVENT_GPS = 1126259462.4

# ============================================================
# Custom sink to capture Q-transform metadata
# ============================================================
@dataclass
class QTransformSink(TSSink):
    """Sink that captures Q-transform output and metadata."""
    qtransforms: list = field(default_factory=list, init=False, repr=False)
    buffers: list = field(default_factory=list, init=False, repr=False)

    def internal(self):
        super().internal()
        for pad in self.sink_pads:
            frame = self.preparedframes.get(pad)
            if frame is None:
                continue
            if "qtransform" in frame.metadata and frame.metadata["qtransform"] is not None:
                self.qtransforms.append(frame.metadata["qtransform"])
            for buf in frame.buffers:
                if not buf.is_gap:
                    self.buffers.append(buf)
            if frame.EOS:
                self.mark_eos(pad)

# ============================================================
# Build Pipeline with Q-Transform
# ============================================================
print("Building pipeline...")
pipeline = Pipeline()

# GWOSCSource for data fetching
source = GWOSCSource(
    name="GWOSC",
    source_pad_names=("strain",),
    detector="H1",
    start_time=1126259446,
    duration=32,
    target_sample_rate=4096,
    chunk_size=8,
)
pipeline.insert(source)

# Bandpass filter (before Q-transform)
bandpass = GWpyFilter(
    name="Bandpass",
    sink_pad_names=("in",),
    source_pad_names=("out",),
    filter_type="bandpass",
    low_freq=20,
    high_freq=500,
)
pipeline.insert(bandpass, link_map={"Bandpass:snk:in": "GWOSC:src:strain"})

# Collect filtered strain for time-domain plot
strain_sink = TimeSeriesSink(
    name="StrainSink",
    sink_pad_names=("in",),
    channel="H1:FILTERED",
)
pipeline.insert(strain_sink, link_map={"StrainSink:snk:in": "Bandpass:src:out"})

# Q-Transform in the streaming pipeline
qtrans = GWpyQTransform(
    name="QTrans",
    sink_pad_names=("in",),
    source_pad_names=("out",),
    qrange=(4, 64),
    frange=(20, 500),
    output_rate=64,
    output_stride=1.0,
    input_sample_rate=4096,
)
pipeline.insert(qtrans, link_map={"QTrans:snk:in": "Bandpass:src:out"})

# Custom sink to capture Q-transform metadata
qsink = QTransformSink(name="QSink", sink_pad_names=("in",))
pipeline.insert(qsink, link_map={"QSink:snk:in": "QTrans:src:out"})

# Run pipeline
print("Running pipeline...")
pipeline.run()

# ============================================================
# Visualize Results
# ============================================================
print(f"Captured {len(qsink.qtransforms)} Q-transform segments")

# Get filtered strain as buffer
filtered_ts = strain_sink.get_result()
filtered_buf = timeseries_to_seriesbuffer(filtered_ts)

# Create figure
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Panel 1: Filtered strain using plot_buffer
ax1 = axes[0]
plot_buffer(filtered_buf, ax=ax1, time_unit='gps', color='#ee0000', linewidth=0.8)
ax1.set_xlim(EVENT_GPS - 0.5, EVENT_GPS + 0.2)
ax1.set_ylabel('Filtered Strain')
ax1.set_title('H1 Bandpass Filtered (20-500 Hz) - SGN Pipeline')
ax1.grid(True, alpha=0.3)

# Panel 2: Q-transform from pipeline metadata
ax2 = axes[1]
if qsink.qtransforms:
    # Find Q-transform segment containing the event
    best_q = None
    for q in qsink.qtransforms:
        t_start = q.times.value[0]
        t_end = q.times.value[-1]
        if t_start <= EVENT_GPS <= t_end:
            best_q = q
            break
    if best_q is None:
        best_q = qsink.qtransforms[len(qsink.qtransforms) // 2]

    pcm = ax2.pcolormesh(
        best_q.times.value,
        best_q.frequencies.value,
        best_q.value.T,
        shading='auto',
        vmin=0,
        vmax=25,
        cmap='viridis',
    )
    ax2.set_yscale('log')
    ax2.set_ylim(20, 500)
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('GPS Time (s)')
    ax2.set_title('H1 Q-Transform (Streaming Pipeline)')
    fig.colorbar(pcm, ax=ax2, label='Normalized Energy')

plt.tight_layout()
plt.savefig('gw150914_sgn_full.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Multi-Detector Comparison (Pure SGN)

Process both LIGO detectors using separate SGN pipelines:

```{.python notest}
#!/usr/bin/env python3
"""Compare GW150914 in H1 and L1 using pure SGN pipelines."""

import matplotlib.pyplot as plt
from sgn.apps import Pipeline
from sgnligo.gwpy.sources import GWOSCSource
from sgnligo.gwpy.transforms import GWpyFilter
from sgnligo.gwpy.sinks import TimeSeriesSink
from sgnligo.gwpy.converters import timeseries_to_seriesbuffer
from sgnts.plotting import plot_buffer

EVENT_GPS = 1126259462.4

def process_detector(detector: str) -> "SeriesBuffer":
    """Run pure SGN pipeline for a single detector."""
    pipeline = Pipeline()

    # GWOSCSource fetches from GWOSC
    source = GWOSCSource(
        name="GWOSC",
        source_pad_names=("strain",),
        detector=detector,
        start_time=1126259446,
        duration=32,
        target_sample_rate=4096,
        chunk_size=8,
    )
    pipeline.insert(source)

    # Bandpass filter
    bandpass = GWpyFilter(
        name="Bandpass",
        sink_pad_names=("in",),
        source_pad_names=("out",),
        filter_type="bandpass",
        low_freq=30,
        high_freq=350,
    )
    pipeline.insert(bandpass, link_map={"Bandpass:snk:in": "GWOSC:src:strain"})

    # Collect output
    sink = TimeSeriesSink(
        name="Sink",
        sink_pad_names=("in",),
        channel=f"{detector}:FILTERED",
    )
    pipeline.insert(sink, link_map={"Sink:snk:in": "Bandpass:src:out"})

    pipeline.run()

    # Convert to SeriesBuffer for SGN plotting
    return timeseries_to_seriesbuffer(sink.get_result())

# Process both detectors
print("Processing H1...")
h1_buf = process_detector("H1")
print("Processing L1...")
l1_buf = process_detector("L1")

# ============================================================
# Plot comparison using SGN plot_buffer
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# H1
plot_buffer(h1_buf, ax=axes[0], time_unit='gps', color='#ee0000', linewidth=0.8)
axes[0].set_xlim(EVENT_GPS - 0.4, EVENT_GPS + 0.1)
axes[0].set_ylabel('H1 Strain')
axes[0].set_title('H1 Bandpass Filtered (Pure SGN Pipeline)')
axes[0].grid(True, alpha=0.3)

# L1
plot_buffer(l1_buf, ax=axes[1], time_unit='gps', color='#4ba6ff', linewidth=0.8)
axes[1].set_xlim(EVENT_GPS - 0.4, EVENT_GPS + 0.1)
axes[1].set_ylabel('L1 Strain')
axes[1].set_xlabel('GPS Time (s)')
axes[1].set_title('L1 Bandpass Filtered (Pure SGN Pipeline)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gw150914_h1_l1_sgn.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved comparison plot")
```

### SGN-TS Plotting Reference

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `plot_buffer(buf, ax=None)` | Plot a SeriesBuffer | `time_unit`, `label`, `color`, `show_gaps` |
| `plot_frame(frame, ax=None)` | Plot a TSFrame (multiple buffers) | Same as above, plus `multichannel` |
| `plot_frames(frames, ax=None)` | Plot multiple frames overlaid | `labels` |

**Time unit options:**
- `'gps'` - Absolute GPS time in seconds (default)
- `'s'` - Time in seconds
- `'ms'` - Time in milliseconds
- `'ns'` - Time in nanoseconds

### Output Examples

Running the GW150914 scripts produces plots showing:

1. **Whitened strain (plot_buffer)**: Noise-normalized signal from the pure SGN streaming pipeline where the chirp is clearly visible
2. **Q-transform (from pipeline metadata)**: Time-frequency representation showing the characteristic "chirp" pattern computed within the streaming pipeline

The chirp pattern in the Q-transform is the signature of a binary black hole merger - the frequency sweeps from ~35 Hz to ~150 Hz in about 0.2 seconds, corresponding to the final orbits before merger.

## Summary

| Component | Purpose | Key Parameters |
|-----------|---------|----------------|
| `seriesbuffer_to_timeseries` | Convert buffer to GWpy | channel, unit |
| `timeseries_to_seriesbuffer` | Convert GWpy to buffer | - |
| `GWpyFilter` | Bandpass/lowpass/highpass/notch | filter_type, frequencies |
| `GWpyQTransform` | Q-transform | qrange, frange, output_rate |
| `GWpySpectrogram` | FFT spectrogram | spec_stride, fft_length |
| `TimeSeriesSource` | GWpy data to pipeline | timeseries, buffer_duration |
| `GWOSCSource` | GWOSC open data | detector, start_time, duration |
| `TimeSeriesSink` | Collect to GWpy | channel, collect_all |

!!! note "PSD and Whitening"
    For power spectral density computation and whitening, use the native SGN-LIGO transforms in `sgnligo.transforms` and `sgnligo.psd` which provide optimized streaming implementations.

For more information:

- [GWpy Documentation](https://gwpy.github.io/docs/stable/)
- [DataSource Tutorial](datasource-tutorial.md) - For using simulated/real detector data
- [GWDataNoiseSource Tutorial](gwdata_noise_source.md) - For realistic noise simulation
