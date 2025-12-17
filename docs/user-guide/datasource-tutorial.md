# DataSource Tutorial: Comprehensive Guide to Data Sources

## Overview

The `datasource` module provides a unified interface for accessing gravitational wave data from various sources. Whether you're working with real-time detector data, archived frame files, or simulated data for testing, `datasource.py` handles the complexity of setting up the appropriate sources and connections.

This tutorial covers:

1. All supported data source types
2. State vector handling and data quality gating
3. Common configuration patterns
4. Integration with SGN pipelines
5. Best practices

## Supported Data Sources

The `datasource` module supports the following data source types:

| Source Type | Description | Use Case |
|------------|-------------|----------|
| `devshm` | Shared memory (real-time) | Live detector data streaming |
| `frames` | GWF frame files (offline) | Archived data analysis |
| `arrakis` | Arrakis data service | LIGO computing clusters |
| `gwdata-noise` | Simulated detector noise | Offline testing |
| `gwdata-noise-realtime` | Simulated noise (real-time) | Real-time pipeline testing |
| `white` | White noise | Basic testing |
| `sin` | Sinusoidal signal | Signal injection testing |
| `impulse` | Impulse signal | Transient testing |
| `white-realtime` | White noise (real-time) | Real-time basic testing |

## Quick Start Example

Here's a simple complete example using the datasource abstraction:

```python
#!/usr/bin/env python3
from sgn.apps import Pipeline
from sgnts.sinks import DumpSeriesSink
from sgnligo.sources import DataSourceInfo, datasource

# Configure data source
data_source_info = DataSourceInfo(
    data_source="gwdata-noise-realtime",
    channel_name=["H1=H1:FAKE-STRAIN"],
    gps_start_time=1400000000,
    gps_end_time=1400000010,
)

# Create pipeline
pipeline = Pipeline()

# Add datasource to pipeline
source_out_links, source_latency_links = datasource(
    pipeline=pipeline,
    info=data_source_info,
    verbose=True,
)

# Get the channel name from the data source info
channel_name = data_source_info.channel_dict["H1"]

# Add a sink to write the data to a file
sink = DumpSeriesSink(
    name="DataSink",
    sink_pad_names=[channel_name],
    fname="output_strain.txt",
    verbose=True,
)

# Add sink to pipeline and connect it
pipeline.insert(
    sink,
    link_map={
        f"DataSink:snk:{channel_name}": source_out_links["H1"],
    }
)

# Run pipeline
print("Running pipeline...")
pipeline.run()
print(f"Done! Data written to output_strain.txt")
```

This will generate 10 seconds of simulated strain data and save it to `output_strain.txt`.

## Real-Time Data: DevShm Source

The `devshm` source reads data from shared memory, typically used for real-time data processing.

!!! info "Simulating the /dev/shm Service for Testing"
    If you don't have access to LIGO's real /dev/shm data service, you can simulate it using `sgn-ligo-fake-frames` in real-time mode. This is useful for testing your pipeline locally.

    **Step 1: Create a state segments file**

    Create a file `state_segments.txt` with three columns (start_gps, end_gps, bitmask_value):

    ```
    1400000000 1400001000 3
    ```

    **Step 2: Start the frame generator in a separate terminal**

    ```bash
    # This will continuously generate frames to simulate real-time data
    sgn-ligo-fake-frames \
        --state-file state_segments.txt \
        --strain-channel H1:FAKE-STRAIN \
        --state-channel H1:FAKE-STATE_VECTOR \
        --output-path /tmp/fake_frames/H1-FAKE-{gps_start_time}-{duration}.gwf \
        --real-time \
        --verbose
    ```

    This process will:
    - Generate frame files continuously in `/tmp/fake_frames/`
    - Write both strain and state vector data
    - Simulate real-time operation by matching current GPS time
    - Clean up old files automatically (keeps last hour by default)

    **Step 3: Configure your pipeline to read from the frames**

    Instead of using `--data-source devshm`, use `--data-source frames` with a glob pattern:

    ```bash
    python your_pipeline.py \
        --data-source frames \
        --channel-name H1=H1:FAKE-STRAIN \
        --state-channel-name H1=H1:FAKE-STATE_VECTOR \
        --state-vector-on-bits H1=3 \
        --frame-cache "/tmp/fake_frames/*.gwf" \
        --verbose
    ```

    Or for a true real-time simulation using DevShmSource, you would need to set up the actual shared memory infrastructure, which is beyond the scope of this tutorial. For most testing purposes, the frame-based approach above is sufficient.

### Basic Usage

```bash
python your_pipeline.py \
    --data-source devshm \
    --channel-name H1=H1:GDS-CALIB_STRAIN \
    --shared-memory-dir H1=/dev/shm/kafka/H1_llhoft \
    --verbose
```

### With State Vector Gating

State vectors allow you to process data only when the detector is in a good state:

```bash
python your_pipeline.py \
    --data-source devshm \
    --channel-name H1=H1:GDS-CALIB_STRAIN \
    --state-channel-name H1=H1:GDS-CALIB_STATE_VECTOR \
    --state-vector-on-bits H1=3 \
    --shared-memory-dir H1=/dev/shm/kafka/H1_llhoft \
    --verbose
```

**How State Vector Gating Works:**

1. Reads both strain channel and state vector channel
2. Applies `BitMask` to check if specified bits are set (e.g., `3` = bits 0 and 1)
3. Uses `Gate` to only pass strain data when state vector condition is met
4. Outputs gaps during bad state periods

**Common State Vector Bits:**

- Bit 0: `HOFT_OK` - h(t) data is valid
- Bit 1: `OBS_INTENT` - Detector intends to be observing
- Bit 2: `SCIENCE_MODE` - Detector is in science mode

A value of `3` means both bits 0 and 1 must be set (HOFT_OK AND OBS_INTENT).

### Multiple Detectors

```bash
python your_pipeline.py \
    --data-source devshm \
    --channel-name H1=H1:GDS-CALIB_STRAIN \
    --channel-name L1=L1:GDS-CALIB_STRAIN \
    --state-channel-name H1=H1:GDS-CALIB_STATE_VECTOR \
    --state-channel-name L1=L1:GDS-CALIB_STATE_VECTOR \
    --state-vector-on-bits H1=3 \
    --state-vector-on-bits L1=3 \
    --shared-memory-dir H1=/dev/shm/kafka/H1_llhoft \
    --shared-memory-dir L1=/dev/shm/kafka/L1_llhoft \
    --verbose
```

## Offline Data: Frames Source

The `frames` source reads archived gravitational wave data from GWF frame files.

### Using Frame Cache Files

A frame cache file lists the paths to frame files:

```bash
# Create frame cache
find /path/to/frames -name "*.gwf" > frames.cache

# Use in pipeline
python your_pipeline.py \
    --data-source frames \
    --channel-name H1=H1:GDS-CALIB_STRAIN \
    --frame-cache frames.cache \
    --gps-start-time 1400000000 \
    --gps-end-time 1400000100 \
    --verbose
```

### Using Glob Patterns

```bash
python your_pipeline.py \
    --data-source frames \
    --channel-name H1=H1:GDS-CALIB_STRAIN \
    --frame-cache "/path/to/frames/H-*.gwf" \
    --gps-start-time 1400000000 \
    --gps-end-time 1400000100 \
    --verbose
```

### With Frame Segments

For analyzing specific time segments:

```bash
python your_pipeline.py \
    --data-source frames \
    --channel-name H1=H1:GDS-CALIB_STRAIN \
    --frame-cache frames.cache \
    --frame-segments-file segments.txt \
    --frame-segments-name "H1:DCH-ANALYSIS_READY:1" \
    --verbose
```

### With Injection Files

For testing with simulated signals injected into noise:

```bash
python your_pipeline.py \
    --data-source frames \
    --channel-name H1=H1:GDS-CALIB_STRAIN \
    --frame-cache frames.cache \
    --noiseless-inj-frame-cache injections.cache \
    --noiseless-inj-channel-name H1=H1:FAKE-INJECTION \
    --gps-start-time 1400000000 \
    --gps-end-time 1400000100 \
    --verbose
```

This will add the injection channel to the noise channel.

## Simulated Data: GWData Noise Sources

Perfect for testing and development without needing real detector data.

### GWData Noise (Offline)

Generates realistic detector noise for offline analysis:

```bash
python your_pipeline.py \
    --data-source gwdata-noise \
    --channel-name H1=H1:FAKE-STRAIN \
    --gps-start-time 1400000000 \
    --gps-end-time 1400000010 \
    --verbose
```

### GWData Noise Realtime

Simulates a real-time data stream:

```bash
python your_pipeline.py \
    --data-source gwdata-noise-realtime \
    --channel-name H1=H1:FAKE-STRAIN \
    --gps-start-time 1400000000 \
    --gps-end-time 1400000100 \
    --verbose
```

This generates data in real-time: 1 second of data per 1 second of wall time.

### With Simulated State Vectors

You can add state vector channels to `gwdata-noise` sources:

```bash
python your_pipeline.py \
    --data-source gwdata-noise-realtime \
    --channel-name H1=H1:FAKE-STRAIN \
    --state-channel-name H1=H1:FAKE-STATE_VECTOR \
    --state-vector-on-bits H1=3 \
    --gps-start-time 1400000000 \
    --gps-end-time 1400000100 \
    --verbose
```

By default, this creates a constant state value of 3 (HOFT_OK + OBS_INTENT).

### Custom State Patterns

Define your own state vector patterns with a segments file:

Create `state_segments.txt`:
```
# Format: start_gps end_gps bitmask_value
1400000000 1400000040 3    # Good state
1400000040 1400000050 0    # Bad state
1400000050 1400000100 3    # Good state resumed
```

Use it in your pipeline:

```bash
python your_pipeline.py \
    --data-source gwdata-noise-realtime \
    --channel-name H1=H1:FAKE-STRAIN \
    --state-channel-name H1=H1:FAKE-STATE_VECTOR \
    --state-vector-on-bits H1=3 \
    --state-segments-file state_segments.txt \
    --gps-start-time 1400000000 \
    --gps-end-time 1400000100 \
    --verbose
```

**What happens during bad states:**
- The pipeline generates gaps in the output during bad state periods (seconds 40-50 in this example)
- Downstream elements receive gap buffers
- This simulates real detector behavior during maintenance, glitches, etc.

### State Sample Rate

Control the sample rate of state vector channels (default 16 Hz):

```bash
python your_pipeline.py \
    --data-source gwdata-noise-realtime \
    --channel-name H1=H1:FAKE-STRAIN \
    --state-channel-name H1=H1:FAKE-STATE_VECTOR \
    --state-sample-rate 32 \
    --state-segments-file state_segments.txt \
    --verbose
```

## Simple Test Sources

For basic testing and debugging.

### White Noise

```bash
python your_pipeline.py \
    --data-source white \
    --channel-name H1=H1:TEST-STRAIN \
    --input-sample-rate 4096 \
    --gps-start-time 1400000000 \
    --gps-end-time 1400000010 \
    --verbose
```

### White Noise (Real-time)

```bash
python your_pipeline.py \
    --data-source white-realtime \
    --channel-name H1=H1:TEST-STRAIN \
    --input-sample-rate 4096 \
    --gps-start-time 1400000000 \
    --gps-end-time 1400000100 \
    --verbose
```

### Sinusoidal Signal

Useful for testing frequency-domain processing:

```bash
python your_pipeline.py \
    --data-source sin \
    --channel-name H1=H1:TEST-SINE \
    --input-sample-rate 4096 \
    --gps-start-time 1400000000 \
    --gps-end-time 1400000010 \
    --verbose
```

### Impulse Signal

Useful for testing transient detection:

```bash
python your_pipeline.py \
    --data-source impulse \
    --channel-name H1=H1:TEST-IMPULSE \
    --input-sample-rate 4096 \
    --impulse-position 100 \
    --gps-start-time 1400000000 \
    --gps-end-time 1400000010 \
    --verbose
```

Set `--impulse-position -1` for random position.

## Using DataSourceInfo in Python

Instead of command-line arguments, you can configure data sources directly in Python:

### Example 1: Real-time Simulated Data

```
from sgn.apps import Pipeline
from sgnligo.sources import DataSourceInfo, datasource

# Configure simulated real-time source with state vector gating
data_source_info = DataSourceInfo(
    data_source="gwdata-noise-realtime",
    channel_name=["H1=H1:FAKE-STRAIN"],
    state_channel_name=["H1=H1:FAKE-STATE_VECTOR"],
    state_vector_on_bits=["H1=3"],
    gps_start_time=1400000000,
    gps_end_time=1400000100,
    state_segments_file="state_segments.txt",
    state_sample_rate=16,
)

pipeline = Pipeline()
source_out_links, source_latency_links = datasource(
    pipeline=pipeline,
    info=data_source_info,
    verbose=True,
)

# source_out_links["H1"] contains the pad reference for gated H1 strain
# Add your processing elements here...
# For example, to add a sink:
# channel_name = data_source_info.channel_dict["H1"]
# sink = MySink(sink_pad_names=[channel_name], ...)
# pipeline.insert(sink, link_map={f"MySink:snk:{channel_name}": source_out_links["H1"]})

pipeline.run()
```

### Example 2: Offline Frame Analysis

```
from sgnligo.sources import DataSourceInfo, datasource
from sgn.apps import Pipeline

data_source_info = DataSourceInfo(
    data_source="frames",
    channel_name=["H1=H1:GDS-CALIB_STRAIN", "L1=L1:GDS-CALIB_STRAIN"],
    frame_cache="frames.cache",
    gps_start_time=1400000000,
    gps_end_time=1400001000,
)

pipeline = Pipeline()
source_out_links, source_latency_links = datasource(
    pipeline=pipeline,
    info=data_source_info,
    verbose=True,
)

# source_out_links["H1"] and source_out_links["L1"] available
# Get channel names: data_source_info.channel_dict["H1"], data_source_info.channel_dict["L1"]
# Add multi-detector analysis elements...

pipeline.run()
```

### Example 3: From Command-Line Options

Most sgn-ligo programs use this pattern:

```python
# <!-- skip-test -->
from argparse import ArgumentParser
from sgn.apps import Pipeline
from sgnligo.sources import DataSourceInfo, datasource

def parse_command_line():
    parser = ArgumentParser(description="My analysis pipeline")

    # Add datasource options
    DataSourceInfo.append_options(parser)

    # Add your custom options
    parser.add_argument("--custom-option", help="My custom option")

    return parser.parse_args()

def main():
    options = parse_command_line()

    # Create DataSourceInfo from command-line options
    data_source_info = DataSourceInfo.from_options(options)

    # Use in pipeline...
    pipeline = Pipeline()
    source_out_links, source_latency_links = datasource(
        pipeline=pipeline,
        info=data_source_info,
        verbose=options.verbose,
    )

    # ... rest of pipeline setup

    pipeline.run()

if __name__ == "__main__":
    main()
```

## Understanding Source Output Links

The `datasource()` function returns two values:

```
source_out_links, source_latency_links = datasource(pipeline, info, verbose=True)
```

### `source_out_links`

A dictionary mapping IFO names to their output pad references:

```python
# Example: {"H1": "GWDataNoiseSource:src:H1:FAKE-STRAIN", "L1": "GWDataNoiseSource:src:L1:FAKE-STRAIN"}
```

Use these to connect downstream elements:

```
pipeline.insert(
    link_map={
        "MyElement:snk:H1_input": source_out_links["H1"],
        "MyElement:snk:L1_input": source_out_links["L1"],
    }
)
```

### `source_latency_links`

A dictionary mapping IFO names to latency measurement pads (if `source_latency=True`), otherwise `None`.

### Getting Channel Names

Channel names are available from the `DataSourceInfo` object:

```
# Access the channel dictionary
channel_name = data_source_info.channel_dict["H1"]
# Example: "H1:GDS-CALIB_STRAIN"
```

Use these when creating sinks or elements that need channel names.

### With State Vector Gating

When using state vector gating, the output links point to the **gated** output:

```
GWDataNoiseSource → SegmentSource (state) → BitMask → Gate → [OUTPUT]
                                                              ↑
                                           source_out_links["H1"] points here
```

Data only flows when the state condition is met.

## Advanced: Arrakis Source

For use on LIGO computing clusters:

```bash
python your_pipeline.py \
    --data-source arrakis \
    --channel-name H1=H1:GDS-CALIB_STRAIN \
    --gps-start-time 1400000000 \
    --gps-end-time 1400000100 \
    --verbose
```

Arrakis automatically handles frame discovery and caching.

## Common Patterns and Best Practices

### Pattern 1: Testing with Simulated Data

Start with `gwdata-noise-realtime` to test your pipeline:

```bash
# Test phase
python my_pipeline.py \
    --data-source gwdata-noise-realtime \
    --channel-name H1=H1:FAKE-STRAIN \
    --gps-start-time 1400000000 \
    --gps-end-time 1400000100

# Production phase (just change data source!)
python my_pipeline.py \
    --data-source devshm \
    --channel-name H1=H1:GDS-CALIB_STRAIN \
    --shared-memory-dir H1=/dev/shm/kafka/H1_llhoft
```

### Pattern 2: Replay Analysis

Use frames to replay past events:

```bash
# Find frames covering the event
find /path/to/frames -name "*1400000000*.gwf" > event_frames.cache

# Run analysis
python my_pipeline.py \
    --data-source frames \
    --channel-name H1=H1:GDS-CALIB_STRAIN \
    --frame-cache event_frames.cache \
    --gps-start-time 1400000000 \
    --gps-end-time 1400000100
```

### Pattern 3: Multi-Detector Coincidence

```python
# <!-- skip-test -->
from sgn.apps import Pipeline
from sgnligo.sources import DataSourceInfo, datasource

data_source_info = DataSourceInfo(
    data_source="gwdata-noise-realtime",
    channel_name=[
        "H1=H1:FAKE-STRAIN",
        "L1=L1:FAKE-STRAIN",
        "V1=V1:FAKE-STRAIN",
    ],
    gps_start_time=1400000000,
    gps_end_time=1400000100,
)

pipeline = Pipeline()
source_out_links, source_latency_links = datasource(pipeline, data_source_info, verbose=True)

# source_out_links now has {"H1": ..., "L1": ..., "V1": ...}
# Process all three detector streams...
```

### Pattern 4: State Vector Only (e.g., for gwistat)

```bash
sgn-ligo-gwistat \
    --data-source gwdata-noise-realtime \
    --channel-name H1:FAKE-STATE_VECTOR \
    --mapping-file bitmask_mapping.json \
    --gps-start-time 1400000000 \
    --gps-end-time 1400000010
```

Note: Some tools like `gwistat` handle state vectors specially and don't use the gating mechanism.

## Troubleshooting

### "Channel not found in frames"

Make sure the channel exists in your frame files:

```bash
# Check frame contents
FrChannels /path/to/frame.gwf
```

### "State vector mismatch"

Ensure you have matching IFOs for strain and state channels:

```bash
# ✅ Correct
--channel-name H1=H1:GDS-CALIB_STRAIN \
--state-channel-name H1=H1:GDS-CALIB_STATE_VECTOR

# ❌ Wrong (mismatched IFOs)
--channel-name H1=H1:GDS-CALIB_STRAIN \
--state-channel-name L1=L1:GDS-CALIB_STATE_VECTOR
```

### "GPS times required"

Some sources require explicit GPS times:

```bash
# ❌ Wrong (offline source needs times)
--data-source gwdata-noise \
--channel-name H1=H1:FAKE-STRAIN

# ✅ Correct
--data-source gwdata-noise \
--channel-name H1=H1:FAKE-STRAIN \
--gps-start-time 1400000000 \
--gps-end-time 1400000010
```

## Summary

The `datasource` abstraction provides a unified interface for:

- **Real-time data** (devshm)
- **Archived data** (frames)
- **Simulated data** (gwdata-noise, white, sin, impulse)
- **Data quality gating** (state vectors with BitMask + Gate)

Key benefits:

1. **Consistent interface** across all data sources
2. **Easy testing** - switch from simulated to real data by changing one parameter
3. **Built-in state vector handling** for data quality
4. **Multi-detector support** out of the box

For specific use cases, see:

- [GWIStat Tutorial](gwistat-tutorial.md) - State vector interpretation
- [LL-DQ Tutorial](ll-dq-tutorial.md) - Horizon distance calculation
- [GWDataNoiseSource Tutorial](gwdata_noise_source.md) - Detailed noise simulation

## Reference

### All DataSourceInfo Parameters

```
DataSourceInfo(
    data_source: str,                           # Required: source type
    channel_name: list[str],                    # Required: IFO=CHANNEL format
    gps_start_time: float = None,               # Start GPS time
    gps_end_time: float = None,                 # End GPS time
    frame_cache: str = None,                    # Frame cache file/glob
    frame_segments_file: str = None,            # Segments file for frames
    frame_segments_name: str = None,            # Segment name in file
    noiseless_inj_frame_cache: str = None,      # Injection frames
    noiseless_inj_channel_name: list[str] = None,  # Injection channels
    state_channel_name: list[str] = None,       # State vector channels
    state_vector_on_bits: list[int] = None,     # Required bits for gating
    shared_memory_dir: list[str] = None,        # DevShm directories
    discont_wait_time: float = 60,              # Discontinuity timeout
    source_queue_timeout: float = 1,            # Queue timeout
    input_sample_rate: int = None,              # Sample rate for fake sources
    impulse_position: int = -1,                 # Impulse position (-1=random)
    real_time: bool = False,                    # Enable real-time mode
    state_segments_file: str = None,            # State pattern file
    state_sample_rate: int = 16,                # State vector sample rate
)
```
