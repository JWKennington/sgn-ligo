# DataSource V2: Composable Data Sources

## Overview

The `datasource_v2` package provides a modern, composable API for creating gravitational wave data sources. It uses dataclasses for clean, type-safe configuration with automatic validation.

| Feature | Original `datasource` | New `datasource_v2` |
|---------|----------------------|---------------------|
| API Style | Mutates pipeline, returns link strings | Returns composed elements |
| Extensibility | Hardcoded source types | Registry pattern with dataclass sources |
| Configuration | Mixed validation/construction | Dataclasses with `_validate()` and `_build()` |
| Composition | Manual element wiring | Uses TSCompose |
| Testing | Difficult to unit test | Easy to test in isolation |

!!! note "Migration Status"
    `datasource_v2` is designed to work alongside the original `datasource` module during migration. Both can coexist in the same codebase.

## Quick Start

### Basic Usage with DataSource Dispatcher

```python
from sgn.apps import Pipeline
from sgn.sinks import NullSink
from sgnligo.sources.datasource_v2 import DataSource

# Create a source using the DataSource dispatcher
source = DataSource(
    data_source="white",
    name="my_source",
    ifos=["H1", "L1"],
    sample_rate=4096,
    t0=1000,
    end=1010,
)

# Build pipeline using connect()
pipeline = Pipeline()
sink = NullSink(name="sink", sink_pad_names=list(source.srcs.keys()))
pipeline.connect(source.element, sink)
pipeline.run()
```

### Direct Source Class Usage

For more control, use source classes directly:

```python
from sgnligo.sources.datasource_v2.sources import WhiteComposedSource, GWDataNoiseComposedSource

# White noise source
source = WhiteComposedSource(
    name="noise",
    ifos=["H1", "L1"],
    sample_rate=4096,
    t0=1000,
    end=1010,
)

# Colored LIGO noise source
source = GWDataNoiseComposedSource(
    name="colored_noise",
    ifos=["H1"],
    t0=1000000000,
    end=1000000100,
)
```

### With CLI Arguments

```python notest
import sys
from sgnligo.sources.datasource_v2 import (
    check_composed_help_options,
    DataSource,
)

# Handle --list-sources and --help-source before full parsing
if check_composed_help_options():
    sys.exit(0)

parser = DataSource.create_cli_parser()
# Add your own options here
args = parser.parse_args()

source = DataSource.from_cli(args, name="my_source")
```

Run with:

```bash
python my_pipeline.py \
    --data-source gwdata-noise \
    --channel-name H1=STRAIN \
    --channel-name L1=STRAIN \
    --gps-start-time 1000000000 \
    --gps-end-time 1000000010
```

## Supported Source Types

All source types from the original module are supported:

| Source Type | Description | Source Class | Required Options |
|-------------|-------------|--------------|------------------|
| `white` | White Gaussian noise | `WhiteSource` | `sample_rate`, `t0`, `end` |
| `sin` | Sinusoidal test signal | `SinSource` | `sample_rate`, `t0`, `end` |
| `impulse` | Single impulse signal | `ImpulseSource` | `sample_rate`, `t0`, `end` |
| `white-realtime` | Real-time white noise | `WhiteRealtimeSource` | `sample_rate` |
| `sin-realtime` | Real-time sinusoidal signal | `SinRealtimeSource` | `sample_rate` |
| `impulse-realtime` | Real-time impulse signal | `ImpulseRealtimeSource` | `sample_rate` |
| `gwdata-noise` | Colored detector noise | `GWDataNoiseComposedSource` | `t0`, `end` |
| `gwdata-noise-realtime` | Real-time colored noise | `GWDataNoiseRealtimeComposedSource` | (none) |
| `frames` | GWF frame files | `FramesComposedSource` | `frame_cache`, `t0`, `end` |
| `devshm` | Shared memory (live) | `DevShmComposedSource` | `shared_memory_dict`, `channel_dict` |
| `arrakis` | Kafka streaming | `ArrakisComposedSource` | `channel_dict` |
| `injected-noise` | Colored noise with injections | `InjectedNoiseSource` | `t0`, `duration` or `end` |

## API Reference

### DataSource Dispatcher

The `DataSource` class dispatches to the appropriate source class based on `data_source`:

```python
from sgnligo.sources.datasource_v2 import DataSource

# Colored noise source
source = DataSource(
    data_source="gwdata-noise",
    name="noise",
    ifos=["H1", "L1"],
    t0=1000000000,
    end=1000000010,
)

# Access the composed element for pipeline connection
element = source.element

# Access source pads
pads = source.srcs  # {"H1:FAKE-STRAIN": SourcePad(...), ...}
```

### Source-Specific Options

#### Fake Sources (white, sin, impulse)

```python notest
from sgnligo.sources.datasource_v2.sources import ImpulseSource

source = ImpulseSource(
    name="impulse_source",
    ifos=["H1"],
    sample_rate=4096,
    t0=1000,
    end=1010,
    impulse_position=100,  # Sample index (-1 for random)
)
```

#### GWData Noise Sources

```python notest
from sgnligo.sources.datasource_v2.sources import GWDataNoiseComposedSource

source = GWDataNoiseComposedSource(
    name="noise",
    ifos=["H1"],
    t0=1000,
    end=1010,
    channel_pattern="{ifo}:FAKE-STRAIN",  # Default pattern
    # Optional state vector gating
    state_vector_on_dict={"H1": 3},  # Bitmask
    state_segments_file="segments.txt",
    state_sample_rate=16,
)
```

#### Frame Sources

```python notest
from sgnligo.sources.datasource_v2.sources import FramesComposedSource

source = FramesComposedSource(
    name="frame_source",
    ifos=["H1"],
    frame_cache="/path/to/frames.cache",
    channel_dict={"H1": "GDS-CALIB_STRAIN"},
    t0=1000000000,
    end=1000001000,
    # Optional segments
    segments_file="segments.xml",
    segments_name="H1:DCH-ANALYSIS_READY:1",
    # Optional injections
    noiseless_inj_frame_cache="/path/to/inj.cache",
    noiseless_inj_channel_dict={"H1": "INJECTION"},
)
```

#### DevShm Sources (Real-time)

```python notest
from sgnligo.sources.datasource_v2.sources import DevShmComposedSource

source = DevShmComposedSource(
    name="live_source",
    ifos=["H1"],
    channel_dict={"H1": "GDS-CALIB_STRAIN"},
    shared_memory_dict={"H1": "/dev/shm/kafka/H1_llhoft"},  # noqa: S108
    state_channel_dict={"H1": "GDS-CALIB_STATE_VECTOR"},
    state_vector_on_dict={"H1": 3},
    discont_wait_time=60.0,
    queue_timeout=1.0,
)
```

#### Injected Noise Sources

```python notest
from sgnligo.sources import InjectedNoiseSource

# With test mode (generates test injections)
source = InjectedNoiseSource(
    name="inj_source",
    ifos=["H1", "L1"],
    t0=1000,
    duration=100,
    test_mode="bbh",  # "bns", "nsbh", or "bbh"
)

# With injection file
source = InjectedNoiseSource(
    name="inj_source",
    ifos=["H1", "L1"],
    t0=1000,
    duration=100,
    injection_file="injections.xml",
)
```

## CLI Arguments

The `DataSource.create_cli_parser()` method creates a parser with these arguments:

| Argument | Description |
|----------|-------------|
| `--data-source` | Source type (required) |
| `--channel-name` | IFO=channel mapping (required, repeatable) |
| `--gps-start-time` | GPS start time |
| `--gps-end-time` | GPS end time |
| `--sample-rate` | Sample rate in Hz |
| `--frame-cache` | Path to frame cache file |
| `--segments-file` | Path to segments XML |
| `--segments-name` | Segment name in XML |
| `--noiseless-inj-frame-cache` | Injection frame cache |
| `--noiseless-inj-channel-name` | Injection channel (repeatable) |
| `--state-channel-name` | State channel (repeatable) |
| `--state-vector-on-bits` | State bitmask (repeatable) |
| `--shared-memory-dir` | Shared memory path (repeatable) |
| `--discont-wait-time` | Discontinuity timeout (default: 60) |
| `--queue-timeout` | Queue timeout (default: 1) |
| `--impulse-position` | Impulse position (default: -1) |
| `--state-segments-file` | State segments file |
| `--state-sample-rate` | State sample rate (default: 16) |
| `--verbose` | Enable verbose output |
| `--help-source SOURCE` | Show help for a specific source type |
| `--list-sources` | List all available source types |

## Source-Specific Help

The CLI provides discovery and help options:

```bash
# List all available sources with descriptions
python my_pipeline.py --list-sources

# Show detailed help for a specific source
python my_pipeline.py --help-source frames
```

Example output from `--list-sources`:

```text
Available data sources:

Offline Sources (require --gps-start-time and --gps-end-time):
  frames                    Read from GWF frame files
  gwdata-noise              Colored Gaussian noise with LIGO PSD
  impulse                   Impulse test signal
  injected-noise            Colored noise with GW injections
  sin                       Sinusoidal test signal
  white                     Gaussian white noise

Real-time Sources:
  arrakis                   Kafka streaming source
  devshm                    Read from shared memory
  gwdata-noise-realtime     Real-time colored Gaussian noise
  impulse-realtime          Real-time impulse test signal
  sin-realtime              Real-time sinusoidal test signal
  white-realtime            Real-time Gaussian white noise

Use --help-source <name> for detailed options for a specific source.
```

## Source Latency Tracking

For monitoring pipeline performance, enable latency tracking via the `latency_interval` parameter:

```python notest
from sgn.apps import Pipeline
from sgnligo.sources.datasource_v2 import DataSource

# Enable latency tracking with 1-second interval
source = DataSource(
    data_source="white",
    name="source",
    ifos=["H1", "L1"],
    sample_rate=4096,
    t0=1000,
    end=1010,
    latency_interval=1.0,  # seconds
)

pipeline = Pipeline()
pipeline.connect(source.element, downstream_element)

# Latency outputs appear as additional source pads: H1_latency, L1_latency
print(list(source.srcs.keys()))
# ['H1', 'L1', 'H1_latency', 'L1_latency']
```

## Extending with Custom Sources

Create custom sources by subclassing `ComposedSourceBase`:

```python notest
from dataclasses import dataclass
from typing import ClassVar, List

from sgnts.compose import TSCompose, TSComposedSourceElement

from sgnligo.sources.composed_base import ComposedSourceBase
from sgnligo.sources.datasource_v2 import register_composed_source


@register_composed_source
@dataclass
class MyCustomSource(ComposedSourceBase):
    """My custom data source."""

    # Required parameters
    ifos: List[str]
    t0: float
    end: float

    # Optional parameters
    verbose: bool = False

    # Class metadata (required)
    source_type: ClassVar[str] = "my-custom"
    description: ClassVar[str] = "My custom data source"

    def _validate(self) -> None:
        """Validate parameters."""
        if self.t0 >= self.end:
            raise ValueError("t0 must be less than end")

    def _build(self) -> TSComposedSourceElement:
        """Build the source element."""
        compose = TSCompose()

        for ifo in self.ifos:
            # Create your source elements
            my_source = MySourceElement(name=f"{self.name}_{ifo}", ...)
            compose.insert(my_source)

        return compose.as_source(name=self.name)
```

Then use it:

```python notest
source = MyCustomSource(
    name="custom",
    ifos=["H1"],
    t0=1000,
    end=2000,
)

# Or via DataSource dispatcher
source = DataSource(
    data_source="my-custom",
    name="custom",
    ifos=["H1"],
    t0=1000,
    end=2000,
)
```

---

## Migration Guide: Porting from datasource to datasource_v2

### Step 1: Update Imports

```python notest
# Before
from sgnligo.sources import DataSourceInfo, datasource

# After
from sgnligo.sources.datasource_v2 import DataSource
```

### Step 2: Update Source Creation

**Before:**
```python notest
info = DataSourceInfo(
    data_source="gwdata-noise",
    channel_name=["H1=H1:STRAIN", "L1=L1:STRAIN"],
    gps_start_time=1000000000,
    gps_end_time=1000000100,
    state_channel_name=["H1=H1:STATE", "L1=L1:STATE"],
    state_vector_on_bits=["H1=3", "L1=3"],
)
```

**After:**
```python notest
source = DataSource(
    data_source="gwdata-noise",
    name="source",
    ifos=["H1", "L1"],
    t0=1000000000,
    end=1000000100,
)
```

**Key changes:**

| Old Field | New Field |
|-----------|-----------|
| `data_source` | `data_source` (same) |
| `channel_name` (list of strings) | `ifos` (list) - channel pattern is automatic |
| `gps_start_time` | `t0` |
| `gps_end_time` | `end` |
| `state_channel_name` (list) | `state_channel_dict` (dict) |
| `state_vector_on_bits` (list of strings) | `state_vector_on_dict` (dict of ints) |
| `input_sample_rate` | `sample_rate` |

### Step 3: Update Pipeline Construction

**Before:**
```python notest
pipeline = Pipeline()

# datasource mutates pipeline
source_links, latency_links = datasource(pipeline, info, verbose=True)

# Add sink with link_map
sink = MySink(name="sink", sink_pad_names=["H1:STRAIN"])
pipeline.insert(sink)
pipeline.insert(link_map={
    "sink:snk:H1:STRAIN": source_links["H1"],
})

pipeline.run()
```

**After:**
```python notest
pipeline = Pipeline()

# DataSource returns a composed element
source = DataSource(data_source="gwdata-noise", name="source", ...)

# Connect using pipeline.connect()
sink = MySink(name="sink", sink_pad_names=list(source.srcs.keys()))
pipeline.connect(source.element, sink)

pipeline.run()
```

### Step 4: Update CLI Argument Handling

**Before:**
```python notest
parser = argparse.ArgumentParser()
DataSourceInfo.append_options(parser)
args = parser.parse_args()

info = DataSourceInfo(
    data_source=args.data_source,
    channel_name=args.channel_name,
    gps_start_time=args.gps_start_time,
    # ... etc
)
```

**After:**
```python notest
from sgnligo.sources.datasource_v2 import (
    check_composed_help_options,
    DataSource,
)

if check_composed_help_options():
    sys.exit(0)

parser = DataSource.create_cli_parser()
args = parser.parse_args()

source = DataSource.from_cli(args, name="source")
```

### Gradual Migration Strategy

For large codebases, migrate gradually:

1. **Add datasource_v2** alongside existing code (no breaking changes)
2. **Migrate new code** to use datasource_v2
3. **Migrate existing code** module by module
4. **Remove old datasource** when migration is complete

Both modules can coexist indefinitely since they use different import paths.
