# SimInspiralSource Tutorial

The `SimInspiralSource` is a data source element that generates gravitational wave
injection signals from compact binary coalescences (CBCs). It can either read injection
parameters from standard LIGO/Virgo injection files (XML or HDF5), or generate
periodic test injections automatically using **test mode**. All waveforms are
generated and projected onto multiple detectors using LALSimulation.

## Overview

Gravitational wave injections are simulated signals added to detector data streams
for testing and validation of search pipelines. The `SimInspiralSource` provides
a way to generate these injection signals with accurate:

- Waveform generation using LALSimulation approximants
- Detector projection including antenna patterns
- Time delays between detectors
- Phase corrections for each detector

This is useful for:

- Testing gravitational wave search pipelines
- Validating detection algorithms
- Creating simulated data with known signals
- Educational demonstrations

## Supported Injection File Formats

### XML Format (LIGOLW)

The standard LIGO/Virgo format for injection files is LIGOLW XML containing a
`sim_inspiral` table. Here's an example structure:

```xml
<?xml version='1.0' encoding='utf-8'?>
<!DOCTYPE LIGO_LW SYSTEM "http://ldas-sw.ligo.caltech.edu/doc/ligolwAPI/html/ligolw_dtd.txt">
<LIGO_LW>
    <Table Name="sim_inspiral:table">
        <Column Name="mass1" Type="real_4"/>
        <Column Name="mass2" Type="real_4"/>
        <Column Name="spin1z" Type="real_4"/>
        <Column Name="spin2z" Type="real_4"/>
        <Column Name="distance" Type="real_4"/>
        <Column Name="inclination" Type="real_4"/>
        <Column Name="coa_phase" Type="real_4"/>
        <Column Name="polarization" Type="real_4"/>
        <Column Name="longitude" Type="real_4"/>
        <Column Name="latitude" Type="real_4"/>
        <Column Name="geocent_end_time" Type="int_4s"/>
        <Column Name="geocent_end_time_ns" Type="int_4s"/>
        <Column Name="waveform" Type="lstring"/>
        <Column Name="f_lower" Type="real_4"/>
        <Stream Name="sim_inspiral:table" Type="Local" Delimiter=",">
            30.0,30.0,0.3,0.3,500,0.4,0,0.8,1.2,0.5,1400000005,0,"IMRPhenomD",20,
        </Stream>
    </Table>
</LIGO_LW>
```

### HDF5 Format

HDF5 injection files should contain datasets for each injection parameter, either
at the root level or in an `injections` group:

```python
import h5py

with h5py.File("injections.hdf5", "w") as f:
    grp = f.create_group("injections")
    grp.create_dataset("mass1", data=[30.0, 35.0])
    grp.create_dataset("mass2", data=[30.0, 30.0])
    grp.create_dataset("distance", data=[500.0, 600.0])
    grp.create_dataset("geocent_end_time", data=[1400000005.0, 1400000008.0])
    grp.create_dataset("approximant", data=[b"IMRPhenomD", b"IMRPhenomD"])
    # ... other parameters
```

## Test Mode

Test mode provides a quick way to generate periodic gravitational wave injections
without creating an injection file. This is useful for:

- Quick testing and development
- Demonstrations and tutorials
- Verifying pipeline operation
- Educational purposes

### Test Mode Types

Three injection types are available:

| Mode | Masses | Distance | Description |
|------|--------|----------|-------------|
| `bns` | 1.4 + 1.4 M☉ | 100 Mpc | Binary Neutron Star |
| `nsbh` | 10 + 1.4 M☉ | 200 Mpc | Neutron Star - Black Hole |
| `bbh` | 30 + 30 M☉ | 500 Mpc | Binary Black Hole |

All test injections use:

- **Approximant**: IMRPhenomD
- **Spins**: Non-spinning (all zero)
- **Inclination**: Face-on (0 radians)
- **Injection interval**: Every 30 seconds
- **Sky position**: Directly overhead State College, PA (40.79°N, 77.86°W)

### Basic Test Mode Example

Using test mode is as simple as specifying `test_mode` instead of `injection_file`:

```python
from sgn.apps import Pipeline
from sgnts.sinks import TSPlotSink
from sgnligo.sources import SimInspiralSource

# Create source with test mode - no injection file needed!
source = SimInspiralSource(
    name="TestInjections",
    test_mode="bbh",  # Options: "bns", "nsbh", "bbh"
    ifos=["H1", "L1"],
    t0=1400000020.0,
    duration=120.0,  # 2 minutes = 4 injections
    sample_rate=4096,
    f_min=20.0,
)

# Create plot sink
sink = TSPlotSink(
    name="Strain",
    sink_pad_names=["H1:INJ-STRAIN", "L1:INJ-STRAIN"],
)

# Build and run pipeline
pipeline = Pipeline()
pipeline.insert(source, sink)
pipeline.insert(
    link_map={
        "Strain:snk:H1:INJ-STRAIN": "TestInjections:src:H1:INJ-STRAIN",
        "Strain:snk:L1:INJ-STRAIN": "TestInjections:src:L1:INJ-STRAIN",
    }
)
pipeline.run()

# Plot results
fig, ax = sink.plot(
    layout="overlay",
    labels={"H1:INJ-STRAIN": "H1", "L1:INJ-STRAIN": "L1"},
    title="BBH Test Mode: 30+30 Msun at 500 Mpc",
    time_unit="gps",
)
```

### Comparing Injection Types

Each test mode produces signals with different characteristics:

```python
from sgnligo.sources import SimInspiralSource
from sgnligo.sources.sim_inspiral_source import TEST_MODE_PARAMS

# Print available test modes and their parameters
for mode, params in TEST_MODE_PARAMS.items():
    print(f"{mode.upper()}: {params['mass1']:.1f}+{params['mass2']:.1f} Msun "
          f"at {params['distance']:.0f} Mpc")
# Output:
# BNS: 1.4+1.4 Msun at 100 Mpc
# NSBH: 10.0+1.4 Msun at 200 Mpc
# BBH: 30.0+30.0 Msun at 500 Mpc
```

**Signal characteristics by type:**

- **BNS**: Long inspiral (100+ seconds from 20 Hz), highest amplitude due to
  closest distance. Multiple injections will overlap in time.
- **NSBH**: Medium inspiral duration, moderate amplitude.
- **BBH**: Short inspiral (seconds), lower amplitude due to greater distance.

### Sky Position: Overhead State College, PA

Test injections are positioned directly overhead (at zenith) of State College,
Pennsylvania at the moment of coalescence. This means:

- **Declination** = Latitude of State College = 40.79° = 0.712 radians
- **Right Ascension** = Changes with time to stay overhead

The right ascension is calculated using the Greenwich Mean Sidereal Time (GMST):

```python
from sgnligo.sources.sim_inspiral_source import (
    calculate_overhead_ra,
    STATE_COLLEGE_LAT_RAD,
    STATE_COLLEGE_LON_RAD,
)
import numpy as np

# Calculate RA for a source overhead at GPS time 1400000000
gps_time = 1400000000.0
ra = calculate_overhead_ra(gps_time, STATE_COLLEGE_LON_RAD)
print(f"RA at t={gps_time}: {np.degrees(ra):.2f} degrees")

# RA changes by ~0.125 degrees per 30 seconds (Earth rotation)
ra_later = calculate_overhead_ra(gps_time + 30, STATE_COLLEGE_LON_RAD)
print(f"RA at t={gps_time + 30}: {np.degrees(ra_later):.2f} degrees")
```

### Test Mode vs Injection File

| Feature | Test Mode | Injection File |
|---------|-----------|----------------|
| Setup complexity | Minimal | Requires file creation |
| Parameter control | Fixed presets | Full control |
| Sky position | Fixed (State College) | Configurable |
| Injection timing | Every 30 seconds | Arbitrary |
| Use case | Testing, demos | Production, analysis |

**Note**: You must specify either `test_mode` OR `injection_file`, but not both.

## Basic Usage

Here's a minimal example using `SimInspiralSource`:

```python
import os
import tempfile

from sgn.apps import Pipeline
from sgnts.sinks import NullSeriesSink
from sgnligo.sources import SimInspiralSource


def create_simple_injection_xml(filepath: str, t0: float) -> None:
    """Create a simple XML injection file with one BBH event."""
    t1 = int(t0 + 5)
    dtd = "http://ldas-sw.ligo.caltech.edu/doc/ligolwAPI/html/ligolw_dtd.txt"
    xml_content = f"""<?xml version='1.0' encoding='utf-8'?>
<!DOCTYPE LIGO_LW SYSTEM "{dtd}">
<LIGO_LW>
    <Table Name="sim_inspiral:table">
        <Column Name="mass1" Type="real_4"/>
        <Column Name="mass2" Type="real_4"/>
        <Column Name="spin1x" Type="real_4"/>
        <Column Name="spin1y" Type="real_4"/>
        <Column Name="spin1z" Type="real_4"/>
        <Column Name="spin2x" Type="real_4"/>
        <Column Name="spin2y" Type="real_4"/>
        <Column Name="spin2z" Type="real_4"/>
        <Column Name="distance" Type="real_4"/>
        <Column Name="inclination" Type="real_4"/>
        <Column Name="coa_phase" Type="real_4"/>
        <Column Name="polarization" Type="real_4"/>
        <Column Name="longitude" Type="real_4"/>
        <Column Name="latitude" Type="real_4"/>
        <Column Name="geocent_end_time" Type="int_4s"/>
        <Column Name="geocent_end_time_ns" Type="int_4s"/>
        <Column Name="waveform" Type="lstring"/>
        <Column Name="f_lower" Type="real_4"/>
        <Stream Name="sim_inspiral:table" Type="Local" Delimiter=",">
            30.0,30.0,0,0,0.3,0,0,0.3,500,0.4,0,0.8,1.2,0.5,{t1},0,"IMRPhenomD",20,
        </Stream>
    </Table>
</LIGO_LW>
"""
    with open(filepath, "w") as f:
        f.write(xml_content)


def main():
    """Run the basic SimInspiralSource example."""
    t0 = 1400000000.0

    # Create temporary injection file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        injection_file = f.name

    try:
        create_simple_injection_xml(injection_file, t0)

        # Create the injection source
        source = SimInspiralSource(
            name="Injections",
            injection_file=injection_file,
            ifos=["H1", "L1"],
            t0=t0,
            duration=10.0,
            sample_rate=4096,
            f_min=20.0,
        )

        # Create a sink for each channel
        sink = NullSeriesSink(
            name="Sink",
            sink_pad_names=["H1:INJ-STRAIN", "L1:INJ-STRAIN"],
        )

        # Build and run the pipeline
        pipeline = Pipeline()
        pipeline.insert(source, sink)
        pipeline.insert(
            link_map={
                "Sink:snk:H1:INJ-STRAIN": "Injections:src:H1:INJ-STRAIN",
                "Sink:snk:L1:INJ-STRAIN": "Injections:src:L1:INJ-STRAIN",
            }
        )
        pipeline.run()
    finally:
        os.unlink(injection_file)


if __name__ == "__main__":
    main()
```

## Complete Example: BBH Injection Visualization

This example creates a pipeline that generates two binary black hole (BBH)
injections and visualizes them across H1, L1, and V1 detectors.

```python
#!/usr/bin/env python3
"""Example script demonstrating SimInspiralSource with BBH injections.

This script creates a pipeline that:
1. Generates two 30+30 solar mass BBH injections at t=5s and t=8s
2. Runs for 10 seconds across H1, L1, V1 detectors
3. Plots the results using TSPlotSink with overlay layout
"""

import tempfile

import matplotlib.pyplot as plt
from sgn.apps import Pipeline
from sgnts.sinks import TSPlotSink

from sgnligo.sources import SimInspiralSource


def create_injection_xml(filepath: str, t0: float) -> None:
    """Create an XML injection file with two BBH events.

    Args:
        filepath: Path to write the XML file
        t0: GPS start time of the pipeline (injections at t0+5 and t0+8)
    """
    # Two 30+30 solar mass BBH mergers
    # First at t0+5 seconds, second at t0+8 seconds
    t1, t2 = int(t0 + 5), int(t0 + 8)
    dtd = "http://ldas-sw.ligo.caltech.edu/doc/ligolwAPI/html/ligolw_dtd.txt"
    xml_content = f"""<?xml version='1.0' encoding='utf-8'?>
<!DOCTYPE LIGO_LW SYSTEM "{dtd}">
<LIGO_LW>
    <Table Name="sim_inspiral:table">
        <Column Name="mass1" Type="real_4"/>
        <Column Name="mass2" Type="real_4"/>
        <Column Name="spin1x" Type="real_4"/>
        <Column Name="spin1y" Type="real_4"/>
        <Column Name="spin1z" Type="real_4"/>
        <Column Name="spin2x" Type="real_4"/>
        <Column Name="spin2y" Type="real_4"/>
        <Column Name="spin2z" Type="real_4"/>
        <Column Name="distance" Type="real_4"/>
        <Column Name="inclination" Type="real_4"/>
        <Column Name="coa_phase" Type="real_4"/>
        <Column Name="polarization" Type="real_4"/>
        <Column Name="longitude" Type="real_4"/>
        <Column Name="latitude" Type="real_4"/>
        <Column Name="geocent_end_time" Type="int_4s"/>
        <Column Name="geocent_end_time_ns" Type="int_4s"/>
        <Column Name="waveform" Type="lstring"/>
        <Column Name="f_lower" Type="real_4"/>
        <Stream Name="sim_inspiral:table" Type="Local" Delimiter=",">
            30.0,30.0,0,0,0.3,0,0,0.3,500,0.4,0,0.8,1.2,0.5,{t1},0,"IMRPhenomD",20,
            30.0,30.0,0,0,-0.2,0,0,-0.2,400,0.6,1.57,0.3,2.5,-0.3,{t2},0,"IMRPhenomD",20,
        </Stream>
    </Table>
</LIGO_LW>
"""
    with open(filepath, "w") as f:
        f.write(xml_content)


def main():
    """Run the BBH injection example pipeline."""
    # GPS start time (arbitrary, but realistic)
    t0 = 1400000000.0
    duration = 10.0

    # Create temporary injection file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        injection_file = f.name

    try:
        # Create injection XML with two BBH events
        print("Creating injection file...")
        create_injection_xml(injection_file, t0)

        # Create injection source for H1, L1, V1
        print("Setting up SimInspiralSource...")
        source = SimInspiralSource(
            name="BBH_Injections",
            injection_file=injection_file,
            ifos=["H1", "L1", "V1"],
            t0=t0,
            duration=duration,
            sample_rate=4096,  # 4096 Hz is sufficient for BBH
            f_min=20.0,
        )

        # Create plot sink for visualization
        sink = TSPlotSink(
            name="Detector_Strain",
            sink_pad_names=["H1:INJ-STRAIN", "L1:INJ-STRAIN", "V1:INJ-STRAIN"],
        )

        # Build and run pipeline
        print("Building pipeline...")
        pipeline = Pipeline()
        pipeline.insert(source, sink)
        pipeline.insert(
            link_map={
                "Detector_Strain:snk:H1:INJ-STRAIN": "BBH_Injections:src:H1:INJ-STRAIN",
                "Detector_Strain:snk:L1:INJ-STRAIN": "BBH_Injections:src:L1:INJ-STRAIN",
                "Detector_Strain:snk:V1:INJ-STRAIN": "BBH_Injections:src:V1:INJ-STRAIN",
            }
        )

        print(f"Running pipeline for {duration} seconds...")
        print(f"  - Two 30+30 Msun BBH injections at t={t0+5:.0f}s and t={t0+8:.0f}s")
        print("  - Detectors: H1, L1, V1")
        pipeline.run()

        # Plot results with overlay
        print("Plotting results...")
        fig, ax = sink.plot(
            layout="overlay",
            labels={
                "H1:INJ-STRAIN": "H1",
                "L1:INJ-STRAIN": "L1",
                "V1:INJ-STRAIN": "V1",
            },
            title="BBH Injections (30+30 Msun) at H1, L1, V1",
            time_unit="gps",
            figsize=(14, 6),
        )

        # Add legend and annotations
        ax.legend(loc="upper right")
        ax.set_ylabel("Strain")

        # Mark the merger times
        ax.axvline(
            x=t0 + 5, color="gray", linestyle="--", alpha=0.5, label="_nolegend_"
        )
        ax.axvline(
            x=t0 + 8, color="gray", linestyle="--", alpha=0.5, label="_nolegend_"
        )
        ax.text(t0 + 5, ax.get_ylim()[1] * 0.9, "Merger 1", ha="center", fontsize=9)
        ax.text(t0 + 8, ax.get_ylim()[1] * 0.9, "Merger 2", ha="center", fontsize=9)

        plt.tight_layout()
        plt.savefig("bbh_injection_example.png", dpi=150)
        print("Saved plot to bbh_injection_example.png")
        plt.show()

    finally:
        # Cleanup
        import os

        os.unlink(injection_file)


if __name__ == "__main__":
    main()
```

Save this as `bbh_injection_example.py` and run it:

```bash
python bbh_injection_example.py
```

## SimInspiralSource Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Element name for the pipeline |
| `injection_file` | str | Path to injection file (XML or HDF5). **Mutually exclusive with `test_mode`** |
| `test_mode` | str | Test mode type: `"bns"`, `"nsbh"`, or `"bbh"`. **Mutually exclusive with `injection_file`** |
| `t0` | float | GPS start time |
| `duration` | float | Duration in seconds |

**Note**: You must specify exactly one of `injection_file` or `test_mode`.

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ifos` | list | `["H1", "L1"]` | List of detector prefixes (e.g., `["H1", "L1", "V1"]`) |
| `sample_rate` | int | `16384` | Sample rate in Hz |
| `f_min` | float | `10.0` | Minimum frequency for waveform generation (Hz) |
| `approximant_override` | str | `None` | Override waveform approximant for all injections |

### Supported Detectors

The following gravitational wave detectors are supported:

| Code | Detector | Location |
|------|----------|----------|
| `H1` | LIGO Hanford | Washington, USA |
| `L1` | LIGO Livingston | Louisiana, USA |
| `V1` | Virgo | Cascina, Italy |
| `K1` | KAGRA | Kamioka, Japan |
| `I1` | LIGO India | India (planned) |

### Supported Waveform Approximants

The source uses LALSimulation and supports approximants including:

**Time-Domain (TD):**
- `TaylorT1`, `TaylorT2`, `TaylorT3`, `TaylorT4`
- `EOBNRv2`, `SEOBNRv4`, `SEOBNRv4_ROM`
- `SpinTaylorT4`, `SpinTaylorT5`

**Frequency-Domain (FD) - automatically converted to TD:**
- `IMRPhenomA`, `IMRPhenomB`, `IMRPhenomC`, `IMRPhenomD`
- `IMRPhenomPv2`, `IMRPhenomXAS`, `IMRPhenomXPHM`
- `TaylorF2`

The source automatically selects between time-domain and frequency-domain
generation based on the approximant.

## Injection Parameters

Each injection in the file can specify the following parameters:

### Mass Parameters
- `mass1`, `mass2`: Component masses in solar masses

### Spin Parameters
- `spin1x`, `spin1y`, `spin1z`: Primary spin components
- `spin2x`, `spin2y`, `spin2z`: Secondary spin components

### Extrinsic Parameters
- `distance`: Luminosity distance in Mpc
- `inclination`: Inclination angle (radians)
- `coa_phase`: Coalescence phase (radians)
- `polarization`: Polarization angle (radians)
- `ra` or `longitude`: Right ascension (radians)
- `dec` or `latitude`: Declination (radians)

### Timing Parameters
- `geocent_end_time`: GPS time of merger (integer seconds)
- `geocent_end_time_ns`: Nanoseconds part of merger time

### Waveform Parameters
- `waveform` or `approximant`: Waveform approximant name
- `f_lower` or `f_ref`: Reference/lower frequency (Hz)

## Combining Injections with Noise

To create realistic simulated data, combine `SimInspiralSource` with
`GWDataNoiseSource` using an Adder transform:

```python
import os
import tempfile

from sgn.apps import Pipeline
from sgnts.sinks import NullSeriesSink
from sgnts.transforms import Adder
from sgnligo.sources import GWDataNoiseSource, SimInspiralSource


def create_injection_xml_for_noise_example(filepath: str, t0: float) -> None:
    """Create an XML injection file with one BBH event."""
    t1 = int(t0 + 5)
    dtd = "http://ldas-sw.ligo.caltech.edu/doc/ligolwAPI/html/ligolw_dtd.txt"
    xml_content = f"""<?xml version='1.0' encoding='utf-8'?>
<!DOCTYPE LIGO_LW SYSTEM "{dtd}">
<LIGO_LW>
    <Table Name="sim_inspiral:table">
        <Column Name="mass1" Type="real_4"/>
        <Column Name="mass2" Type="real_4"/>
        <Column Name="spin1x" Type="real_4"/>
        <Column Name="spin1y" Type="real_4"/>
        <Column Name="spin1z" Type="real_4"/>
        <Column Name="spin2x" Type="real_4"/>
        <Column Name="spin2y" Type="real_4"/>
        <Column Name="spin2z" Type="real_4"/>
        <Column Name="distance" Type="real_4"/>
        <Column Name="inclination" Type="real_4"/>
        <Column Name="coa_phase" Type="real_4"/>
        <Column Name="polarization" Type="real_4"/>
        <Column Name="longitude" Type="real_4"/>
        <Column Name="latitude" Type="real_4"/>
        <Column Name="geocent_end_time" Type="int_4s"/>
        <Column Name="geocent_end_time_ns" Type="int_4s"/>
        <Column Name="waveform" Type="lstring"/>
        <Column Name="f_lower" Type="real_4"/>
        <Stream Name="sim_inspiral:table" Type="Local" Delimiter=",">
            30.0,30.0,0,0,0.3,0,0,0.3,500,0.4,0,0.8,1.2,0.5,{t1},0,"IMRPhenomD",20,
        </Stream>
    </Table>
</LIGO_LW>
"""
    with open(filepath, "w") as f:
        f.write(xml_content)


def main():
    """Combine injections with noise using Adder."""
    t0 = 1400000000.0

    # Create temporary injection file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        injection_file = f.name

    try:
        create_injection_xml_for_noise_example(injection_file, t0)

        # Create noise source
        noise = GWDataNoiseSource(
            name="Noise",
            channel_dict={"H1": "H1:NOISE"},
            t0=t0,
            duration=10.0,
        )

        # Create injection source
        injections = SimInspiralSource(
            name="Injections",
            injection_file=injection_file,
            ifos=["H1"],
            t0=t0,
            duration=10.0,
            sample_rate=16384,
            f_min=20.0,
        )

        # Add signals together
        adder = Adder(
            name="Adder",
            sink_pad_names=["H1:NOISE", "H1:INJ-STRAIN"],
            source_pad_names=["H1:STRAIN"],
        )

        # Create sink (use NullSeriesSink for testing, TSPlotSink for visualization)
        sink = NullSeriesSink(name="Sink", sink_pad_names=["H1:STRAIN"])

        # Build pipeline
        pipeline = Pipeline()
        pipeline.insert(noise, injections, adder, sink)
        pipeline.insert(
            link_map={
                "Adder:snk:H1:NOISE": "Noise:src:H1:NOISE",
                "Adder:snk:H1:INJ-STRAIN": "Injections:src:H1:INJ-STRAIN",
                "Sink:snk:H1:STRAIN": "Adder:src:H1:STRAIN",
            }
        )
        pipeline.run()
    finally:
        os.unlink(injection_file)


if __name__ == "__main__":
    main()
```

## Technical Details

### Waveform Caching

The `SimInspiralSource` implements efficient waveform caching for signals that
span multiple output buffers. Waveforms are generated once and cached, then
sliced appropriately for each output buffer.

### Detector Projection

Waveforms are projected onto each detector using `lalsimulation.SimDetectorStrainREAL8TimeSeries`,
which accurately accounts for:

- Antenna pattern response (F+ and F×)
- Time delays due to detector locations
- Phase corrections

### Output Channel Names

Output channels are named as `{IFO}:INJ-STRAIN`, for example:
- `H1:INJ-STRAIN`
- `L1:INJ-STRAIN`
- `V1:INJ-STRAIN`

## Conclusion

The `SimInspiralSource` provides a powerful way to generate accurate gravitational
wave injection signals for testing and validation. Combined with other SGN elements,
you can build sophisticated pipelines for gravitational wave data analysis development.
