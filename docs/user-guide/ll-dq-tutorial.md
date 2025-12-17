# LL-DQ Tutorial: Calculating Horizon Distance and Range

## Overview

LL-DQ (Low-Latency Data Quality) is a tool for calculating and monitoring the horizon distance and range of gravitational wave detectors. The horizon distance represents how far away a detector can see a specific type of binary merger (typically a 1.4 + 1.4 solar mass binary neutron star). This is a key metric for understanding detector sensitivity in real-time.

This tutorial will walk you through:

1. Calculating horizon distance with simulated data
2. Understanding the output
3. Monitoring range history with Kafka
4. Using state vectors for data quality gating

## Prerequisites

Before starting, ensure you have installed the sgn-ligo package:

```bash
pip install -e /path/to/sgn-ligo
```

## Quick Start: Basic Horizon Distance Calculation

The simplest way to get started is to use simulated gravitational wave detector data and calculate the horizon distance.

### Step 1: Run with Simulated Data

```bash
sgn-ligo-ll-dq \
    --data-source gwdata-noise-realtime \
    --channel-name H1=H1:FAKE-STRAIN \
    --gps-start-time 1400000000 \
    --gps-end-time 1400000010 \
    --whiten-sample-rate 2048 \
    --psd-fft-length 8 \
    --analysis-tag test \
    --verbose
```

This command:
- Generates 10 seconds of simulated LIGO Hanford strain data
- Whitens the data at 2048 Hz (downsampled from the native 16384 Hz)
- Calculates the power spectral density (PSD) using 8-second FFT windows
- Computes horizon distance for a 1.4 + 1.4 solar mass binary neutron star merger
- Prints the range history to console

### Step 2: Understanding the Output

You'll see JSON output showing the range and horizon distance over time:

```json
{
  "topic": "sgnl.test.range_history",
  "tags": ["H1"],
  "data_type": "time_series",
  "timestamp": 1735056789.123,
  "data": {
    "time": [1400000008.0, 1400000009.0, 1400000010.0],
    "data": [
      {
        "horizon_distance_Mpc": 85.3,
        "range_Mpc": 42.1
      },
      {
        "horizon_distance_Mpc": 86.1,
        "range_Mpc": 42.5
      },
      ...
    ]
  }
}
```

Each data point contains:
- `horizon_distance_Mpc`: Distance in megaparsecs to which the detector can see an optimally oriented binary neutron star merger
- `range_Mpc`: Average distance accounting for all sky locations and orientations (typically ~50% of horizon distance)

!!! note "Channel Name Format"
    The `--channel-name` option uses the format `IFO=CHANNEL-NAME`. For example, `H1=H1:FAKE-STRAIN` means interferometer `H1` with channel name `H1:FAKE-STRAIN`. This format allows the tool to associate channels with specific detectors.

!!! note "Why 8 seconds of delay?"
    The first horizon distance measurement appears after 8 seconds because that's the `--psd-fft-length`. The PSD estimator needs at least one full FFT window before it can produce a valid spectrum.

## Understanding Key Parameters

### Waveform Parameters

The horizon distance calculation uses a binary neutron star (BNS) reference:

- **Masses**: 1.4 + 1.4 solar masses (typical neutron stars)
- **Approximant**: IMRPhenomD (inspiral-merger-ringdown waveform model)
- **Frequency range**: 15 Hz to 900 Hz

You can customize these with command-line options:

```bash
sgn-ligo-ll-dq \
    --data-source gwdata-noise-realtime \
    --channel-name H1=H1:FAKE-STRAIN \
    --gps-start-time 1400000000 \
    --gps-end-time 1400000030 \
    --horizon-approximant SEOBNRv4 \
    --horizon-f-min 20.0 \
    --horizon-f-max 1024.0 \
    --verbose
```

### PSD Estimation Parameters

The quality of the horizon distance calculation depends on accurate PSD estimation:

- `--psd-fft-length`: Length in seconds for FFT windows (default: 8)
  - Longer windows → better frequency resolution, more latency
  - Shorter windows → faster updates, less accurate at low frequencies

- `--whiten-sample-rate`: Sample rate for whitening (default: 2048 Hz)
  - Must be at least 2× the highest frequency of interest
  - Lower rates reduce computational cost

```bash
sgn-ligo-ll-dq \
    --data-source gwdata-noise-realtime \
    --channel-name H1=H1:FAKE-STRAIN \
    --gps-start-time 1400000000 \
    --gps-end-time 1400000060 \
    --psd-fft-length 16 \
    --whiten-sample-rate 4096 \
    --verbose
```

## Sending Results to Kafka

For real-time monitoring, you can send the range history to a Kafka server:

```bash
sgn-ligo-ll-dq \
    --data-source gwdata-noise-realtime \
    --channel-name H1=H1:FAKE-STRAIN \
    --gps-start-time 1400000000 \
    --gps-end-time 1400000100 \
    --output-kafka-server localhost:9092 \
    --analysis-tag production \
    --verbose
```

The data will be published to the Kafka topic: `sgnl.production.range_history`

The topic name format is: `sgnl.<analysis-tag>.range_history`

You can consume this data with any Kafka client for:
- Real-time monitoring dashboards
- Alerting systems
- Long-term range history tracking
- Detector performance analysis

## Using a Reference PSD

If you have a reference PSD from a quiet period of detector operation, you can use it as a baseline:

```bash
sgn-ligo-ll-dq \
    --data-source gwdata-noise-realtime \
    --channel-name H1=H1:FAKE-STRAIN \
    --gps-start-time 1400000000 \
    --gps-end-time 1400000100 \
    --reference-psd /path/to/H1_reference_psd.xml.gz \
    --track-psd \
    --verbose
```

When using `--reference-psd`:
- The PSD starts from the reference spectrum
- With `--track-psd`, the PSD adapts over time as new data arrives
- Without `--track-psd`, the PSD remains fixed at the reference

This is useful for:
- Consistent horizon distance calculations across different time periods
- Comparing current performance to historical performance
- Faster startup (no need to wait for PSD convergence)

## Advanced: Data Quality Gating with State Vectors

In real detector operations, you want to calculate horizon distance only when the detector is in a good state. You can use state vectors to gate the data.

### Step 1: Create State Segments File

Create a file `state_segments.txt` defining when the detector is in good vs. bad states:

```bash
# Format: start_gps end_gps bitmask_value
# Value 3 = bits 0 and 1 set (HOFT_OK + OBS_INTENT) = good state
# Value 0 = no bits set = bad state
1400000000 1400000040 3    # Good observing
1400000040 1400000050 0    # Bad state (e.g., glitch, maintenance)
1400000050 1400000100 3    # Good observing resumed
```

### Step 2: Run with State Vector Gating

```bash
sgn-ligo-ll-dq \
    --data-source gwdata-noise-realtime \
    --channel-name H1=H1:FAKE-STRAIN \
    --state-channel-name H1=H1:FAKE-STATE_VECTOR \
    --state-vector-on-bits H1=3 \
    --state-segments-file state_segments.txt \
    --gps-start-time 1400000000 \
    --gps-end-time 1400000100 \
    --output-kafka-server localhost:9092 \
    --analysis-tag production \
    --verbose
```

This pipeline:
1. Generates simulated strain data (H1:FAKE-STRAIN)
2. Generates state vector data from `state_segments.txt`
3. Applies a bitmask filter: only passes data when state & 3 == 3
4. Gates the strain: only processes data during good states
5. Calculates horizon distance only during good segments
6. Outputs gaps during bad state periods (seconds 40-50)

The bitmask value `3` means bits 0 and 1 must be set:
- Bit 0: HOFT_OK (h(t) data is valid)
- Bit 1: OBS_INTENT (detector intends to be observing)

!!! tip "State Vector Behavior"
    During bad states (seconds 40-50), the pipeline will output gap buffers in the range history. Downstream consumers should handle these gaps appropriately (e.g., show "no data" in dashboards).

## Working with Real Detector Data

When you're ready to work with real detector data, you can use the `devshm` source:

```bash
sgn-ligo-ll-dq \
    --data-source devshm \
    --channel-name H1=H1:GDS-CALIB_STRAIN \
    --state-channel-name H1=H1:GDS-CALIB_STATE_VECTOR \
    --state-vector-on-bits H1=3 \
    --shared-memory-dir H1=/dev/shm/kafka/H1_llhoft \
    --output-kafka-server kafka.ligo.org:9092 \
    --analysis-tag O4 \
    --verbose
```

This connects to the real-time data stream from shared memory and processes live detector data.

## Complete Example Script

Here's a complete example script that demonstrates the full workflow:

```bash
#!/bin/bash
# ll_dq_demo.sh - Complete LL-DQ demonstration

# Configuration
ANALYSIS_TAG="demo"
START_TIME=1400000000
END_TIME=1400000100
KAFKA_SERVER="localhost:9092"  # Set to empty string to disable Kafka output

# Create state segments file for data quality
cat > state_segments.txt << EOF
# Simulating detector states over 100 seconds
# Format: start_gps end_gps bitmask_value
1400000000 1400000030 3    # Good observing
1400000030 1400000035 0    # Glitch/bad state
1400000035 1400000070 3    # Good observing resumed
1400000070 1400000075 1    # Only HOFT_OK, no OBS_INTENT
1400000075 1400000100 3    # Good observing
EOF

echo "=== Running LL-DQ with simulated data ==="

# Build the command
CMD="sgn-ligo-ll-dq \
    --data-source gwdata-noise-realtime \
    --channel-name H1=H1:FAKE-STRAIN \
    --state-channel-name H1=H1:FAKE-STATE_VECTOR \
    --state-vector-on-bits H1=3 \
    --state-segments-file state_segments.txt \
    --gps-start-time $START_TIME \
    --gps-end-time $END_TIME \
    --whiten-sample-rate 2048 \
    --psd-fft-length 8 \
    --horizon-approximant IMRPhenomD \
    --horizon-f-min 15.0 \
    --horizon-f-max 900.0 \
    --analysis-tag $ANALYSIS_TAG \
    --verbose"

# Add Kafka server if specified
if [ -n "$KAFKA_SERVER" ]; then
    CMD="$CMD --output-kafka-server $KAFKA_SERVER"
fi

# Run the command
eval $CMD

# Clean up
rm -f state_segments.txt

echo -e "\n=== Done! ==="
```

## Summary

LL-DQ provides a powerful way to:
- Monitor detector sensitivity in real-time
- Calculate horizon distance for binary neutron star mergers
- Track detector performance over time
- Integrate with Kafka for live monitoring
- Apply data quality gating using state vectors

Key features:
- Works with both real detector data (`devshm`) and simulated data (`gwdata-noise-realtime`)
- Flexible PSD estimation with customizable parameters
- Support for reference PSDs
- State vector gating for data quality control
- Real-time output to Kafka for monitoring systems

For more information on the SGN framework and other tools, see the [SGN documentation](https://docs.ligo.org/greg/sgn/).
