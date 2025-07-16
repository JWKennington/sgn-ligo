# GWIStat Tutorial: Analyzing Interferometer State Vectors

## Overview

GWIStat (Gravitational Wave Interferometer Status) is a tool for reading and interpreting state vector data from gravitational wave detectors. State vectors are bitmask values where each bit represents a different detector state or condition. This tutorial will walk you through:

1. Creating fake frame files with state vector data
2. Reading and interpreting the state vectors
3. Understanding the JSON output format
4. Sending results to Kafka or viewing them locally

## Prerequisites

Before starting, ensure you have installed the sgn-ligo package:

```bash
pip install -e /path/to/sgn-ligo
```

## Step 1: Understanding State Vectors

State vectors in gravitational wave detectors are integer values where each bit position has a specific meaning. For example:

- Bit 0: `HOFT_OK` - Indicates if the h(t) data is valid
- Bit 1: `OBS_INTENT` - Indicates if the detector is in observation mode
- Bit 2-31: Various other detector states

When multiple bits are set, the state vector represents multiple simultaneous conditions.

## Step 2: Creating Test Data with Fake Frames

First, let's generate some test frame files containing state vector data. The `sgn-ligo-fake-frames` tool creates frames with:

- A state vector channel with bitmask patterns defined in a segment file
- A strain channel with realistic colored noise

### Create a Segment File

The `sgn-ligo-fake-frames` tool requires a segment file that defines the bitmask values over time. Create a file called `segments.txt`:

```bash
# Create segment file
cat > segments.txt << EOF
# Segment file for test data
# Format: start_gps end_gps bitmask_value
1400000000 1400000016 1    # Only HOFT_OK
1400000016 1400000032 2    # Only OBS_INTENT
1400000032 1400000048 3    # Both HOFT_OK and OBS_INTENT
1400000048 1400000064 0    # No bits set
1400000064 1400000080 1    # Only HOFT_OK
EOF
```

### Generate Frame Files

```bash
# Create output directory
mkdir -p test_frames

# Generate frames with state vector data
sgn-ligo-fake-frames \
    --segment-file segments.txt \
    --state-channel L1:FAKE-STATE_VECTOR \
    --strain-channel L1:FAKE-STRAIN \
    --state-sample-rate 16 \
    --strain-sample-rate 16384 \
    --frame-duration 16 \
    --gps-start-time 1400000000 \
    --gps-end-time 1400000090 \
    --output-path "test_frames/{instruments}-{description}-{gps_start_time}-{duration}.gwf" \
    --description FAKE_DATA \
    --verbose
```

This creates frame files with:
- State vector data at 16 Hz (typical for state monitoring)
- Strain data at 16384 Hz (standard for gravitational wave analysis)

The state vector data follows the pattern defined in your segment file:

1. **Segment 1 (0-16s)**: Value 1 (0b01) - Only HOFT_OK
2. **Segment 2 (16-32s)**: Value 2 (0b10) - Only OBS_INTENT
3. **Segment 3 (32-48s)**: Value 3 (0b11) - Both HOFT_OK and OBS_INTENT
4. **Segment 4 (48-64s)**: Value 0 (0b00) - No bits set
5. **Segment 5 (64-80s)**: Value 1 (0b01) - Only HOFT_OK

!!! info "Sample Rates"
    State vector channels typically use lower sample rates (16 Hz) since they change infrequently. Strain channels require much higher sample rates (16384 Hz or 4096 Hz minimum) to capture gravitational wave signals. The tool supports different sample rates for each channel type.

!!! info "Frame Generation"
    We generate 90 seconds of data to ensure all 5 frames are written due to FrameSink's internal stride buffering. The segment file defines patterns up to 80 seconds.

## Step 3: Creating a Bit Mapping File

GWIStat uses a JSON file to map bit positions to their meanings. Create a file called `bitmask_mapping.json`:

```json
{
  "0": "HOFT_OK",
  "1": "OBS_INTENT",
  "2": "SCIENCE_MODE",
  "3": "INJECTION_MODE",
  "4": "EXCITATION_ACTIVE",
  "5": "TIMING_OK",
  "6": "OVERFLOW_DETECTED",
  "7": "DATA_VALID"
}
```

You can define meanings for any bit positions (0-31) that are relevant to your detector configuration.

## Step 4: Running GWIStat

Now let's analyze the state vector data using GWIStat:

### Basic Usage (Pretty Print to Console)

```bash
sgn-ligo-gwistat \
    --data-source frames \
    --frame-cache "test_frames/*.gwf" \
    --channel-name L1:FAKE-STATE_VECTOR \
    --mapping-file bitmask_mapping.json \
    --gps-start-time 1400000000 \
    --gps-end-time 1400000080 \
    --verbose
```

This will output JSON to the console showing the interpreted state vectors.

### Understanding the Output

The output will be JSON formatted like this:

```json
{
  "topic": "gwistat.gwistat",
  "tags": [],
  "data_type": "time_series",
  "timestamp": 1735056789.123,
  "data": {
    "time": [1400000000.0, 1400000000.0625, 1400000000.125, ...],
    "data": [
      {
        "value": 1,
        "active_bits": [0],
        "meanings": ["HOFT_OK"]
      },
      {
        "value": 1,
        "active_bits": [0],
        "meanings": ["HOFT_OK"]
      },
      ...
    ]
  }
}
```

Each entry in the data array contains:
- `value`: The decimal value of the state vector
- `active_bits`: List of bit positions that are set to 1
- `meanings`: Human-readable meanings for the active bits

## Step 5: Sending to Kafka

To send the results to a Kafka server instead of printing to console:

```bash
sgn-ligo-gwistat \
    --data-source frames \
    --frame-cache "test_frames/*.gwf" \
    --channel-name L1:FAKE-STATE_VECTOR \
    --mapping-file bitmask_mapping.json \
    --gps-start-time 1400000000 \
    --gps-end-time 1400000080 \
    --output-kafka-server localhost:9092 \
    --kafka-topic state_vector_analysis \
    --kafka-tag L1 \
    --verbose
```

This sends the interpreted state vector data to the Kafka topic `gwistat.state_vector_analysis` with the tag `L1`.

## Step 6: Real-Time Analysis with DevShm

GWIStat can read from shared memory for real-time analysis. You can either connect to an existing data stream or generate test data using `sgn-ligo-fake-frames`.

### Generating Real-Time Test Data

First, let's create real-time test data that simulates a live data stream:

```bash
# Create a segment file with state changes
cat > devshm_segments.txt << EOF
# Single segment for continuous observing mode (GPS times in seconds)
1400000000  2000000000  3     # Normal observing (HOFT_OK + OBS_INTENT)
EOF

# Create a local directory to simulate shared memory
mkdir -p ./dev/shm/kafka/L1_test

# Start real-time frame generation (run in separate terminal)
sgn-ligo-fake-frames \
    --segment-file devshm_segments.txt \
    --state-channel L1:TEST-STATE_VECTOR \
    --strain-channel L1:TEST-STRAIN \
    --state-sample-rate 16 \
    --strain-sample-rate 16384 \
    --frame-duration 1 \
    --output-path "./dev/shm/kafka/L1_test/{instruments}-{description}-{gps_start_time}-{duration}.gwf" \
    --description TEST \
    --real-time \
    --history 300 \
    --verbose
```

This will:
- Generate 1-second frames continuously in real-time
- Write to local directory `./dev/shm/kafka/L1_test/` (simulating shared memory)
- Maintain a 5-minute rolling buffer of frames (automatically deleting old frames)
- Use a constant state value of 3 (HOFT_OK + OBS_INTENT)
- Sync frame generation with wall-clock time (1 second of data per second of real time)

### Reading from DevShm

Now you can analyze the real-time data stream:

```bash
# In another terminal, start real-time analysis
sgn-ligo-gwistat \
    --data-source devshm \
    --shared-memory-dir ./dev/shm/kafka/L1_test \
    --channel-name L1:TEST-STATE_VECTOR \
    --mapping-file bitmask_mapping.json \
    --discont-wait-time 60 \
    --queue-timeout 2 \
    --verbose
```

### Connecting to Production Data

For real detector data, point to the appropriate shared memory location:

```bash
sgn-ligo-gwistat \
    --data-source devshm \
    --shared-memory-dir /dev/shm/kafka/L1_llhoft \
    --channel-name L1:GDS-CALIB_STATE_VECTOR \
    --mapping-file bitmask_mapping.json \
    --discont-wait-time 60 \
    --queue-timeout 2 \
    --verbose
```

## Troubleshooting

### Common Issues

1. **No frames found**: Ensure your glob pattern matches the frame files:
   ```bash
   ls -la test_frames/*.gwf
   ```

2. **Channel not found**: Verify the channel name matches exactly what's in the frames:
   ```bash
   # List channels in a frame file
   gwpy-frinfo test_frames/L1-FAKE_DATA-1400000000-16.gwf
   ```

3. **Time mismatch**: Ensure GPS times match the data in your frames

4. **Segment file required**: The `sgn-ligo-fake-frames` tool requires a segment file via the `--segment-file` argument. The file must have three columns: start_gps, end_gps, and bitmask_value

### Debugging Output

Use `--verbose` to see detailed processing information:
- First few samples from each buffer
- Frame file loading progress
- Bit interpretation details

## Complete Example Script

Here's a complete example script that demonstrates the full workflow:

```bash
#!/bin/bash
# gwistat_demo.sh - Complete GWIStat demonstration

# Set up test directory
TEST_DIR="/tmp/gwistat_demo_$$"
mkdir -p "$TEST_DIR/frames"

# Create bit mapping file
cat > "$TEST_DIR/mapping.json" << EOF
{
  "0": "HOFT_OK",
  "1": "OBS_INTENT",
  "2": "SCIENCE_MODE",
  "3": "INJECTION_MODE"
}
EOF

# Create segment file for frame generation
cat > "$TEST_DIR/segments.txt" << EOF
# Test segments
1400000000 1400000016 1
1400000016 1400000032 2
1400000032 1400000048 3
1400000048 1400000064 0
1400000064 1400000080 1
EOF

echo "=== Generating test frames ==="
sgn-ligo-fake-frames \
    --segment-file "$TEST_DIR/segments.txt" \
    --state-channel L1:TEST-STATE_VECTOR \
    --strain-channel L1:TEST-STRAIN \
    --state-sample-rate 16 \
    --strain-sample-rate 16384 \
    --frame-duration 16 \
    --gps-start-time 1400000000 \
    --gps-end-time 1400000090 \
    --output-path "$TEST_DIR/frames/{instruments}-TEST-{gps_start_time}-{duration}.gwf" \
    --description TEST

echo -e "\n=== Analyzing state vectors ==="
sgn-ligo-gwistat \
    --data-source frames \
    --frame-cache "$TEST_DIR/frames/*.gwf" \
    --channel-name L1:TEST-STATE_VECTOR \
    --mapping-file "$TEST_DIR/mapping.json" \
    --gps-start-time 1400000000 \
    --gps-end-time 1400000080 \
    --verbose

# Clean up
rm -rf "$TEST_DIR"
```

## Summary

GWIStat provides a powerful way to:
- Read state vector data from frame files or real-time streams
- Interpret bitmask values using customizable mappings
- Output structured JSON for further analysis
- Send results to Kafka for integration with monitoring systems

The tool is particularly useful for:
- Debugging detector state issues
- Monitoring detector status over time
- Creating alerts based on specific bit patterns
- Analyzing correlations between different state bits

For more information on the SGN framework and other tools, see the [SGN documentation](https://docs.ligo.org/greg/sgn/).