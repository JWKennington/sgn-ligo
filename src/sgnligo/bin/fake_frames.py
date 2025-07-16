"""Generate fake frame files with multiple channels for testing.

This tool creates GW frame files containing:
1. A state vector channel with bitmask data that changes over time segments
2. A strain channel with realistic colored noise based on LIGO/Virgo PSDs

The state vector uses custom segments from a file with three columns: start end value
where start and end are absolute GPS times in seconds.

This is useful for testing state-based gating and monitoring tools.
"""

# Copyright (C) 2025

from argparse import ArgumentParser

import numpy as np
from sgn.apps import Pipeline
from sgnts.sources import SegmentSource
from sgnts.transforms import Resampler

from sgnligo.base import now
from sgnligo.sinks import FrameSink
from sgnligo.sources import GWDataNoiseSource


def parse_command_line():
    parser = ArgumentParser(description=__doc__)

    parser.add_argument(
        "--state-channel",
        metavar="channel",
        default="L1:FAKE-STATE_VECTOR",
        help="State vector channel name (default: L1:FAKE-STATE_VECTOR)",
    )
    parser.add_argument(
        "--strain-channel",
        metavar="channel",
        default="L1:FAKE-STRAIN",
        help="Strain channel name (default: L1:FAKE-STRAIN)",
    )
    parser.add_argument(
        "--state-sample-rate",
        metavar="Hz",
        type=int,
        default=16,
        help="Sample rate for state vector channel in Hz (default: 16)",
    )
    parser.add_argument(
        "--strain-sample-rate",
        metavar="Hz",
        type=int,
        default=16384,
        help="Sample rate for strain channel in Hz (default: 16384)",
    )
    parser.add_argument(
        "--frame-duration",
        metavar="seconds",
        type=int,
        default=16,
        help="Duration of each frame file in seconds (default: 16)",
    )
    parser.add_argument(
        "--gps-start-time",
        metavar="seconds",
        type=float,
        help="GPS start time in seconds (if not provided, uses current GPS time)",
    )
    parser.add_argument(
        "--gps-end-time",
        metavar="seconds",
        type=float,
        help="GPS end time in seconds (if not provided, uses start + duration)",
    )
    parser.add_argument(
        "--duration",
        metavar="seconds",
        type=float,
        default=80,
        help="Total duration in seconds when using current GPS time (default: 80)",
    )
    parser.add_argument(
        "--output-path",
        metavar="path",
        default="{instruments}-{description}-{gps_start_time}-{duration}.gwf",
        help=(
            "Output path pattern for frame files (default: "
            "{instruments}-{description}-{gps_start_time}-{duration}.gwf)"
        ),
    )
    parser.add_argument(
        "--description",
        metavar="desc",
        default="BITMASK_TEST",
        help="Description for frame files (default: BITMASK_TEST)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Be verbose")
    parser.add_argument(
        "--segment-file",
        metavar="path",
        required=True,
        help=("Path to segment file with three columns: " "start end value (required)"),
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Enable real-time mode for continuous frame generation",
    )
    parser.add_argument(
        "--history",
        metavar="seconds",
        type=int,
        default=3600,
        help="How many seconds of history to keep in real-time mode (default: 3600)",
    )
    parser.add_argument(
        "--cleanup-interval",
        metavar="seconds",
        type=int,
        default=300,
        help="How often to check for old files to delete in seconds (default: 300)",
    )

    return parser.parse_args()


def read_segments_from_file(filename, verbose=False):
    """Read segments from a file with three columns: start end value.

    The start and end times in the file are absolute GPS times in seconds,
    which are converted to nanoseconds for internal use.

    Args:
        filename: Path to the segment file
        verbose: Whether to print verbose output

    Returns:
        tuple: (segments, values) where segments are absolute GPS time ranges
               in nanoseconds and values are the corresponding bitmask values
    """
    if verbose:
        print(f"Reading segments from {filename}")

    # Read the file
    data = np.loadtxt(filename)

    # Ensure we have 3 columns
    if data.ndim == 1:
        # Single row
        data = data.reshape(1, -1)

    if data.shape[1] != 3:
        raise ValueError(
            f"Segment file must have 3 columns (start end value), got {data.shape[1]}"
        )

    segments = []
    values = []

    for i, (start, end, value) in enumerate(data):
        # Convert times to nanoseconds
        start_ns = int(start * 1e9)
        end_ns = int(end * 1e9)

        segments.append((start_ns, end_ns))
        values.append(int(value))

        if verbose:
            print(f"Segment {i+1}: {start}s - {end}s, Value: {int(value)}")

    return tuple(segments), tuple(values)


def generate_fake_frames(options, segments, values):
    """Create and run the pipeline to generate fake frame files.

    Pipeline structure:

        SegmentSource (state vector)    GWDataNoiseSource (strain)
             |                                    |
             |                               Resampler
             |                                    |
             +------------------------------------+
                              |
                          FrameSink

    Args:
        options: Command line options
        segments: Segment time ranges in nanoseconds
        values: Bitmask values for segments
    """

    # Extract IFO from channel names and ensure consistency
    state_ifo = options.state_channel.split(":")[0]
    strain_ifo = options.strain_channel.split(":")[0]

    if state_ifo != strain_ifo:
        raise ValueError(
            f"IFO mismatch: state channel uses {state_ifo}, "
            f"strain channel uses {strain_ifo}"
        )

    ifo = state_ifo

    gps_start = options.gps_start_time
    gps_end = options.gps_end_time

    if options.verbose:
        print("\nCreating pipeline with:")
        print(f"  IFO: {ifo}")
        print(f"  State channel: {options.state_channel}")
        print(f"  Strain channel: {options.strain_channel}")
        print(f"  State sample rate: {options.state_sample_rate} Hz")
        print(f"  Strain sample rate: {options.strain_sample_rate} Hz")
        print(f"  Frame duration: {options.frame_duration} seconds")
        print(
            "  Time range: {} - {}".format(
                gps_start,
                gps_end if gps_end is not None else "indefinite",
            )
        )
        print(f"  Segments provided: {len(segments)}")
        if segments:
            seg_start = segments[0][0] / 1e9
            seg_end = segments[0][1] / 1e9
            print(f"  First segment: GPS {seg_start:.1f} - {seg_end:.1f}")

    pipeline = Pipeline()

    # Create the segment source with bitmask values
    # SegmentSource doesn't support None for end, use max GPS time
    # Maximum single precision signed integer (~year 2038)
    MAX_GPS_TIME = float(np.iinfo(np.int32).max)  # 2147483647
    segment_end = gps_end if gps_end is not None else MAX_GPS_TIME
    state_source = SegmentSource(
        name="StateSrc",
        source_pad_names=("state",),
        rate=options.state_sample_rate,
        t0=gps_start,
        end=segment_end,
        segments=segments,
        values=values,
    )

    # Create noise source for strain channel
    # GWDataNoiseSource generates at 16384 Hz based on the PSD
    noise_source = GWDataNoiseSource(
        name="NoiseSrc",
        channel_dict={ifo: options.strain_channel},
        t0=gps_start,
        end=gps_end,
        verbose=options.verbose,
        real_time=options.real_time if hasattr(options, "real_time") else False,
    )

    # Resample strain to match the requested sample rate if needed
    # GWDataNoiseSource always outputs at 16384 Hz based on PSDs
    # State and strain channels can have different sample rates in output frames
    if options.strain_sample_rate != 16384:
        resampler = Resampler(
            name="Resampler",
            source_pad_names=(options.strain_channel,),
            sink_pad_names=(options.strain_channel,),
            inrate=16384,  # GWDataNoiseSource outputs at this rate
            outrate=options.strain_sample_rate,
        )
        use_resampler = True
    else:
        resampler = None
        use_resampler = False

    # Create frame sink for both channels
    frame_sink = FrameSink(
        name="FrameSnk",
        channels=[options.state_channel, options.strain_channel],
        duration=options.frame_duration,
        path=options.output_path,
        description=options.description,
        force=True,
        history_seconds=(
            options.history
            if hasattr(options, "real_time")
            and options.real_time
            and hasattr(options, "history")
            else None
        ),
        cleanup_interval=options.cleanup_interval,
    )

    # Add elements to pipeline
    if use_resampler:
        pipeline.insert(state_source, noise_source, resampler, frame_sink)
    else:
        pipeline.insert(state_source, noise_source, frame_sink)

    # Connect the elements
    link_map = {
        f"FrameSnk:snk:{options.state_channel}": "StateSrc:src:state",
    }

    if use_resampler:
        link_map[f"Resampler:snk:{options.strain_channel}"] = (
            f"NoiseSrc:src:{options.strain_channel}"
        )
        link_map[f"FrameSnk:snk:{options.strain_channel}"] = (
            f"Resampler:src:{options.strain_channel}"
        )
    else:
        link_map[f"FrameSnk:snk:{options.strain_channel}"] = (
            f"NoiseSrc:src:{options.strain_channel}"
        )

    pipeline.insert(link_map=link_map)

    if options.verbose:
        print("\nRunning pipeline...")
        if gps_end is not None:
            print(f"Expected duration: {gps_end - gps_start} seconds")
            print(f"Frame duration: {options.frame_duration} seconds")
            expected_frames = (gps_end - gps_start) / options.frame_duration
            print(f"Expected number of frames: {expected_frames}")
        else:
            print("Real-time mode: will run indefinitely, synced with wall time")

    # Run the pipeline
    if options.verbose:
        print("Starting pipeline.run()...")

    pipeline.run()

    if options.verbose:
        print("Pipeline execution completed.")
        if gps_end is not None:
            expected_frames = (gps_end - gps_start) / options.frame_duration
            print(f"\nExpected number of frames: {int(expected_frames)}")
            print(f"Total duration: {gps_end - gps_start} seconds")


def main():
    options = parse_command_line()

    # Read segments from file first
    segments, values = read_segments_from_file(options.segment_file, options.verbose)

    if options.verbose:
        print(f"\nSegments loaded successfully: {len(segments)} segments")

    if options.real_time:
        # Real-time mode: start from current GPS time and run indefinitely
        if options.gps_start_time is None:
            options.gps_start_time = float(int(now()))
            if options.verbose:
                print(f"Real-time mode starting at GPS time: {options.gps_start_time}")

        # For real-time mode, we don't set an end time (run indefinitely)
        options.gps_end_time = None

        if options.verbose:
            print(f"History retention: {options.history} seconds")
            print("Running in real-time mode (press Ctrl+C to stop)...")
    else:
        # Normal batch mode
        if options.gps_start_time is None:
            # Use current GPS time
            options.gps_start_time = float(int(now()))
            if options.verbose:
                print(f"Using current GPS time: {options.gps_start_time}")

        if options.gps_end_time is None:
            # Calculate end time from duration
            options.gps_end_time = options.gps_start_time + options.duration
            if options.verbose:
                print(
                    f"Calculated end time: {options.gps_end_time} "
                    f"(duration: {options.duration}s)"
                )

    # Generate frames (same function for both real-time and batch modes)
    generate_fake_frames(options, segments, values)


if __name__ == "__main__":
    main()
