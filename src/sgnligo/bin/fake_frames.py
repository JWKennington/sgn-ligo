"""Generate fake frame files containing strain and state vector data for testing."""

from argparse import ArgumentParser

import numpy as np
from sgn.apps import Pipeline
from sgnts.sources import SegmentSource
from sgnts.transforms import Resampler

from sgnligo.base import now, read_segments_and_values_from_file
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
        "--state-file",
        metavar="path",
        required=True,
        help=("Path to state file with three columns: " "start end value (required)"),
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


def generate_fake_frames(
    state_channel,
    strain_channel,
    state_sample_rate,
    strain_sample_rate,
    frame_duration,
    gps_start_time,
    gps_end_time,
    output_path,
    description,
    segments,
    values,
    verbose=False,
    real_time=False,
    history=None,
    cleanup_interval=300,
):
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
        state_channel: State vector channel name
        strain_channel: Strain channel name
        state_sample_rate: Sample rate for state vector channel in Hz
        strain_sample_rate: Sample rate for strain channel in Hz
        frame_duration: Duration of each frame file in seconds
        gps_start_time: GPS start time in seconds
        gps_end_time: GPS end time in seconds (None for real-time mode)
        output_path: Output path pattern for frame files
        description: Description for frame files
        segments: Segment time ranges in nanoseconds
        values: Bitmask values for segments
        verbose: Whether to print verbose output
        real_time: Enable real-time mode for continuous frame generation
        history: How many seconds of history to keep in real-time mode
        cleanup_interval: How often to check for old files to delete in seconds
    """

    # Extract IFO from channel names and ensure consistency
    state_ifo = state_channel.split(":")[0]
    strain_ifo = strain_channel.split(":")[0]

    if state_ifo != strain_ifo:
        raise ValueError(
            f"IFO mismatch: state channel uses {state_ifo}, "
            f"strain channel uses {strain_ifo}"
        )

    ifo = state_ifo

    gps_start = gps_start_time
    gps_end = gps_end_time

    if verbose:
        print("\nCreating pipeline with:")
        print(f"  IFO: {ifo}")
        print(f"  State channel: {state_channel}")
        print(f"  Strain channel: {strain_channel}")
        print(f"  State sample rate: {state_sample_rate} Hz")
        print(f"  Strain sample rate: {strain_sample_rate} Hz")
        print(f"  Frame duration: {frame_duration} seconds")
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
        rate=state_sample_rate,
        t0=gps_start,
        end=segment_end,
        segments=segments,
        values=values,
    )

    # Create noise source for strain channel
    # GWDataNoiseSource generates at 16384 Hz based on the PSD
    noise_source = GWDataNoiseSource(
        name="NoiseSrc",
        channel_dict={ifo: strain_channel},
        t0=gps_start,
        end=gps_end,
        verbose=verbose,
        real_time=real_time,
    )

    # Resample strain to match the requested sample rate if needed
    # GWDataNoiseSource always outputs at 16384 Hz based on PSDs
    # State and strain channels can have different sample rates in output frames
    if strain_sample_rate != 16384:
        resampler = Resampler(
            name="Resampler",
            source_pad_names=(strain_channel,),
            sink_pad_names=(strain_channel,),
            inrate=16384,  # GWDataNoiseSource outputs at this rate
            outrate=strain_sample_rate,
        )
        use_resampler = True
    else:
        resampler = None
        use_resampler = False

    # Create frame sink for both channels
    frame_sink = FrameSink(
        name="FrameSnk",
        channels=[state_channel, strain_channel],
        duration=frame_duration,
        path=output_path,
        description=description,
        force=True,
        max_files=history if real_time and history else None,
    )

    # Add elements to pipeline
    if use_resampler:
        pipeline.insert(state_source, noise_source, resampler, frame_sink)
    else:
        pipeline.insert(state_source, noise_source, frame_sink)

    # Connect the elements
    link_map = {
        f"FrameSnk:snk:{state_channel}": "StateSrc:src:state",
    }

    if use_resampler:
        link_map[f"Resampler:snk:{strain_channel}"] = f"NoiseSrc:src:{strain_channel}"
        link_map[f"FrameSnk:snk:{strain_channel}"] = f"Resampler:src:{strain_channel}"
    else:
        link_map[f"FrameSnk:snk:{strain_channel}"] = f"NoiseSrc:src:{strain_channel}"

    pipeline.insert(link_map=link_map)

    if verbose:
        print("\nRunning pipeline...")
        if gps_end is not None:
            print(f"Expected duration: {gps_end - gps_start} seconds")
            print(f"Frame duration: {frame_duration} seconds")
            expected_frames = (gps_end - gps_start) / frame_duration
            print(f"Expected number of frames: {expected_frames}")
        else:
            print("Real-time mode: will run indefinitely, synced with wall time")

    # Run the pipeline
    if verbose:
        print("Starting pipeline.run()...")

    pipeline.run()

    if verbose:
        print("Pipeline execution completed.")
        if gps_end is not None:
            expected_frames = (gps_end - gps_start) / frame_duration
            print(f"\nExpected number of frames: {int(expected_frames)}")
            print(f"Total duration: {gps_end - gps_start} seconds")


def main():
    options = parse_command_line()

    # Read segments from file first
    segments, values = read_segments_and_values_from_file(
        options.state_file, options.verbose
    )

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
    generate_fake_frames(
        state_channel=options.state_channel,
        strain_channel=options.strain_channel,
        state_sample_rate=options.state_sample_rate,
        strain_sample_rate=options.strain_sample_rate,
        frame_duration=options.frame_duration,
        gps_start_time=options.gps_start_time,
        gps_end_time=options.gps_end_time,
        output_path=options.output_path,
        description=options.description,
        segments=segments,
        values=values,
        verbose=options.verbose,
        real_time=options.real_time,
        history=options.history,
        cleanup_interval=options.cleanup_interval,
    )


if __name__ == "__main__":
    main()
