"""Read GW Frame data and interpret bitmask values."""

from __future__ import annotations

import glob
import json
import os
import tempfile
from argparse import ArgumentParser
from dataclasses import dataclass

from sgn.apps import Pipeline
from sgn.base import SourcePad
from sgnts.base import EventBuffer, EventFrame, Offset, TSTransform

from sgnligo.sinks import KafkaSink
from sgnligo.sources import DevShmSource
from sgnligo.sources.framecachesrc import FrameReader


@dataclass
class BitMaskInterpreter(TSTransform):
    """Interpret frame data as bitmask and map values to meanings.

    Outputs EventFrames compatible with KafkaSink.  This element has one sink
    pad and one source pad.

    Args:
        mapping_file:
            str, path to JSON file containing bitmask to meaning mappings.
            The JSON file can be in one of two formats:

            1. Simple format (backward compatible):
               {
                   "0": "BIT_0_MEANING",
                   "1": "BIT_1_MEANING",
                   ...
               }

            2. Extended format with both bit and value meanings:
               {
                   "bits": {
                       "0": "BIT_0_MEANING",
                       "1": "BIT_1_MEANING",
                       ...
                   },
                   "values": {
                       "0": "NO_DATA",
                       "3": "SPECIFIC_STATE",
                       "7": "ANOTHER_STATE",
                       ...
                   }
               }

            The extended format allows defining meanings for specific
            combinations of bits (composite values).
        kafka_topic:
            str, Kafka topic name for output (default: "gwistat")
        verbose:
            bool, be verbose
    """

    mapping_file: str | None = None
    kafka_topic: str = "gwistat"
    verbose: bool = False

    def __post_init__(self):
        super().__post_init__()

        assert self.mapping_file is not None
        assert len(self.sink_pads) == 1

        # Load the mapping from JSON file
        with open(self.mapping_file, "r") as f:
            mappings = json.load(f)

        # Support both old format (direct bit mappings) and new format
        if isinstance(mappings, dict) and ("bits" in mappings or "values" in mappings):
            # New format with separate bit and value mappings
            self.bit_mappings = mappings.get("bits", {})
            self.value_mappings = mappings.get("values", {})
        else:
            # Old format - assume it's just bit mappings
            self.bit_mappings = mappings
            self.value_mappings = {}

        if self.verbose:
            print(f"Loaded bit mappings: {self.bit_mappings}")
            if self.value_mappings:
                print(f"Loaded value mappings: {self.value_mappings}")

    def new(self, pad: SourcePad) -> EventFrame:
        """Interpret incoming data as bitmask and output EventFrame for KafkaSink.

        Args:
            pad:
                SourcePad, the source pad to produce EventFrames

        Returns:
            EventFrame, the EventFrame with interpreted bitmask data
        """
        frame = self.preparedframes[self.sink_pads[0]]

        # Collect all interpretations
        time_list = []
        data_list = []

        for buf in frame:
            if not buf.is_gap:
                # Process each data point
                for i, value in enumerate(buf.data):
                    # Calculate timestamp for this sample
                    # buf.offset is in offset units (samples at MAX_RATE)
                    # Convert to seconds using Offset.tosec(...)
                    sample_offset = buf.offset + Offset.fromsamples(i, buf.sample_rate)
                    sample_time = (
                        Offset.tosec(sample_offset) + Offset.offset_ref_t0 / 1e9
                    )

                    # Interpret the value as a bitmask
                    active_bits = []
                    bit_meanings = []
                    value_meaning = None

                    # Check each bit position
                    for bit_pos in range(32):  # Assuming 32-bit integers
                        bit_mask = 1 << bit_pos
                        if int(value) & bit_mask:
                            active_bits.append(bit_pos)
                            # Look up meaning if available
                            bit_key = str(bit_pos)
                            if bit_key in self.bit_mappings:
                                bit_meanings.append(self.bit_mappings[bit_key])

                    # Check if this specific value has a composite meaning
                    value_key = str(int(value))
                    if value_key in self.value_mappings:
                        value_meaning = self.value_mappings[value_key]

                    interpretation = {
                        "value": int(value),
                        "active_bits": active_bits,
                        "bit_meanings": bit_meanings,
                    }

                    # Add value meaning if available
                    if value_meaning:
                        interpretation["value_meaning"] = value_meaning

                    time_list.append(sample_time)
                    data_list.append(interpretation)

                    if (
                        self.verbose and len(time_list) <= 5
                    ):  # Only print first few for verbosity
                        if value_meaning:
                            print(
                                f"t={sample_time:.3f}: value={int(value)} -> "
                                f"{value_meaning} (bits: {bit_meanings})"
                            )
                        else:
                            print(
                                f"t={sample_time:.3f}: value={int(value)} -> "
                                f"{bit_meanings}"
                            )

        # Create EventFrame with kafka EventBuffer
        # Use the configured topic name (will be set from command line)
        kafka_data = {self.kafka_topic: {"time": time_list, "data": data_list}}

        # Get start and end times from the frame
        # frame.offset is the start time offset in samples at MAX_RATE
        # frame.end_offset is the end time offset in samples at MAX_RATE
        # EventBuffer expects nanosecond timestamps (integers)
        frame_start_ns = frame.offset + Offset.offset_ref_t0
        frame_end_ns = frame.end_offset + Offset.offset_ref_t0

        event_buffer = EventBuffer.from_span(
            frame_start_ns,
            frame_end_ns,
            [kafka_data],
        )

        return EventFrame(
            data=[event_buffer],
            EOS=frame.EOS,
        )


def parse_command_line():
    parser = ArgumentParser(description=__doc__)

    # Data source selection
    parser.add_argument(
        "--data-source",
        choices=["devshm", "frames", "gwdata-noise-realtime"],
        default="devshm",
        help=(
            "Data source type: devshm (real-time), frames (offline), or "
            "gwdata-noise-realtime (simulated real-time)"
        ),
    )

    # DevShmSource options
    parser.add_argument(
        "--shared-memory-dir",
        metavar="path",
        help="Path to shared memory directory (required for devshm source)",
    )

    # FrameReader options
    parser.add_argument(
        "--frame-cache",
        metavar="file",
        help="Frame cache file or pattern (required for frames source)",
    )
    parser.add_argument(
        "--channel-name",
        metavar="channel",
        required=True,
        help="Channel name to read (e.g., L1:GDS-CALIB_STATE_VECTOR)",
    )
    parser.add_argument(
        "--mapping-file",
        metavar="file",
        required=True,
        help="Path to JSON file containing bitmask mappings",
    )
    parser.add_argument(
        "--gps-start-time",
        metavar="seconds",
        type=float,
        help="GPS start time in seconds",
    )
    parser.add_argument(
        "--gps-end-time",
        metavar="seconds",
        type=float,
        help="GPS end time in seconds",
    )
    parser.add_argument(
        "--discont-wait-time",
        metavar="seconds",
        type=float,
        default=60,
        help="Time to wait before dropping data (default: 60 s)",
    )
    parser.add_argument(
        "--queue-timeout",
        metavar="seconds",
        type=float,
        default=1,
        help="Time to wait for next file from queue (default: 1 s)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Be verbose")

    # Kafka options
    parser.add_argument(
        "--output-kafka-server",
        metavar="addr",
        help=(
            "Kafka server address (e.g., localhost:9092). "
            "If not provided, pretty print to stdout"
        ),
    )
    parser.add_argument(
        "--kafka-topic",
        metavar="topic",
        default="gwistat",
        help="Kafka topic name (default: gwistat)",
    )
    parser.add_argument(
        "--kafka-tag",
        metavar="tag",
        help="Tag to include with Kafka messages (e.g., detector name)",
    )
    parser.add_argument(
        "--state-segments-file",
        metavar="file",
        help="Path to file with three columns (start end value) defining state vector "
        "segments for gwdata-noise-realtime source (optional)",
    )

    return parser.parse_args()


def bitmask_interpreter_pipeline(options):
    """Create and run the bitmask interpreter pipeline.

    Pipeline structure:

        DataSource (DevShmSource or FrameReader)
             |
        BitMaskInterpreter
             |
         NullSeriesSink
    """

    pipeline = Pipeline()

    # Create the appropriate data source
    if options.data_source == "devshm":
        if not options.shared_memory_dir:
            raise ValueError("--shared-memory-dir is required for devshm source")

        source = DevShmSource(
            name="DataSrc",
            shared_memory_dirs=options.shared_memory_dir,
            channel_names=[options.channel_name],
            discont_wait_time=options.discont_wait_time,
            queue_timeout=options.queue_timeout,
            verbose=options.verbose,
        )
    elif options.data_source == "gwdata-noise-realtime":
        # Use SegmentSource to generate simulated state vector data
        # State vectors are integer bitmask values sampled at 16 Hz
        import numpy as np
        from sgnts.sources import SegmentSource

        # Read state segments from file if provided, otherwise use default
        if options.state_segments_file is not None:
            from sgnligo.base import read_segments_and_values_from_file

            state_segments, state_values = read_segments_and_values_from_file(
                options.state_segments_file, options.verbose
            )
        else:
            # Default state segments: single segment with value 3 (bits 0 and 1 set)
            # Simulates HOFT_OK + OBS_INTENT - typical observing state
            if options.gps_start_time is not None:
                start_ns = int(options.gps_start_time * 1e9)
                if options.gps_end_time is not None:
                    end_ns = int(options.gps_end_time * 1e9)
                else:
                    # For real-time mode without end time, use max int32
                    end_ns = int(np.iinfo(np.int32).max * 1e9)
                    options.gps_end_time = float(np.iinfo(np.int32).max)
            else:
                from sgnligo.base import now

                start_time = float(int(now()))
                start_ns = int(start_time * 1e9)
                # Must provide end time when gps_start_time is None
                if options.gps_end_time is None:
                    raise ValueError(
                        "--gps-end-time is required when --gps-start-time is not "
                        "specified"
                    )
                end_ns = int(options.gps_end_time * 1e9)
                options.gps_start_time = start_time

            state_segments = ((start_ns, end_ns),)
            state_values = (3,)  # Bits 0 and 1 set: HOFT_OK + OBS_INTENT

        # SegmentSource doesn't support None for end time
        seg_end = (
            options.gps_end_time
            if options.gps_end_time is not None
            else float(np.iinfo(np.int32).max)
        )

        # Use channel name as the pad name for consistency with DevShmSource/FrameReader
        source = SegmentSource(
            name="DataSrc",
            source_pad_names=(options.channel_name,),
            rate=16,  # 16 Hz is typical for state vectors
            t0=options.gps_start_time,
            end=seg_end,
            segments=state_segments,
            values=state_values,
        )

        if options.verbose:
            print(f"Created simulated state vector source for {options.channel_name}")
            print(f"  Time range: {options.gps_start_time} - {options.gps_end_time}")
            print(f"  State segments: {state_segments}")
            print(f"  State values: {state_values}")
    else:  # frames
        if not options.frame_cache:
            raise ValueError("--frame-cache is required for frames source")

        # Handle glob patterns or create cache file

        if "*" in options.frame_cache or options.frame_cache.endswith(".gwf"):
            # It's a glob pattern or direct gwf file
            if "*" in options.frame_cache:
                frame_files = sorted(glob.glob(options.frame_cache))
            else:
                # Single gwf file
                frame_files = [options.frame_cache]

            if not frame_files:
                raise ValueError(f"No files found matching: {options.frame_cache}")

            # Create a temporary cache file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".cache", delete=False
            ) as f:
                for frame_file in frame_files:
                    # LAL cache format:
                    # observatory frametype start_time duration file_path
                    # Extract info from filename (assumes standard naming)
                    basename = os.path.basename(frame_file)
                    parts = basename.split("-")
                    if len(parts) == 4:
                        obs = parts[0]
                        frametype = parts[1]
                        start_time = parts[2]
                        duration = parts[3].split(".")[0]
                    else:
                        raise ValueError(
                            f"Could not parse {basename} into LAL Cache convention"
                        )
                    abs_path = os.path.abspath(frame_file)
                    cache_line = (
                        f"{obs} {frametype} {start_time} {duration} "
                        f"file://localhost{abs_path}\n"
                    )
                    f.write(cache_line)

                cache_file = f.name
                if options.verbose:
                    print(f"Created temporary cache file: {cache_file}")
        else:
            # It's already a cache file
            cache_file = options.frame_cache

        # Extract IFO from channel name
        ifo = options.channel_name.split(":")[0]

        # Create FrameReader with proper parameters
        source = FrameReader(
            name="DataSrc",
            framecache=cache_file,
            channel_names=[options.channel_name],
            instrument=ifo,
            t0=options.gps_start_time if options.gps_start_time else None,
            end=options.gps_end_time if options.gps_end_time else None,
        )

    # Add source and rest of pipeline
    pipeline.insert(
        source,
        BitMaskInterpreter(
            name="BitMaskInt",
            mapping_file=options.mapping_file,
            kafka_topic=options.kafka_topic,
            verbose=options.verbose,
            source_pad_names=("output",),
            sink_pad_names=("input",),
        ),
        KafkaSink(
            name="KafkaSink",
            sink_pad_names=("data",),
            output_kafka_server=options.output_kafka_server,
            time_series_topics=[options.kafka_topic],
            tag=[options.kafka_tag] if options.kafka_tag else [],
            prefix="gwistat.",
        ),
    )

    # Connect the elements
    pipeline.insert(
        link_map={
            "BitMaskInt:snk:input": f"DataSrc:src:{options.channel_name}",
            "KafkaSink:snk:data": "BitMaskInt:src:output",
        }
    )

    # Run the pipeline
    pipeline.run()


def main():
    options = parse_command_line()
    bitmask_interpreter_pipeline(options)


if __name__ == "__main__":
    main()
