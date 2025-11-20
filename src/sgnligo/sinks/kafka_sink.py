"""A sink element to send data to kafka topics."""

# Copyright (C) 2024 Yun-Jing Huang

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Optional

from lal import LIGOTimeGPS
from ligo.scald.io import kafka
from sgn.base import SinkElement

from sgnligo.base import now


class LIGOJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles LIGO-specific types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, LIGOTimeGPS):
            # Convert LIGOTimeGPS to float (GPS seconds)
            return float(obj)
        # Let the base class handle other types
        return super().default(obj)


@dataclass
class KafkaSink(SinkElement):
    """Send data to kafka topics or pretty print to stdout

    Args:
        output_kafka_server:
            str or None, The kafka server to write data to. If None, pretty
            print to stdout
        time_series_topics:
            list[str], The kafka topics to write time-series data to
        trigger_topics:
            list[str], The kafka topics to write trigger data to
        tag:
            str, The tag to write the kafka data with
        prefix:
            str, The prefix of the kafka topic
        interval:
            int, The interval at which to write the data to kafka
    """

    output_kafka_server: Optional[str] = None
    time_series_topics: Optional[list[str]] = None
    trigger_topics: Optional[list[str]] = None
    tag: Optional[list[str]] = None
    prefix: str = ""
    interval: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()

        # Initialize client only if kafka server is provided
        # Handle both None and the string "None"
        if self.output_kafka_server is not None and self.output_kafka_server != "None":
            self.client = kafka.Client("kafka://{}".format(self.output_kafka_server))
        else:
            self.client = None

        if self.tag is None:
            self.tag = []

        if self.time_series_topics is not None:
            self.time_series_data = {}
            for topic in self.time_series_topics:
                self.time_series_data[topic] = {"time": [], "data": []}
        else:
            self.time_series_data = None

        if self.trigger_topics is not None:
            self.trigger_data = {}
            for topic in self.trigger_topics:
                self.trigger_data[topic] = []
        else:
            self.trigger_data = None

        self.last_sent = now()

    def _pretty_print(self, topic, data, data_type="time_series"):
        """Pretty print data to stdout in a formatted way."""
        output = {
            "topic": self.prefix + topic,
            "tags": self.tag,
            "data_type": data_type,
            "timestamp": now(),
            "data": data,
        }
        print(json.dumps(output, indent=2, cls=LIGOJSONEncoder))
        sys.stdout.flush()

    def write(self):
        if self.time_series_data is not None:
            for topic, data in self.time_series_data.items():
                if len(data["time"]) > 0:
                    if self.client is not None:
                        self.client.write(self.prefix + topic, data, tags=self.tag)
                    else:
                        self._pretty_print(topic, data, "time_series")
                    self.time_series_data[topic] = {"time": [], "data": []}

        if self.trigger_data is not None:
            for topic, data in self.trigger_data.items():
                if len(data) > 0:
                    if self.client is not None:
                        self.client.write(self.prefix + topic, data, tags=self.tag)
                    else:
                        self._pretty_print(topic, data, "trigger")
                    self.trigger_data[topic] = []

    def pull(self, pad, frame):
        """Incoming frames are expected to be an EventFrame containing EventBuffers.
        The data in the EventBuffer are expected to in the format of
        {topic: {"time": [t1, t2, ...], "data": [d1, d2, ...]}}
        """
        for event in frame.events:
            if event is not None:
                if isinstance(event, dict):
                    for topic, data in event.items():
                        if (
                            self.time_series_topics is not None
                            and topic in self.time_series_topics
                        ):
                            self.time_series_data[topic]["time"].extend(data["time"])
                            self.time_series_data[topic]["data"].extend(data["data"])
                        elif (
                            self.trigger_topics is not None
                            and topic in self.trigger_topics
                        ):
                            self.trigger_data[topic].extend(data)
                        else:
                            raise ValueError("Unknwon topic")
                else:
                    raise ValueError("Unknown data type ")

        if frame.EOS:
            self.mark_eos(pad)

    def internal(self):
        if self.interval is None:
            # Don't wait
            self.write()
        else:
            time_now = now()
            if time_now - self.last_sent > self.interval:
                self.write()
                self.last_sent = time_now

        if self.at_eos:
            print("shutdown: KafkaSink: close", file=sys.stderr)
            if self.client is not None:
                self.client.close()
