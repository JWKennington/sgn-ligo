"""A sink element to send data to kafka topics."""

# Copyright (C) 2024 Yun-Jing Huang

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ligo.scald.io import kafka
from sgn.base import SinkElement

from sgnligo.base import now


@dataclass
class KafkaSink(SinkElement):
    """Send data to kafka topics

    Args:
        output_kafka_server:
            str, The kafka server to write data to
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
        assert isinstance(self.output_kafka_server, str)
        super().__post_init__()

        self.client = kafka.Client("kafka://{}".format(self.output_kafka_server))
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

    def write(self):
        if self.time_series_data is not None:
            for topic, data in self.time_series_data.items():
                if len(data["time"]) > 0:
                    self.client.write(self.prefix + topic, data, tags=self.tag)
                    self.time_series_data[topic] = {"time": [], "data": []}

        if self.trigger_data is not None:
            for topic, data in self.trigger_data.items():
                if len(data) > 0:
                    self.client.write(self.prefix + topic, data, tags=self.tag)
                    self.trigger_data[topic] = []

    def pull(self, pad, frame):
        """Incoming frames are expected to be an EventFrame containing {"kafka":
        EventBuffer}. The data in the EventBuffer are expected to in the format of
        {topic: {"time": [t1, t2, ...], "data": [d1, d2, ...]}}
        """
        events = frame["kafka"].data
        if events is not None:
            for topic, data in events.items():
                if (
                    self.time_series_topics is not None
                    and topic in self.time_series_topics
                ):
                    self.time_series_data[topic]["time"].extend(data["time"])
                    self.time_series_data[topic]["data"].extend(data["data"])
                elif self.trigger_topics is not None and topic in self.trigger_topics:
                    self.trigger_data[topic].extend(data)

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
            print("shutdown: KafkaSink: close")
            self.client.close()
