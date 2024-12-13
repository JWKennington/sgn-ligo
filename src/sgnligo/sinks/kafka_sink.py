from dataclasses import dataclass

from ligo.scald.io import kafka
from sgnts.base import SinkElement


@dataclass
class KafkaSink(SinkElement):
    """Send data to kafka topics

    Args:
        output_kafka_server:
            str, The kafka server to write data to
        topics:
            list[str], The kafka topics to write data to
        tag:
            str, The tag to write the kafka data
    """

    output_kafka_server: str = None
    topics: list[str] = None
    tag: list[str] = None

    def __post_init__(self):
        assert isinstance(self.output_kafka_server, str)
        assert isinstance(self.topics, list)
        super().__post_init__()

        self.client = kafka.Client("kafka://{}".format(self.output_kafka_server))
        if self.tag is None:
            self.tag = []

    def pull(self, pad, frame):
        """Incoming frames are expected to be an EventFrame containing {"kafka":
        EventBuffer}. The data in the EventBuffer are expected to in the format of
        {topic: {"time": [t1, t2, ...], "data": [d1, d2, ...]}}
        """
        events = frame["kafka"].data
        # append data to deque
        for topic in self.topics:
            t = topic.split(".")[-1]
            if events is not None and t in events:
                self.client.write(topic, events[t], tags=self.tag)

        if frame.EOS:
            self.mark_eos(pad)

    def internal(self):
        if self.at_eos:
            self.client.close()
