from dataclasses import dataclass

from ligo.scald.io import kafka

from sgnts.base import TSSink


@dataclass
class KafkaSink(TSSink):
    """
    Push data to kafka

    Parameters:
    -----------
    output_kafka_server: str
        The kafka server to write data to
    topics: str
        The kafka topics to write data to
    tags: str
        The tags to write the kafka data
    verbose: bool
        Be verbose
    reduce_time: float
        Will reduce data every reduce_time, in seconds
    """

    output_kafka_server: str = None
    topics: str = None
    prepare_kafka_data: bool = False
    routes: str = None
    tags: list = None
    verbose: bool = False
    reduce_time: float = 2

    def __post_init__(self):
        assert isinstance(self.output_kafka_server, str)
        assert isinstance(self.topics, list)
        super().__post_init__()

        self.cnt = {p: 0 for p in self.sink_pads}
        self.last_reduce_time = None

        self.client = kafka.Client("kafka://{}".format(self.output_kafka_server))
        # self.kafka_data = defaultdict(lambda: {'time': [], 'data': []})
        self.last_t0 = None

    def pull(self, pad, bufs):
        """
        getting the buffer on the pad just modifies the name to show this final
        graph point and the prints it to prove it all works.
        """
        self.cnt[pad] += 1
        bufst0 = bufs[0].t0 / 1_000_000_000
        metadata = bufs.metadata
        if self.last_t0 is None:
            self.last_t0 = bufst0

        if "kafka" in metadata:
            mkafka = metadata["kafka"]

            # append data to deque
            for topic in self.topics:
                t = topic.split(".")[-1]
                if t in mkafka:
                    self.client.write(topic, mkafka[t], tags=self.tags)

            self.last_t0 = bufst0

        if bufs.EOS:
            self.mark_eos(pad)

        if self.verbose is True:
            print(self.cnt[pad], bufs)

    @property
    def EOS(self):
        """
        If buffers on any sink pads are End of Stream (EOS), then mark this whole element as EOS
        """
        return any(self.at_eos.values())
