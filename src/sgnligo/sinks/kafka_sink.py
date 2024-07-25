from dataclasses import dataclass
from collections import deque, defaultdict

import numpy as np
from sgnts.base import TSSink

from ligo.scald.io import kafka


@dataclass
class KafkaSink(TSSink):
    """
    Push data to kafka

    Parameters:
    -----------
    output_kafka_server: str
        The kafka server to write data to
    topic: str
        The kafka topic to write data to
    metadata_key: str
        The dictionary key of the frame metadata to retrieve the data
    tags: str
        The tags to write the kafka data
    verbose: bool
        Be verbose
    reduce_time: float
        Will reduce data every reduce_time, in seconds
    """
    output_kafka_server: str = None
    topic: str = None
    metadata_key: str = None
    prepare_kafka_data: bool = False
    routes: str = None
    tags: list = None
    verbose: bool = False
    reduce_time: float = 2

    def __post_init__(self):
        assert isinstance(self.output_kafka_server, str)
        assert isinstance(self.topic, str)
        assert isinstance(self.metadata_key, str)
        super().__post_init__()

        self.cnt = {p: 0 for p in self.sink_pads}
        self.last_reduce_time = None

        self.client = kafka.Client("kafka://{}".format(self.output_kafka_server))
        #self.kafka_data = defaultdict(lambda: {'time': [], 'data': []})
        self.last_t0 = None

    def pull(self, pad, bufs):
        """
        getting the buffer on the pad just modifies the name to show this final
        graph point and the prints it to prove it all works.
        """
        self.cnt[pad] += 1
        bufst0 = bufs[0].t0/1_000_000_000
        metadata = bufs.metadata
        if self.last_t0 is None:
            self.last_t0 = bufst0

        if "kafka" in metadata:
            mkafka = metadata["kafka"]

            # append data to deque
            if self.metadata_key in mkafka:
                m = mkafka[self.metadata_key]
                #if 'time' in m:
                #self.kafka_data[self.metadata_key]['time'].append(m['time'])
                #self.kafka_data[self.metadata_key]['data'].append(m['data'])
                #else:
                #    self.kafka_data[self.route]['time'].append(bufst0)
                #    self.kafka_data[self.route]['data'].append(mkafka[self.metadata_key])

                # write to kafka
                #if bufst0 - self.last_t0 >= self.reduce_time:
                #self.client.write(self.topic, self.kafka_data[self.metadata_key], tags=self.tags)
                self.client.write(self.topic, m, tags=self.tags)
                #self.kafka_data[self.metadata_key]['time'] = []
                #self.kafka_data[self.metadata_key]['data'] = []

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


