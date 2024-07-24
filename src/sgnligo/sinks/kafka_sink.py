from dataclasses import dataclass
from collections import deque, defaultdict

import numpy as np
from sgnts.base import TSSink

from ligo.scald.io import kafka


@dataclass
class KafkaSink(TSSink):
    """
    Push data to kafka
    """
    output_kafka_server: str = None
    topic: str = None
    metadata_key: str = None
    route: str = None
    tags: list = None
    verbose: bool = False
    wait_time: float = 2

    def __post_init__(self):
        assert isinstance(self.output_kafka_server, str)
        assert isinstance(self.topic, str)
        assert isinstance(self.metadata_key, str)
        super().__post_init__()

        self.cnt = {p: 0 for p in self.sink_pads}
        self.last_reduce_time = None

        self.client = kafka.Client("kafka://{}".format(self.output_kafka_server))
        self.kafka_data = defaultdict(lambda: {'time': [], 'data': []})
        self.last_t0 = None

    def pull(self, pad, bufs):
        """
        getting the buffer on the pad just modifies the name to show this final
        graph point and the prints it to prove it all works.
        """
        #super().pull(pad, bufs)
        #bufs = self.preparedframes[pad]
        self.cnt[pad] += 1
        if self.last_t0 is None:
            self.last_t0 = bufs[0].t0

        if self.metadata_key in bufs.metadata:
            self.kafka_data[self.route]['time'].append(bufs[0].t0)
            self.kafka_data[self.route]['data'].append(bufs.metadata[self.metadata_key])

        if bufs[0].t0 - self.last_t0 >= int(self.wait_time*1_000_000_000):
            print('Writing out to kafka')

            self.client.write(self.topic, self.kafka_data, )
            self.kafka_data[self.route]['time'] = []
            self.kafka_data[self.route]['data'] = []

            self.last_t0 = bufs[0].t0

        
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


