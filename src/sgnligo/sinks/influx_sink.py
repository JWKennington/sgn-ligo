from dataclasses import dataclass
from collections import deque, defaultdict
import yaml
from ligo.scald.io import influx

import numpy as np
from sgnts.base import TSSink

@dataclass
class InfluxSink(TSSink):
    """
    Push data to influx
    """
    instrument: str = None
    metadata_key: str = None
    scald_config: str = None
    route: str = None
    verbose: bool = False
    wait_time: float = 2

    def __post_init__(self):
        super().__post_init__()

        self.cnt = {p: 0 for p in self.sink_pads}
        self.last_reduce_time = None

        self.last_t0 = None
        # set up aggregator sink
        with open(self.scald_config, 'r') as f:
            agg_config = yaml.safe_load(f)
        self.agg_sink = influx.Aggregator(**agg_config["backends"]["default"])

        # register measurement schemas for aggregators
        self.agg_sink.load(path=self.scald_config)
        self.timedeq = deque(maxlen = 100)
        self.datadeq = {self.route: deque(maxlen = 100)}

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
            # FIXME: only works when data are integers?? if float, I get `urllib3 response status: 400 | response reason: Bad Request`
            self.timedeq.append(int(bufs[0].t0/1_000_000_000))
            self.datadeq[self.route].append(int(bufs.metadata[self.metadata_key]))

        data = {self.route: {self.instrument: {'time': list(self.timedeq), 'fields': {'data': list(self.datadeq[self.route])}}}}
        if bufs[0].t0 - self.last_t0 >= int(self.wait_time*1_000_000_000):
            self.last_t0 = round(int(bufs[0].t0), -2)

            print('Writing out to influx')
            self.agg_sink.store_columns(self.route, data[self.route], aggregate="max")
            self.timedeq.clear()
            self.datadeq[self.route].clear()
        
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


