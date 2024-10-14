import h5py
import matplotlib.pyplot as plt
from scipy.signal import correlate

from sgnts.base import Audioadapter, Offset, TSSink
from .influx_sink import *
from .kafka_sink import *
from .strike_sink import StrikeSink
from ..base import ArrayOps


@dataclass
class ImpulseSink(TSSink):
    """
    A fake sink element
    """

    original_templates: str = None
    bank_index: int = None
    template_duration: float = None
    plotname: str = None
    impulse_pad: str = None
    data_pad: str = None
    verbose: bool = False
    bankno: int = None

    def __post_init__(self):
        super().__post_init__()
        self.cnt = {p: 0 for p in self.sink_pads}
        self.A = Audioadapter(lib=ArrayOps)
        self.Ainput = Audioadapter()

    def pull(self, pad, bufs):
        """
        getting the buffer on the pad just modifies the name to show this final
        graph point and the prints it to prove it all works.
        """
        # super().pull(pad, bufs)
        # bufs = self.preparedframes[pad]
        # FIXME: use preparedframes
        self.cnt[pad] += 1
        impulse_offset = bufs.metadata["impulse_offset"]
        self.impulse_offset = impulse_offset
        # if bufs.EOS:
        #    self.mark_eos(pad)
        padname = pad.name.split(":")[-1]
        if padname == self.data_pad:
            if bufs.buffers is not None:
                if self.verbose:
                    print(self.cnt[pad], bufs)
            for buf in bufs:
                if buf.end_offset > impulse_offset:
                    if buf.offset < impulse_offset + Offset.fromsec(
                        self.template_duration + 1
                    ):
                        # only save data around the impulse
                        buf.data = buf.data[self.bankno]
                        self.A.push(buf)
                    else:
                        recovered_impulse_offset, match = self.impulse_test()
                        print(f"{impulse_offset=} {recovered_impulse_offset=} {match=}")
                        if impulse_offset == recovered_impulse_offset and match > 0.997:
                            print("Impulse test passed")
                        else:
                            print("Impulse test failed")
                        self.mark_eos(pad)
                else:
                    # not at the impulse yet
                    pass
        elif padname == self.impulse_pad:
            for buf in bufs:
                if buf.offset > impulse_offset - Offset.fromsec(
                    1
                ) and buf.offset < impulse_offset + Offset.fromsec(
                    self.template_duration
                ):
                    # only save data around the impulse
                    self.Ainput.push(buf)

    @property
    def EOS(self):
        """
        If buffers on any sink pads are End of Stream (EOS), then mark this whole element as EOS
        """
        return any(self.at_eos.values())

    def impulse_test(self):
        """
        filter output against time reversed template
        """
        # TODO: find a way to determine bankno
        # subbanks = bank['subbanks']
        # print("srates", subbanks[bankno]["rates"])
        # if n == -1:
        #    n = subbanks[bankno]["ntemp"]
        # ntot = subbanks[bankno]["ntemp"]
        # print(f"Running impulse test for {n} templates out of {ntot} total...")
        # self.full_template_length = 2048 * (1+4+8)

        f1 = h5py.File(self.original_templates, "r")
        full_templates = np.array(f1["full_templates1"])
        f1.close()

        nfull_temp = full_templates.shape[0]
        # assert nfull_temp == subbanks[bankno]["ntemp"], (
        #    "Template number does not match, check the --bankno given"
        #    + " to make sure the same bank is being compared"
        # )
        # print(nfull_temp, ntot)
        n = nfull_temp
        if self.verbose:
            print(
                "number of templates",
                nfull_temp,
            )
        filter_output = self.A.copy_samples(self.A.size)
        if self.verbose:
            print("filter_output shape", filter_output.shape)
        # bankno=0
        # filter_output = filter_output[: nfull_temp // 2].cpu().numpy()[bankno]
        # filter_output = filter_output.cpu().numpy()[bankno]
        filter_output = filter_output.cpu().numpy()

        # only filter with current number of time slices
        # full_template_length =  2048 * 13
        # full_templates = full_templates[:, -full_template_length :]

        if self.verbose:
            print("full_templates.shape", full_templates.shape)
        outid = 0
        full_templates_flipped = np.flip(full_templates, 1)

        response = []

        cmaximumnormeds = np.zeros(shape=int(n / 2))
        istart = int((nfull_temp - n) / 4)
        iend = istart + int(n / 2)
        imiddle = int((istart + iend) / 2)
        # complex
        if self.verbose:
            print("Calculating response")
        for i, k in enumerate(range(istart, iend)):
            # template pairs
            real = full_templates_flipped[2 * k]
            imag = full_templates_flipped[2 * k + 1]

            h = np.array(real + imag * 1j)
            normh = np.linalg.norm(h)

            realo = filter_output[2 * k]
            imago = filter_output[2 * k + 1]
            o = np.array(realo + imago * 1j)
            normo = np.linalg.norm(o)

            # correlate
            response1 = np.abs(correlate(o, h, "valid")) / normh / normo
            response.append(response1)

            # find peak
            cmaximum = response1.max()
            cmaximumnormeds[i] = cmaximum
            if k == imiddle:
                outid = np.where(response1 == cmaximum)[0][0]

        cavgn = np.average(cmaximumnormeds)

        plotname = self.plotname
        if plotname is not None:
            # for plotting response
            if self.verbose:
                print("Plotting...")
            m = imiddle
            im = int(n / 4)
            # output = np.pad(filter_output[m].real, (self.impulse_position, 0), "constant")
            output = filter_output[m].real
            # res = np.pad(response[im], (self.impulse_position, 0), "constant")
            res = response[im]
            # indata = np.pad(
            #    torch.cat((self.indata)).cpu(), (self.impulse_position, 0), "constant"
            # )
            indata = self.Ainput.copy_samples(self.Ainput.size)
            # indata = np.zeros(2048)
            data = [indata, output, res] + [
                full_templates[m],
                full_templates_flipped[m],
            ]
            names = ["input", "output", "response"] + [
                "full\ntemplate",
                "full\ntemplate\nreversed",
            ]
            self.plot_wave(data, names, plotname, cmaximumnormeds, cavgn)

        return Offset.fromsamples(outid, 2048) + self.A.offset, cavgn

    # Plotting
    def plot_wave(self, data, dataname, figname, matchdata, cavgn):
        """
        Plot the input and output of the impulse response
        """
        plt.figure(figsize=(18, 10))
        n = len(data)
        maxlen = max(data, key=len)
        maxlen = len(maxlen)
        for i in range(n):
            plt.subplot(n, 1, i + 1)
            plt.plot(data[i])
            plt.xlim(0, maxlen)
            plt.ylabel(dataname[i])
            plt.tick_params(bottom=False, labelbottom=False)
        plt.tick_params(bottom=True, labelbottom=True)
        plt.tight_layout()
        plt.savefig(figname + "response")
        plt.clf()

        # Plot the match across templates
        plt.figure(figsize=(6, 4))
        plt.plot(matchdata, ".")
        plt.title("impulse response")
        plt.xlabel("template id")
        plt.ylabel("match")
        plt.axhline(cavgn, color="red", label=f"avg:\n{cavgn}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figname + "match")
        plt.clf()
