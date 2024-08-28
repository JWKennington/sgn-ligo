from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import copy
import numpy
import os
import sys
import warnings
import scipy
import tempfile

import lal

from ligo.lw import ligolw
from ligo.lw import lsctables
from ligo.lw import array as ligolw_array
from ligo.lw import param as ligolw_param
from ligo.lw import utils as ligolw_utils
from ligo.lw.utils import process as ligolw_process

from .psd import condition_psd, HorizonDistance

Attributes = ligolw.sax.xmlreader.AttributesImpl


class DefaultContentHandler(ligolw.LIGOLWContentHandler):
    pass


ligolw_array.use_in(DefaultContentHandler)
ligolw_param.use_in(DefaultContentHandler)
lsctables.use_in(DefaultContentHandler)


@dataclass
class BankFragment:
    rate: int
    start: float
    end: float
    # orthogonal_template_bank: Sequence[Any]
    # singular_values: Sequence[Any]
    # mix_matrix: Sequence[Any]
    # chifacs: Sequence[Any]
    # sum_of_squares_weights: Sequence[Any]


@dataclass
class Bank:
    sngl_inspiral_table: Sequence[Any]
    filter_length: float
    gate_threshold: float
    logname: str
    snr_threshold: float
    template_bank_filename: str
    bank_id: str
    bank_type: str
    autocorrelation_bank: Sequence[Any]
    autocorrelation_mask: Sequence[Any]
    sigmasq: Sequence[Any]
    bank_correlation_matrix: Sequence[Any]
    horizon_factors: dict[int, float]
    processed_psd: Sequence[Any]
    bank_fragments: list[Any]

    def get_rates(self):
        return set(bank_fragment.rate for bank_fragment in self.bank_fragments)


def preferred_horizon_distance_template(banks):
    template_id, m1, m2, s1z, s2z = min(
        (row.template_id, row.mass1, row.mass2, row.spin1z, row.spin2z)
        for bank in banks
        for row in bank.sngl_inspiral_table
    )
    return template_id, m1, m2, s1z, s2z


def horizon_distance_func(banks):
    """
    Takes a dictionary of objects returned by read_banks keyed by instrument
    """
    # span is [15 Hz, 0.85 * Nyquist frequency]
    # find the Nyquist frequency for the PSD to be used for each
    # instrument.  require them to all match
    nyquists = set((max(bank.get_rates()) / 2.0 for bank in banks))
    if len(nyquists) != 1:
        warnings.warn(
            "all banks should have the same Nyquist frequency to define a consistent horizon distance function (got %s)"
            % ", ".join("%g" % rate for rate in sorted(nyquists))
        )
    # assume default 4 s PSD.  this is not required to be correct, but
    # for best accuracy it should not be larger than the true value and
    # for best performance it should not be smaller than the true
    # value.
    deltaF = 1.0 / 4.0
    # use the minimum template id as the cannonical horizon function
    template_id, m1, m2, s1z, s2z = preferred_horizon_distance_template(banks)

    return template_id, HorizonDistance(
        15.0,
        0.85 * max(nyquists),
        deltaF,
        m1,
        m2,
        spin1=(0.0, 0.0, s1z),
        spin2=(0.0, 0.0, s2z),
    )


def read_banks(filename, contenthandler, verbose=False):
    """Read SVD banks from a LIGO_LW xml file."""

    # Load document
    xmldoc = ligolw_utils.load_url(
        filename, contenthandler=contenthandler, verbose=verbose
    )

    banks = []

    # FIXME in principle this could be different for each bank included in
    # this file, but we only put one in the file for now
    # FIXME, right now there is only one instrument so we just pull out the
    # only psd there is
    try:
        raw_psd = list(lal.series.read_psd_xmldoc(xmldoc).values())[0]
    except ValueError:
        # the bank file does not contain psd ligolw element.
        raw_psd = None

    for root in (
        elem
        for elem in xmldoc.getElementsByTagName(ligolw.LIGO_LW.tagName)
        if elem.hasAttribute("Name") and elem.Name == "gstlal_svd_bank_Bank"
    ):

        # Create new SVD bank object
        bank = Bank.__new__(Bank)

        # Read sngl inspiral table
        bank.sngl_inspiral_table = lsctables.SnglInspiralTable.get_table(root)
        bank.sngl_inspiral_table.parentNode.removeChild(bank.sngl_inspiral_table)

        # Read root-level scalar parameters
        bank.filter_length = ligolw_param.get_pyvalue(root, "filter_length")
        bank.gate_threshold = ligolw_param.get_pyvalue(root, "gate_threshold")
        bank.logname = ligolw_param.get_pyvalue(root, "logname") or None
        bank.snr_threshold = ligolw_param.get_pyvalue(root, "snr_threshold")
        bank.template_bank_filename = ligolw_param.get_pyvalue(
            root, "template_bank_filename"
        )
        bank.bank_id = ligolw_param.get_pyvalue(root, "bank_id")
        bank.bank_type = ligolw_param.get_pyvalue(root, "bank_type")

        try:
            bank.newdeltaF = ligolw_param.get_pyvalue(root, "new_deltaf")
            bank.working_f_low = ligolw_param.get_pyvalue(root, "working_f_low")
            bank.f_low = ligolw_param.get_pyvalue(root, "f_low")
            bank.sample_rate_max = ligolw_param.get_pyvalue(root, "sample_rate_max")
        except ValueError:
            pass

        # Read root-level arrays
        bank.autocorrelation_bank = (
            ligolw_array.get_array(root, "autocorrelation_bank_real").array
            + 1j * ligolw_array.get_array(root, "autocorrelation_bank_imag").array
        )
        bank.autocorrelation_mask = ligolw_array.get_array(
            root, "autocorrelation_mask"
        ).array
        bank.sigmasq = ligolw_array.get_array(root, "sigmasq").array
        bank_correlation_real = ligolw_array.get_array(
            root, "bank_correlation_matrix_real"
        ).array
        bank_correlation_imag = ligolw_array.get_array(
            root, "bank_correlation_matrix_imag"
        ).array
        bank.bank_correlation_matrix = (
            bank_correlation_real + 1j * bank_correlation_imag
        )

        # prepare the horizon distance factors
        bank.horizon_factors = dict(
            (row.template_id, sigmasq**0.5)
            for row, sigmasq in zip(bank.sngl_inspiral_table, bank.sigmasq)
        )

        if raw_psd is not None:
            # reproduce the whitening psd and attach a reference to the psd
            bank.processed_psd = condition_psd(
                raw_psd,
                bank.newdeltaF,
                minfs=(bank.working_f_low, bank.f_low),
                maxfs=(bank.sample_rate_max / 2.0 * 0.90, bank.sample_rate_max / 2.0),
            )
        else:
            bank.processed_psd = None

        # Read bank fragments
        bank.bank_fragments = []
        for el in (
            node for node in root.childNodes if node.tagName == ligolw.LIGO_LW.tagName
        ):
            frag = BankFragment(
                rate=ligolw_param.get_pyvalue(el, "rate"),
                start=ligolw_param.get_pyvalue(el, "start"),
                end=ligolw_param.get_pyvalue(el, "end"),
            )

            # Read arrays
            frag.chifacs = ligolw_array.get_array(el, "chifacs").array
            try:
                frag.mix_matrix = ligolw_array.get_array(el, "mix_matrix").array
            except ValueError:
                frag.mix_matrix = None
            frag.orthogonal_template_bank = ligolw_array.get_array(
                el, "orthogonal_template_bank"
            ).array
            try:
                frag.singular_values = ligolw_array.get_array(
                    el, "singular_values"
                ).array
            except ValueError:
                frag.singular_values = None
            try:
                frag.sum_of_squares_weights = ligolw_array.get_array(
                    el, "sum_of_squares_weights"
                ).array
            except ValueError:
                frag.sum_of_squares_weights = None
            bank.bank_fragments.append(frag)

        banks.append(bank)
    template_id, func = horizon_distance_func(banks)
    template_id = abs(
        template_id
    )  # make sure horizon_distance_func did not pick the noise model template
    horizon_norm = None
    for bank in banks:
        if template_id in bank.horizon_factors and bank.bank_type == "signal_model":
            assert horizon_norm is None
            horizon_norm = bank.horizon_factors[template_id]
    for bank in banks:
        bank.horizon_distance_func = func
        bank.horizon_factors = dict(
            (tid, f / horizon_norm) for (tid, f) in bank.horizon_factors.items()
        )
    xmldoc.unlink()
    return banks


def parse_bank_files(svd_banks, verbose, snr_threshold=None):
    """
    given a dictionary of lists of svd template bank file names parse them
    into a dictionary of bank classes
    """

    banks = {}

    for instrument, filename in svd_banks.items():
        for n, bank in enumerate(
            read_banks(filename, contenthandler=DefaultContentHandler, verbose=verbose)
        ):
            # Write out sngl inspiral table to temp file for
            # trigger generator
            # FIXME teach the trigger generator to get this
            # information a better way
            bank.template_bank_filename = tempfile.NamedTemporaryFile(
                suffix=".xml.gz", delete=False
            ).name
            xmldoc = ligolw.Document()
            # FIXME if this table reference is from a DB this
            # is a problem (but it almost certainly isn't)
            xmldoc.appendChild(ligolw.LIGO_LW()).appendChild(
                bank.sngl_inspiral_table.copy()
            ).extend(bank.sngl_inspiral_table)
            ligolw_utils.write_filename(
                xmldoc, bank.template_bank_filename, verbose=verbose
            )
            xmldoc.unlink()  # help garbage collector
            bank.logname = "%sbank%d" % (instrument, n)
            banks.setdefault(instrument, []).append(bank)
            if snr_threshold is not None:
                bank.snr_threshold = snr_threshold

    # FIXME remove when this is no longer an issue
    if not banks:
        raise ValueError(
            "Could not parse bank files into valid bank dictionary.\n\t- Perhaps you are using out-of-date svd bank files?  Please ensure that they were generated with the same code version as the parsing code"
        )
    return banks
