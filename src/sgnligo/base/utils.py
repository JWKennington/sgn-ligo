import lal
from lal import LIGOTimeGPS

import os

import time
import types

def now():
    """
    A convenience function to return the current gps time
    """
    return LIGOTimeGPS(lal.UTCToGPS(time.gmtime()), 0)

def from_T050017(url):
    """
    Parse a URL in the style of T050017-00.
    """
    filename, _ = os.path.splitext(url)
    obs, desc, start, dur = filename.split("-")
    return obs, desc, int(start), int(dur)

def state_vector_on_off_bits(bit):
    """
    Format the given bitmask appropriately as an integer
    """
    if isinstance(bit, str):
        if not bit.startswith("0b"):
            bit = "0b" + bit
        bit = int(bit, 2)
    else:
        bit = int(bit)

    return bit

def parse_list_to_dict(lst: list, value_transform: types.FunctionType = None, sep: str = '=', key_is_range: bool = False, range_sep: str = ':') -> dict:
    """A general list to dict argument parsing coercion function

    Args:
        lst:
            list, a list of the form ['A=V1', 'B=V2', ...], where "=" only has to match the sep_str argument
        value_transform:
            Function, default None. An optional transformation function to apply on values of the dictionary
        sep:
            str, default '=', the separator string between dict keys and values in list elements
        key_is_range:
            bool, default False. If True, the keys of the list are compound and contain range information e.g. "start:stop:remaining,list,of,items"
        range_sep:
            str, default ':' the separator string for range key information

    Returns:
        dict of the form {'A': value_transform('V1'), ...}

    Examples:
        >>> parse_list_to_dict(["H1=LSC-STRAIN", "H2=SOMETHING-ELSE"])  # doctest: +SKIP
        {'H1': 'LSC-STRAIN', 'H2': 'SOMETHING-ELSE'}

        >>> parse_list_to_dict(["0000:0002:H1=LSC_STRAIN_1,L1=LSC_STRAIN_2", "0002:0004:H1=LSC_STRAIN_3,L1=LSC_STRAIN_4", "0004:0006:H1=LSC_STRAIN_5,L1=LSC_STRAIN_6"], key_is_range=True)  # doctest: +SKIP
        {'0000': {'H1': 'LSC_STRAIN_1', 'L1': 'LSC_STRAIN_2'}, '0001': {'H1': 'LSC_STRAIN_1', 'L1': 'LSC_STRAIN_2'}, '0002': {'H1': 'LSC_STRAIN_3', 'L1': 'LSC_STRAIN_4'}, '0003': {'H1': 'LSC_STRAIN_3', 'L1': 'LSC_STRAIN_4'}, '0004': {'H1': 'LSC_STRAIN_5', 'L1': 'LSC_STRAIN_6'}, '0005': {'H1': 'LSC_STRAIN_5', 'L1': 'LSC_STRAIN_6'}}
    """
    if lst is None:
        return
    coerced = {}
    if key_is_range:
        # This will produce tuples (start, stop, str-to-dict)
        splits = [e.split(range_sep) for e in lst]
        for start, stop, val in splits:
            for i in range(int(start), int(stop)):
                key = str(i).zfill(4)
                coerced[key] = parse_list_to_dict([l.strip() for l in val.split(',')], value_transform=value_transform, sep=sep, key_is_range=False)
    else:
        if len(lst) == 1 and sep not in lst[0]:  # non-dict entry
            return lst[0]
        coerced = dict([e.split(sep) for e in lst])
        if value_transform is not None:
            for k in coerced:
                coerced[k] = value_transform(coerced[k])
        return coerced
    return coerced

