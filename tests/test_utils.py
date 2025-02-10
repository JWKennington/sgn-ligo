#!/usr/bin/env python3

from sgnligo.base import utils


def test_utils():
    utils.now()
    utils.from_T050017("file:///H1-TEST-0-1.tst")
    utils.state_vector_on_off_bits("01")
    utils.state_vector_on_off_bits("0b01")
    utils.state_vector_on_off_bits(1)
    utils.parse_list_to_dict(None)
    utils.parse_list_to_dict(["blah"])
    utils.parse_list_to_dict(
        ["H1=LSC-STRAIN", "H2=SOMETHING-ELSE"], value_transform=str
    )
    utils.parse_list_to_dict(
        [
            "0000:0002:H1=LSC_STRAIN_1,L1=LSC_STRAIN_2",
            "0002:0004:H1=LSC_STRAIN_3,L1=LSC_STRAIN_4",
            "0004:0006:H1=LSC_STRAIN_5,L1=LSC_STRAIN_6",
        ],
        key_is_range=True,
    )


if __name__ == "__main__":
    test_utils()
