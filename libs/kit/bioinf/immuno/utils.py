"""Implements some utility functions for working with immuno data"""

import re

from kit.hashes import str_to_hash


HLA_REGEX = r"HLA-([ABC])\*?(\d{1,3}):(\d{1,3})"


def get_immuno_setup_hash(immuno_setup):
    mhc_1_setup = immuno_setup["mhc_1"]
    return get_mhc_1_setup_hash(mhc_1_setup)


def get_mhc_1_setup_hash(mhc_1_setup):
    if isinstance(mhc_1_setup, list):
        _mhc_1_alleles = sorted(mhc_1_setup)
    else:
        _mhc_1_alleles = sorted([_allele for _allele in mhc_1_setup.split('+')])
    return str_to_hash('+'.join(_mhc_1_alleles), 6)


def get_predictor_setup_hash(predictor_setup):
    mhc_1_predictor = predictor_setup["mhc_1"]
    return mhc_1_predictor.get_predictor_hash()


def get_mhc_class(allele_name):
    """Returns the MHC class of an allele

    :param allele_name: str - The allele name
    :return: 1 for class 1, 2 for class 2, None for invalid allele name
    """

    if allele_name == "None":
        return None

    h = get_mhc_name_split(allele_name)
    if h[0] in ("A", "B", "C", "E", "F", "G", "K", "L"):
        return 1
    if h[0] in ("DRB1", "DRB3", "DRB4", "DRB5", "DQB1"):
        return 2
    raise ValueError(f"Invalid allele name: {allele_name}")


def get_mhc_name_split(allele_name, species="human"):
    """Splits an MHC allele name into its components

    e.g.:
    - HLA-A*02:01 -> (A, 2, 1)
    - HLA-A*02:01g -> (A, 2, 1)
    - HLA-DRB1*01:01 -> (DRB1, 1, 1)
    ...
    """

    if species != "human":
        raise NotImplementedError("Only human HLA is supported")
    return re.findall(
        r"^(?:HLA-)*(A|B|C|E|F|G|K|L|DRB1|DRB3|DRB4|DRB5|DQB1)\*(\d+):(\d+)[gQNL]*$",
        allele_name,
    )[0]


def to_mhc_prefix(allele_name):
    """Converts an allele name to its prefix (e.g. for a filename)"""
    if allele_name in ("DRBX*NNNN"):
        result = "None"
    else:
        h = get_mhc_name_split(allele_name)
        result = f"{h[0]}_{h[1]}_{h[2]}"
    return result


def from_mhc_prefix(hla_prefix, species="human"):
    """Converts an allele prefix (e.g. from a filename) to its full name"""
    h = hla_prefix.split("_")
    if species != "human":
        raise NotImplementedError("Only human HLA is supported")
    return f"HLA-{h[0]}*{h[1]}:{h[2]}"


def str_to_hla_names(text, min_hla_gene_cnt=0, max_hla_gene_cnt=2):
    """Finds all the HLA names within a string and returns them as a list"""

    # HLA-[ABC]\d{2}:\d{2}
    candidate_list = [
        f"HLA-{x[0]}*{int(x[1]):02d}:{int(x[2]):02d}"
        for x in re.findall(HLA_REGEX, text)
    ]
    if len(candidate_list) != (
        text.count("HLA-") - text.count("HLA-D")
    ):  # HLA-D are class 2
        candidate_list = []

    if text.count("HLA") > 1:
        if (
            not (min_hla_gene_cnt <= text.count("HLA-A") <= max_hla_gene_cnt)
            or not (min_hla_gene_cnt <= text.count("HLA-B") <= max_hla_gene_cnt)
            or not (min_hla_gene_cnt <= text.count("HLA-C") <= max_hla_gene_cnt)
        ):
            candidate_list = []

    candidate_list = list(set(candidate_list))
    candidate_list.sort()

    return candidate_list
