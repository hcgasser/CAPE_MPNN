""" Utility functions for the loch module. """

from kit.bioinf.utils import chains_to_seq, get_seq_hash

def get_set_hash(seq_hashes):
    seq_hashes = sorted(seq_hashes)
    seq_hashes = ''.join(seq_hashes)
    return str_to_hash(seq_hashes)
