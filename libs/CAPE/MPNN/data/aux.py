def S_to_seqs(S, chain_encoding_all):
    """Converts a tensor of indices to a list of sequences.

    :param S: tensor of indices
    :return: list of sequences
    """

    seqs = []
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

    for s, ce in zip(S, chain_encoding_all):
        c_prev = int(ce[0])
        seq = []
        for a, c in zip(s, ce):
            if int(c) == 0:
                break
            if int(c) != c_prev:
                seq.append('/')
                c_prev = int(c)
            seq.append(alphabet[int(a)])
        seqs.append("".join(seq))
    return seqs
