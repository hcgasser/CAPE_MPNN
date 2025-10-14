import hashlib
from collections import defaultdict

from Bio import PDB

from kit.bioinf import AA1_FULL, NEXT_CHAIN


def keep_chains_in_structure(structure, keep_chains):
    if keep_chains is None:
        return {}
    
    to_remove = {}   # model.detach_child has to be called after the loop
    for model in structure:
        for chain in model:
            c_id = chain.id 

            if c_id not in keep_chains:
                to_remove[model] = to_remove.get(model, [])
                to_remove[model].append(c_id)

    for model, _to_remove in to_remove.items():
        for c_id in _to_remove:
            model.detach_child(c_id)

    return to_remove


def structure_to_seq(structure, return_full=True, gaps='-', aa3_replace=None, aa_ids=[' ']):
    seqs_mod = []

    for model in structure:  # each model could be a different confirmation of the molecule
        # (also NMR Nuclear Magnetic Resonance produces multiple models)

        residues = {}
        for chain in model:
            residues[chain.id] = {}
            for residue in chain:
                if residue.id[0] in aa_ids:  # ignore heteroatoms
                    residues[chain.id][residue.id[1]] = residue.get_resname()

        seqs = {}
        for chain_id, chain_dict in residues.items():
            seq = ""
            chain_dict = dict(sorted(chain_dict.items()))
            _tmp = list(chain_dict)
            if len(_tmp) == 0:
                continue
            rmin, rmax = _tmp[0], _tmp[-1]
            seq_all = [gaps] * (rmax - rmin + 1)
            for pos, res3 in chain_dict.items():
                if aa3_replace is not None and res3 in aa3_replace:
                    res3 = aa3_replace[res3]
                if res3 in PDB.Polypeptide.standard_aa_names:
                    res1 = PDB.Polypeptide.protein_letters_3to1[res3]
                    seq += res1
                    seq_all[pos - rmin] = res1
                elif res3 in ['UNK']:
                    seq_all[pos - rmin] = 'X'
                elif seq not in ("", None):
                    raise Exception(f"unknown residue {res3}")
                else:
                    seq = None

            if seq is not None:
                seqs[chain_id] = ''.join(seq_all) if return_full else seq
        seqs_mod.append(seqs)
    return seqs_mod


def get_seq_hash(chains, translate=("", "", "*-")):
    """Returns the SHA-256 hash of a sequence.

    :param seq: str - sequence
    :param translate: tuple - characters to translate
        the sequence with before its hash is computed
    :return hash_code: str - SHA-256 hash of the sequence
    """

    seq = chains_to_seq(chains)

    seq = seq.translate(str.maketrans(*translate))

    # Create a SHA-256 hash object
    hash_object = hashlib.sha256()

    # Encode the string as bytes and update the hash object
    hash_object.update(seq.encode("utf-8"))

    # Get the hexadecimal representation of the hash digest
    hash_code = hash_object.hexdigest()

    return hash_code


def chains_to_seq(chains):
    seq = None
    if isinstance(chains, dict):  # potential complex
        keys = sorted(chains)
        chains = [chains[k] for k in keys]
        seq = "/".join(chains)
    elif isinstance(chains, str):
        seq = chains
    else:
        raise Exception("'chains' wrong datatye")
    return seq


def calc_seq_aa_cnts(seqs, standardize=False, aggregate=False):
    if isinstance(seqs, str):
        seqs = [seqs]

    if aggregate:
        seqs = [''.join([s for s in seqs])]

    aa_cnts = defaultdict(lambda: [0 for _ in seqs])
    for i, seq in enumerate(seqs):
        length = len(seq)
        for s in seq:
            assert s in AA1_FULL or s == NEXT_CHAIN
            aa_cnts[s][i] += 1

        if standardize:
            for aa, cnts in aa_cnts.items():
                aa_cnts[aa][i] = aa_cnts[aa][i]/length

    return aa_cnts


def check_same_sequences(seq1, seq2):
    chains1 = set(seq1.split('/'))
    chains2 = set(seq2.split('/'))

    return chains1 == chains2