class PrefixTree():
    alphabet = None
    n_alphabet = None
    alphabet_to_idx = None

    @classmethod
    def set_alphabet(cls, alphabet):
        cls.alphabet = alphabet
        cls.n_alphabet = len(alphabet)
        cls.alphabet_to_idx = {l: i for i, l in enumerate(alphabet)}

    def __init__(self, cnt_nodes=0):
        """Initialize a prefix tree.
        
        The tree is represented as a list of children, where each child corresponds to a letter in the alphabet.
        """

        self.children = [None] * self.n_alphabet
        self.cnt_nodes = cnt_nodes  # number of nodes in the sub-tree below this node (excluding this node)
        self.max_depth = 0  # max depth of the sub-tree below this node (excluding this node)

    def get(self, l):
        """Get the child node corresponding to the letter l."""

        return self.children[self.alphabet_to_idx[l]]

    def get_kmer(self, kmer):
        """Get the node of the kmer."""

        if len(kmer) > 0:
            child = self.get(kmer[0])
            if child is not None:
                return child.get_kmer(kmer[1:])
            else:
                return None
        return self

    def add_kmer(self, kmer):
        _new_nodes = 0
        if len(kmer) > 0:
            l = kmer[0]
            child = self.get(l)
            if child is None:
                child = PrefixTree(1)
                self.children[self.alphabet_to_idx[l]] = child
                _new_nodes += 1

            _new_nodes += child.add_kmer(kmer[1:])
            self.cnt_nodes += _new_nodes
            self.max_depth = max(self.max_depth, child.max_depth + 1)
        return _new_nodes

    def add_seq(self, seq, length):
        missing = set()
        for start in range(0, len(seq)):
            kmer = seq[start:start + length]
            if all([(s in self.alphabet) for s in kmer]):
                self.add_kmer(kmer)
            else:
                missing.add(kmer)
        return missing

    def has_kmer(self, kmer, pos=0, disregard=None):
        """ Check if the tree contains the kmer

        pos: the currently checked position in the kmer
        disregard: a list of positions to disregard in the kmer        
        """

        if pos < len(kmer):
            if disregard is None or pos not in disregard:
                child = self.get(kmer[pos])
                if child is None:
                    return False
                elif pos+1 == len(kmer):
                    return True
                else:
                    return child.has_kmer(kmer, pos=pos+1, disregard=disregard)
            elif len(kmer) > 1:
                has = []
                for child in self.children:
                    if child is not None:
                        has.append(child.has_kmer(kmer, pos=pos+1, disregard=disregard))
                return any(has)
        return True


        # if pos == 0 and disregard is not None:
        #     disregard = [d if d > 0 else len(kmer) + d for d in disregard]
        # if len(kmer) > 0:
        #     if disregard is None or pos not in disregard:
        #         child = self.get(kmer[0])
        #         if child is None:
        #             return False
        #         else:
        #             return child.has_kmer(kmer[1:], pos=pos + 1, disregard=disregard)
        #     elif len(kmer) > 1:
        #         has = []
        #         for child in self.children:
        #             if child is not None:
        #                 has.append(child.has_kmer(kmer[2:], pos=pos + 2, disregard=disregard))
        #         return any(has)
        # return True