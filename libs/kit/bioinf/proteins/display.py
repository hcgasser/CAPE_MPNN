from collections import defaultdict

import numpy as np

from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
import blosum as bl

from IPython.display import HTML, display
from kit.bioinf import AA1_FULL, NEXT_CHAIN, AA_CHARACTERISTICS_1
from kit.bioinf.utils import calc_seq_aa_cnts


def print_seq(seq,
              name=None, info=None, spaces=10, line_breaks=50,
              immuno_setup=None, predictor_setup=None, ref_seq=None,
              latex=False, only_first_chain=False, 
              c_unknown='gray', c_epitope='yellow', c_anchors='red'):
    if not latex:
        text = "<span style='font-family: DejaVu Sans Mono'>"
        if name is not None:
            text += f'<h4>{name}</h4>'
        if info is not None:
            text += f'{info}<br>'
    else:
        text = ''
        if name is not None:
            _name = name.replace('_', '\\_')
            text += f"\\noindent \\textbf{{{_name}}}\\quad "
        if info is not None:
            text += f"{info} "
        text += "\\\\ "
        
    chains = seq.split(NEXT_CHAIN)
    if only_first_chain:
        chains = [chains[0]]

    result_chains = []
    for seq in chains:
        _seq = [[s, None] for s in seq]

        if immuno_setup is not None and predictor_setup is not None:
            if 'mhc_1' in immuno_setup:
                positions_epitope, positions_anchors, positions_unknown = \
                    predictor_setup['mhc_1'].get_presented_positions(seq, immuno_setup['mhc_1'])

                colors = {
                    c_unknown: positions_unknown,
                    c_epitope: positions_epitope,
                    c_anchors: positions_anchors
                }

                for color, positions in colors.items():
                    for p in positions:
                        _seq[p][1] = color

        result = []
        for s in _seq:
            if s[1] is None:
                result.append(s[0])
            else:
                if not latex:
                    result.append(f"<span style='color: {s[1]};'>{s[0]}</span>")
                else:
                    result.append(f"\\textcolor{{{s[1]}}}{{{s[0]}}}")

        # add line breaks and spaces
        _result = []
        c_spaces, c_breaks = 0, 0
        for v in result:
            if spaces is not None and c_spaces == spaces:
                if not latex:
                    _result.append(" ")
                else:
                    _result.append("\\quad ")
                c_spaces = 0
            if line_breaks is not None and c_breaks == line_breaks:
                if not latex:
                    _result.append("<br>")
                else:
                    _result.append("\\\\ ")
                c_breaks = 0
            _result.append(v)
            c_spaces += 1
            c_breaks += 1
        result = _result

        result_chains.append(''.join(result))

    if not latex:
        text += '/<br>'.join(result_chains)
    else:
        text += '\\\\ '.join(result_chains)

    # add BLOSUM62 scores
    if not latex and ref_seq is not None:
        text += "<br>"
        ref_chains = ref_seq.split('/')

        blosum_matrix = bl.BLOSUM(62)
        blosum_scores = []
        blosum_scores_min, blosum_scores_max = float('inf'), float('-inf')
        blosum_score, blosum_score_ref = 0, 0
        for c, seq in enumerate(ref_chains):
            blosum_scores.append([])
            for p, s in enumerate(seq):
                bs = blosum_matrix[s][chains[c][p]]
                blosum_scores[c].append(bs)
                blosum_score += bs
                blosum_score_ref += blosum_matrix[s][s]
            blosum_scores_min = np.min(blosum_scores[c] + [blosum_scores_min])
            blosum_scores_max = np.max(blosum_scores[c] + [blosum_scores_max])

        norm = Normalize(vmin=blosum_scores_min, vmax=blosum_scores_max)
        cmap = get_cmap('RdYlGn')
        color_max, color_min = cmap(1.0), cmap(0.0)

        rows_1, rows_2 = [], []
        for c, seq in enumerate(ref_chains):
            c_spaces, c_breaks = 0, 0
            row_1, row_2 = "", ""
            for p, s in enumerate(seq):
                if spaces is not None and c_spaces == spaces:
                    row_1 += " "
                    row_2 += " "
                    c_spaces = 0
                if line_breaks is not None and c_breaks == line_breaks:
                    rows_1.append(row_1)
                    rows_2.append(row_2)
                    row_1, row_2 = "", ""
                    c_breaks = 0

                bs = blosum_scores[c][p]

                row_1 += "-" if bs < 0 else "+"

                normalized_score = norm(bs)
                color = cmap(normalized_score)
                hex_color = '#{:02x}{:02x}{:02x}'.format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
                row_2 += f"<span style='color:{hex_color};'>{int(np.abs(blosum_scores[c][p]))}</span>"

                c_spaces += 1
                c_breaks += 1

            rows_1.append(row_1)
            rows_2.append(row_2)
            for row_1, row_2 in zip(rows_1, rows_2):
                text += f"{row_1}<br>{row_2}<br>"
            text += '<br>'

        text += f"BLOSUM-62 score: {int(blosum_score)} vs. {int(blosum_score_ref)}<br>"

    if not latex:
        text += '</span>'

    if not latex:
        display(HTML(text))
    return text


def print_seq_stats(seqs, standardize=True):
    if isinstance(seqs, str):
        seqs = [seqs]
    c_seqs = len(seqs)

    aa_cnts = calc_seq_aa_cnts(seqs)

    # aa_cnts = defaultdict(lambda: [0 for _ in seqs])
    # for i, seq in enumerate(seqs):
    #     length = [len(seq) for seq in seqs]
    #     for s in seq:
    #         assert s in AA1_FULL or s == NEXT_CHAIN
    #         aa_cnts[s][i] += 1

    for c, AAs in AA_CHARACTERISTICS_1.items():
        text, c_total = '', [0 for _ in seqs]
        for a in AAs:
            text += f"  {a:13s}: "
            for i in range(c_seqs):
                v = aa_cnts[a][i] / len(seqs[i])
                c_total[i] += v
                text += f"{v:.2f} "
            text += "\n"

        top_text = f"{c:15s}: "
        for i in range(c_seqs):
            top_text += f"{c_total[i]:.2f} "
        top_text += "\n"

        text = top_text + text
        print(text)

    if standardize is not None:
        if standardize == True:
            standardize = len(seqs[0])
        for aa, cnts in aa_cnts.items():
            aa_cnts[aa] = [c/standardize for c in cnts]
    return aa_cnts