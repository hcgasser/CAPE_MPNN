def calc_sequence_recovery(seq_data, seq_generated):
    assert len(seq_data) == len(seq_generated)
    same = 0
    for d, g in zip(seq_data, seq_generated):
        if d == g:
            same += 1

    return same/len(seq_data)