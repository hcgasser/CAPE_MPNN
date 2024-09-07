import os
from kit.bioinf.sf import SequenceFrame

sf = SequenceFrame()
sf.from_fasta(
    os.path.join(os.environ["PF"], "libs", "kit", "tests", "test_data", "fastafile.fa")
)


def test_overall_length():
    # overall we expect entries:
    assert all([len(sf) == 2, len(sf.df) == 2, len(sf.df_src) == 5])


def test_double_entries():
    # we exptect 3 entries with this seq_hash: 1. [base + base repeat] and 2. base without stop token and 3. base with '-'
    df_tmp = sf.query(
        'seq_hash == "0a97f5424395723bcb911d284b01975be2450aea8e87236bacc513f5ac630cbd"'
    )
    assert len(df_tmp) == 1
    assert len(df_tmp.iloc[0].accession) == 4

    df_tmp = sf.query(
        'seq_hash == "7df0d0b1f27c8cb17a9aaff5302af36d7b1ed72a2d5a40528358a6b91199b16a"'
    )
    assert len(df_tmp) == 1
    assert len(df_tmp.iloc[0].accession) == 1


def test_final_stop_token():
    df_tmp = sf.query(
        'seq_hash == "7df0d0b1f27c8cb17a9aaff5302af36d7b1ed72a2d5a40528358a6b91199b16a"'
    )
    assert len(df_tmp) == 1
    row = df_tmp.iloc[0]
    assert (
        "MGGKWSKSSMVGWPAVRERMRRAEPAAEGVGAVSRDLERHGAITSSNTAKNNAALAWLEAQEEEEVGFPVRPQVPLRPMTYKAAIDLSHFLKEKGGLEGLIYSQKRQDILDLWVYHTQGYFPDWQNYTPGPGTRFPLTFGWCFKLVPVEPEKVEAANEGENNCLLHPMSLHGMEDSEGEVLQWKFDSRLALRHXAREKHPEYYKDC*"
        in row.seq_src
    )
    assert (
        row.seq
        == "MGGKWSKSSMVGWPAVRERMRRAEPAAEGVGAVSRDLERHGAITSSNTAKNNAALAWLEAQEEEEVGFPVRPQVPLRPMTYKAAIDLSHFLKEKGGLEGLIYSQKRQDILDLWVYHTQGYFPDWQNYTPGPGTRFPLTFGWCFKLVPVEPEKVEAANEGENNCLLHPMSLHGMEDSEGEVLQWKFDSRLALRHXAREKHPEYYKDC"
    )

    sf.set_view(None, final_stop_token=True)
    df_tmp = sf.query(
        'seq_hash == "0a97f5424395723bcb911d284b01975be2450aea8e87236bacc513f5ac630cbd"'
    )
    row = df_tmp.iloc[0]
    assert (
        row.seq
        == "MGGKWSKSSMVGWPAVRERMRRAEPAAEGVGAVSRDLERHGAITSSNTAKNNAALAWLEAQEEEEVGFPVRPQVPLRPMTYKAAIDLSHFLKEKGGLEGLIYSQKRQDILDLWVYHTQGYFPDWQNYTPGPGTRFPLTFGWCFKLVPVEPEKVEAANEGENNCLLHPMSLHGMEDSEGEVLQWKFDSRLALRHMAREKHPEYYKDC*"
    )
    assert (
        "MGGKWSKSSMVGWPAVRERMRRAEPAAEGVGAVSRDLERHGAITSSNTAKNNAALAWLEAQEEEEVGFPVRPQVPLRPMTYKAAIDLSHFLKEKGGLEGLIYSQKRQDILDLWVYHTQGYFPDWQNYTPGPGTRFPLTFGWCFKLVPVEPEKVEAANEGENNCLLHPMSLHGMEDSEGEVLQWKFDSRLALRHMAREKHPEYYKDC*"
        in row.seq_src
    )
    assert (
        "MGGKWSKSSMVGWPAVRERMRRAEPAAEGVGAVSRDLERHGAITSSNTAKNNAALAWLEAQEEEEVGFPVRPQVPLRPMTYKAAIDLSHFLKEKGGLEGLIYSQKRQDILDLWVYHTQGYFPDWQNYTPGPGTRFPLTFGWCFKLVPVEPEKVEAANEGENNCLLHPMSLHGMEDSEGEVLQWKFDSRLALRHMAREKHPEYYKDC"
        in row.seq_src
    )

    df_tmp = sf.query(
        'seq_hash == "7df0d0b1f27c8cb17a9aaff5302af36d7b1ed72a2d5a40528358a6b91199b16a"'
    )
    row = df_tmp.iloc[0]
    assert (
        row.seq
        == "MGGKWSKSSMVGWPAVRERMRRAEPAAEGVGAVSRDLERHGAITSSNTAKNNAALAWLEAQEEEEVGFPVRPQVPLRPMTYKAAIDLSHFLKEKGGLEGLIYSQKRQDILDLWVYHTQGYFPDWQNYTPGPGTRFPLTFGWCFKLVPVEPEKVEAANEGENNCLLHPMSLHGMEDSEGEVLQWKFDSRLALRHXAREKHPEYYKDC*"
    )
    assert (
        "MGGKWSKSSMVGWPAVRERMRRAEPAAEGVGAVSRDLERHGAITSSNTAKNNAALAWLEAQEEEEVGFPVRPQVPLRPMTYKAAIDLSHFLKEKGGLEGLIYSQKRQDILDLWVYHTQGYFPDWQNYTPGPGTRFPLTFGWCFKLVPVEPEKVEAANEGENNCLLHPMSLHGMEDSEGEVLQWKFDSRLALRHXAREKHPEYYKDC*"
        in row.seq_src
    )
    assert (
        "MGGKWSKSSMVGWPAVRERMRRAEPAAEGVGAVSRDLERHGAITSSNTAKNNAALAWLEAQEEEEVGFPVRPQVPLRPMTYKAAIDLSHFLKEKGGLEGLIYSQKRQDILDLWVYHTQGYFPDWQNYTPGPGTRFPLTFGWCFKLVPVEPEKVEAANEGENNCLLHPMSLHGMEDSEGEVLQWKFDSRLALRHXAREKHPEYYKDC"
        not in row.seq_src
    )

    sf.set_view(None, final_stop_token=False)
    df_tmp = sf.query(
        'seq_hash == "0a97f5424395723bcb911d284b01975be2450aea8e87236bacc513f5ac630cbd"'
    )
    row = df_tmp.iloc[0]
    assert (
        row.seq
        == "MGGKWSKSSMVGWPAVRERMRRAEPAAEGVGAVSRDLERHGAITSSNTAKNNAALAWLEAQEEEEVGFPVRPQVPLRPMTYKAAIDLSHFLKEKGGLEGLIYSQKRQDILDLWVYHTQGYFPDWQNYTPGPGTRFPLTFGWCFKLVPVEPEKVEAANEGENNCLLHPMSLHGMEDSEGEVLQWKFDSRLALRHMAREKHPEYYKDC"
    )
    assert (
        "MGGKWSKSSMVGWPAVRERMRRAEPAAEGVGAVSRDLERHGAITSSNTAKNNAALAWLEAQEEEEVGFPVRPQVPLRPMTYKAAIDLSHFLKEKGGLEGLIYSQKRQDILDLWVYHTQGYFPDWQNYTPGPGTRFPLTFGWCFKLVPVEPEKVEAANEGENNCLLHPMSLHGMEDSEGEVLQWKFDSRLALRHMAREKHPEYYKDC*"
        in row.seq_src
    )
    assert (
        "MGGKWSKSSMVGWPAVRERMRRAEPAAEGVGAVSRDLERHGAITSSNTAKNNAALAWLEAQEEEEVGFPVRPQVPLRPMTYKAAIDLSHFLKEKGGLEGLIYSQKRQDILDLWVYHTQGYFPDWQNYTPGPGTRFPLTFGWCFKLVPVEPEKVEAANEGENNCLLHPMSLHGMEDSEGEVLQWKFDSRLALRHMAREKHPEYYKDC"
        in row.seq_src
    )

    df_tmp = sf.query(
        'seq_hash == "7df0d0b1f27c8cb17a9aaff5302af36d7b1ed72a2d5a40528358a6b91199b16a"'
    )
    row = df_tmp.iloc[0]
    assert (
        row.seq
        == "MGGKWSKSSMVGWPAVRERMRRAEPAAEGVGAVSRDLERHGAITSSNTAKNNAALAWLEAQEEEEVGFPVRPQVPLRPMTYKAAIDLSHFLKEKGGLEGLIYSQKRQDILDLWVYHTQGYFPDWQNYTPGPGTRFPLTFGWCFKLVPVEPEKVEAANEGENNCLLHPMSLHGMEDSEGEVLQWKFDSRLALRHXAREKHPEYYKDC"
    )
    assert (
        "MGGKWSKSSMVGWPAVRERMRRAEPAAEGVGAVSRDLERHGAITSSNTAKNNAALAWLEAQEEEEVGFPVRPQVPLRPMTYKAAIDLSHFLKEKGGLEGLIYSQKRQDILDLWVYHTQGYFPDWQNYTPGPGTRFPLTFGWCFKLVPVEPEKVEAANEGENNCLLHPMSLHGMEDSEGEVLQWKFDSRLALRHXAREKHPEYYKDC*"
        in row.seq_src
    )
    assert (
        "MGGKWSKSSMVGWPAVRERMRRAEPAAEGVGAVSRDLERHGAITSSNTAKNNAALAWLEAQEEEEVGFPVRPQVPLRPMTYKAAIDLSHFLKEKGGLEGLIYSQKRQDILDLWVYHTQGYFPDWQNYTPGPGTRFPLTFGWCFKLVPVEPEKVEAANEGENNCLLHPMSLHGMEDSEGEVLQWKFDSRLALRHXAREKHPEYYKDC"
        not in row.seq_src
    )
