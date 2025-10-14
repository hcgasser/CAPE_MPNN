#!/usr/bin/env python

import os
import argparse
import sys
import traceback
import pdb
import copy

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import defaultdict

from kit.path import join
from kit.log import setup_logger
from kit.bioinf import generate_random_aa_seq, AA1_STD, N_AA_STD, AA1_TO_IDX
from kit.bioinf.immuno.mhc_1 import Mhc1Predictor
from kit.bioinf.immuno.mhc_1._pwm import Mhc1PredictorPwm
from kit.bioinf.immuno.utils import get_mhc_1_setup_hash
from kit.data import str_to_file, file_to_str
from kit.bioinf.pwm import count_amino_acid_occurrences, calc_PWMs, save_PWMs


def score_peptides(peptides, PWMs_log, aa_na=None):
    scores = []
    for peptide in tqdm(peptides):
        scores.append(score_peptide(peptide, PWMs_log, aa_na))

    return scores


def score_peptide(peptide, PWMs_log, aa_na=None):
    length = len(peptide)
    pwm_log = PWMs_log[length].values
    pep = np.array([AA1_TO_IDX.get(p, N_AA_STD) for p in peptide])
    if aa_na is not None:
        pwm_log = np.vstack([pwm_log, np.ones((1, length))*aa_na])
    return pwm_log[pep, np.arange(length)].sum()


def get_percentiles(peptides, scores, percentiles, length):
    allele_scores = {"peptide": peptides, f"score": scores}
    df_allele_scores = pd.DataFrame(allele_scores).set_index("peptide")
    df_allele_scores['length'] = df_allele_scores.apply(lambda x: len(x.name), axis=1)

    df_percentiles = pd.DataFrame(
        index=[f"pc_{p}" for p in percentiles],
        columns=["value"],
    )
    df_percentiles.index.name = "Info"

    df_percentiles.loc[
        [f"pc_{p}" for p in percentiles], 'value'
    ] = np.percentile(
        df_allele_scores.query(f"length == {length}")[f"score"],
        percentiles,
    )

    return df_percentiles


def process_presented_peptides(allele, df_ranks, lengths, limit_rank, folder, generate_percentiles=True):
    percentiles = [99.9, 99.5, 99.0, 98.0, 97.5, 95.0, 90.0, 75.0, 50.0]
    allele_for_path = allele.replace("*", "_")

    # get presented peptides
    peptides = list(df_ranks.index)
    peptides_presented = []
    for peptide, row in tqdm(
        df_ranks.iterrows(), "get presented peptides", leave=False
    ):
        if row[allele] <= limit_rank:
            peptides_presented.append(peptide)   


    counts = {}
    PWMs = {}
    PWMs_log = {}

    # count how often each AA is observed at each position
    counts[allele] = count_amino_acid_occurrences(peptides_presented)

    # convert counts to probs and log probs
    _pwms, _log_pwms = calc_PWMs(counts[allele])
    PWMs[allele], PWMs_log[allele] = _pwms, _log_pwms

    # save the produced PWMs
    save_PWMs(allele, PWMs[allele], PWMs_log[allele], folder)

    # score all random peptides with the log pwm
    scores = score_peptides(peptides, PWMs_log[allele])
    allele_scores = {"peptide": peptides, f"{allele}_score": scores}
    df_ranks = df_ranks.join(pd.DataFrame(allele_scores).set_index("peptide"))

    if generate_percentiles:
        # generate percentiles
        for length in lengths:
            df_percentiles = get_percentiles(peptides, scores, percentiles, length)
            df_percentiles.rename(columns={'value': allele}, inplace=True)

            df_recall = pd.DataFrame(
                index=[f"recall_{p}" for p in percentiles],
                columns=[allele],
            )
            df_precision = pd.DataFrame(
                index=[f"precision_{p}" for p in percentiles],
                columns=[allele],
            )
            df_F1 = pd.DataFrame(
                index=[f"F1_{p}" for p in percentiles],
                columns=[allele],
            )
            for p in percentiles:
                score = df_percentiles.loc[f"pc_{p}", allele]
                true_positives = df_ranks.query(
                    f"`{allele}_score` > {score} "
                    f"and `{allele}` <= {limit_rank} "
                    f"and length == {length} "
                ).shape[0]
                false_negatives = df_ranks.query(
                    f"`{allele}_score` <= {score} "
                    f"and `{allele}` <= {limit_rank} "
                    f"and length == {length} "
                ).shape[0]
                false_positives = df_ranks.query(
                    f"`{allele}_score` > {score} "
                    f"and `{allele}` > {limit_rank} "
                    f"and length == {length} "
                ).shape[0]

                # calculate recall vs the principal classifier
                df_recall.loc[f"recall_{p}", allele] = (
                    true_positives / (true_positives + false_negatives)
                    if (true_positives + false_negatives) > 0
                    else None
                )

                # calculate precision vs the principal classifier
                df_precision.loc[f"precision_{p}", allele] = (
                    true_positives / (true_positives + false_positives)
                    if (true_positives + false_positives) > 0
                    else None
                )

                # calculate F1 vs the principal classifier
                df_F1.loc[f"F1_{p}", allele] = (
                    true_positives
                    / (true_positives + 0.5 * (false_positives + false_negatives))
                    if (true_positives + 0.5 * (false_positives + false_negatives))
                    > 0
                    else None
                )

            _df = pd.concat([df_percentiles, df_recall, df_precision, df_F1])
            _df.index.name = 'Info'
            _df.to_csv(
                os.path.join(
                    folder,
                    "pwm",
                    allele_for_path,
                    f"pc-{allele.replace('*', '_')}-{limit_rank}-{length}.csv",
                )
            )
    


def main(_args):
    ''' Main function to rank and score random peptides for MHC-I binding and generate pwm matrices
    based on these

    Example: MHC-I_rank_peptides.py --output ${PF}/data/input/immuno/mhc_1/Mhc1PredictorPwm --alleles ${MHC_Is} --peptides_per_length 1000000
    
    '''

    lengths = [int(length) for length in _args.lengths.split("+")]
    alleles = _args.alleles.split("+")
    peptides_per_length = _args.peptides_per_length
    backend = _args.backend
    tasks = _args.tasks.split("+")
    limit_rank = 0.02

    if _args.output != "":
        folder = join(_args.output)
    else:
        folder = join(
            os.environ["DATA"],
            "processed",
            "MHC_class_I",
            "random_peptides",
            _args.lengths,
            f"random_{int(peptides_per_length)}",
            backend,
        )

    join(folder, "pwm")
    finished_alleles_file_path = os.path.join(folder, "finished_alleles.txt")
    finished_alleles = file_to_str(finished_alleles_file_path).split("\n")

    if tasks != ["stats"]:
        eval_alleles = [allele for allele in alleles if allele not in finished_alleles]
    else:
        eval_alleles = [allele for allele in alleles]

    predictor_pwm = None
    if "stats" in tasks:
        predictor_pwm = Mhc1PredictorPwm(data_dir_path=folder, limit=limit_rank)


    random_peptides_file = os.path.join(folder, "random_peptides.txt")

    if os.path.exists(random_peptides_file):
        print("Read existing random_peptides from disk")
        random_peptides = file_to_str(random_peptides_file).split("\n")
    else:
        print("Generate new random_peptides")
        random_peptides = []
        for length in tqdm(lengths, "Generate random peptides"):
            random_peptides += [
                generate_random_aa_seq(length) for _ in range(peptides_per_length)
            ]
        str_to_file("\n".join(random_peptides), random_peptides_file)

    df_ranks = pd.DataFrame(index=random_peptides, columns=["length"])
    df_ranks.index.name = "peptide"
    df_ranks["length"] = df_ranks.apply(lambda row: len(row.name), axis=1)

    predictor = None
    pbar = tqdm(eval_alleles, "Alleles")

    for allele in pbar:
        allele_for_path = allele.replace("*", "_")
        allele_rank_file_path = join(folder, "ranks",f"{allele_for_path}.csv")
        if "rank" in tasks:
            selection = "y"
            if os.path.exists(allele_rank_file_path):
                selection = input(
                    f"{allele_rank_file_path} already exists. Would you like to overwrite? (y/n)"
                )

            if selection == "y":
                pbar.set_description(f"{allele} - rank")
                predictor = (
                    Mhc1Predictor.get_predictor(backend)()
                    if predictor is None
                    else predictor
                )
                predictor.queue_peptides(random_peptides, allele)
                predictor.predict_missing_peptides()
                df_ranks[allele] = df_ranks.apply(
                    lambda row: predictor.peptide_percentile(row.name, allele), axis=1
                )
                df_ranks[[allele]].sort_index().to_csv(allele_rank_file_path)

        if "pwm" in tasks:
            pbar.set_description(f"{allele} - pwm")
            if allele not in df_ranks:
                df = pd.read_csv(allele_rank_file_path).set_index(
                    "peptide"
                )
                df_ranks = df_ranks.join(df, how="left")

            process_presented_peptides(allele, df_ranks, lengths, limit_rank, folder)
            str_to_file(f"{allele}\n", finished_alleles_file_path, append=True)

        if "stats" in tasks:
            for length in lengths:
                predictor_pwm.load_allele(allele, length)
            predictor_pwm.load_agg_ranks(allele, [allele], save_definition=False)
            peptides_presented = predictor_pwm.get_presented_any(allele)
            counts = count_amino_acid_occurrences(peptides_presented)
            percentile_statistics = predictor_pwm.calc_percentile_statistics(allele)
            predictor_pwm.save_percentile_statistics(allele, percentile_statistics)

    if "agg" in tasks:
        mhc_1_setup_hash = get_mhc_1_setup_hash(alleles)
        if predictor_pwm is None:
            predictor_pwm = Mhc1PredictorPwm(data_dir_path=folder, limit=limit_rank)
        predictor_pwm.load_agg_ranks(mhc_1_setup_hash, alleles)
        peptides_presented = predictor_pwm.get_presented_any(mhc_1_setup_hash)
        counts = count_amino_acid_occurrences(peptides_presented)
        predictor_pwm.calc_PWMs(mhc_1_setup_hash, counts)
        predictor_pwm.save_PWMs(mhc_1_setup_hash)
        percentile_statistics = predictor_pwm.calc_percentile_statistics(mhc_1_setup_hash)
        predictor_pwm.save_percentile_statistics(mhc_1_setup_hash, percentile_statistics)

        # # load all allele ranks
        # for allele in alleles:
        #     allele_for_path = allele.replace("*", "_")
        #     allele_rank_file_path = join(folder, "ranks",f"{allele_for_path}.csv")
        #     if allele not in df_ranks:
        #         df = pd.read_csv(allele_rank_file_path).set_index(
        #             "peptide"
        #         )
        #         df_ranks = df_ranks.join(df, how="left")

        # # get the name for the new setup
        
        # # check if all peptides are present
        # assert set(df_ranks.index) == set(random_peptides)

        # df_ranks[mhc_1_setup_hash] = df_ranks.apply(
        #     lambda row: np.min([row[_allele] for _allele in alleles]),
        #     axis=1
        # )

        # process_presented_peptides(mhc_1_setup_hash, df_ranks, lengths, limit_rank, folder)



                


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument(
        "--lengths",
        type=str,
        default="8+9+10",
        help="the lengths of peptides to evaluate (e.g. 8+9+10)",
    )
    argparser.add_argument(
        "--peptides_per_length",
        type=int,
        default=100,
        help="the number of peptides per length to evaluate (e.g. 100000)",
    )
    argparser.add_argument(
        "--alleles",
        type=str,
        default="",
        help=(
            "the MHC alleles to evaluate the random peptides on "
            "(e.g. HLA-A*02:01+HLA-A*24:02+HLA-B*07:02+HLA-B*39:01+HLA-C*07:01+HLA-C*16:01)"
        ),
    )
    argparser.add_argument(
        "--backend",
        type=str,
        default="Mhc1PredictorNetMhcPan",
        help="the backend predictor to use (e.g. netMHCpan or MHCflurry)",
    )
    argparser.add_argument(
        "--tasks",
        type=str,
        default="rank+pwm",
        help=(
            "rank uses the backend to rank the peptides, "
            "pwm loads previously ranked peptides and produces position weight matrices, "
            "stats calculates the percentiles and F1 scores for the PWMs, "
            "agg aggregates the results of the alleles into a single pwm "
        ),
    )
    argparser.add_argument(
        "--output", type=str, default="", help="the directory path for the output"
    )
    args = argparser.parse_args()

    try:
        setup_logger()
        main(args)
    except Exception as e:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
