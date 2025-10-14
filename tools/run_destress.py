import os
import subprocess
import argparse

from kit.data import file_to_str
from kit.loch.oo import Loch


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--destress_prog_dir_path",
    type=str,
    help="the path to the destress directory",
)
argparser.add_argument(
    "--destress_input_dir_path",
    type=str,
    help="the path to the destress input directory",
)
args = argparser.parse_args()

if args.destress_prog_dir_path is None:
    destress_prog_dir_path = os.path.join(os.environ['PROGRAMS'], 'de-stress')
elif not os.path.exists(args.destress_prog_dir_path):
    raise FileNotFoundError(f"Path '{args.destress_prog_dir_path}' does not exist")
else:
    destress_prog_dir_path = args.destress_prog_dir_path

if args.destress_input_dir_path is None:
    destress_input_dir_path = os.path.join(os.environ['PF'], 'artefacts', 'CAPE-MPNN', 'eval', 'de-stress')
elif not os.path.exists(args.destress_input_dir_path):
    raise FileNotFoundError(f"Path '{args.destress_input_dir_path}' does not exist")
else:
    destress_input_dir_path = args.destress_input_dir_path

loch_path = os.path.join(os.environ['PF'], 'artefacts', 'CAPE-MPNN', 'loch')
loch = Loch(loch_path=loch_path)

for predictor_structure_name in ['exp', 'AF']:
    destress_input_file_path = os.path.join(destress_input_dir_path, f"for_destress_{predictor_structure_name}.txt")
    seq_hashes = [x for x in file_to_str(destress_input_file_path).split('\n') if x != '']
    print(f"{predictor_structure_name}: {len(seq_hashes)}")

    loch.run_destress(seq_hashes, destress_prog_dir_path, predictor_structure_name=predictor_structure_name)

