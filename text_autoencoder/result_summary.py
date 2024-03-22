#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import pandas as pd
import glob
import os
import re
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='summarize evaluation')
    parser.add_argument("--dir", type=str, default=None)
    return parser

def summarize_result(args):
    steps = ['5', '10', '20', '30', '50'] #, 
    ppl, accuracy, dist, ent, cnt, selfbleu, rep = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(int), defaultdict(float), defaultdict(float)
    for fname in glob.glob(args.dir+"/*.txt"):
        step = re.search(r'samples_([0-9]+)x', os.path.basename(fname))
        step = step.group(1) if step else None
        if step is not None and step in steps:
            with open(fname, 'r') as f:
                # Read the lines from the current position to the end of the file
                lines = f.readlines()
                # Get the last 12 lines
                last_7_lines = lines[-7:] # lines[-6:]
                
                if 'Gen PPL' in last_7_lines[5]:
                    last_7_lines = last_7_lines[3:]
                else:
                    acc_search = re.search(r'Gen ACC: (\d+\.\d+)', last_7_lines[4])
                    accuracy[step] += float(acc_search.group(1))
                div_search = re.search(r'Diversity: (\d+\.\d+)/(\d+\.\d+)/(\d+\.\d+)', last_7_lines[0])
                ppl_search = re.search(r'Gen PPL: (\d+\.\d+)', last_7_lines[2])
                               
                if div_search and ppl_search:
                    # Get the two floats from the match object
                    dist[step] += float(div_search.group(1))
                    ent[step] += float(div_search.group(2))
                    selfbleu[step] += float(div_search.group(3))
                    # rep[step] += float(div_search.group(4))
                    ppl[step] += float(ppl_search.group(1))
                    cnt[step] += 1
    results = []
    for step in steps:
        results.append([step, f"{ppl[step]/cnt[step]:.3f}", f"{accuracy[step]/cnt[step]:.2%}", f"{dist[step]/cnt[step]:.2f}/{ent[step]/cnt[step]:.3f}", f"{selfbleu[step]/cnt[step]:.2f}"])

    df = pd.DataFrame(results, columns=["Step"] + ["PPL"] + ["ACC"] + ["Diversity"] + ["Self_bleu"])
    df.to_csv(os.path.join(args.dir, "summary.csv"), sep="\t")


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    summarize_result(args)