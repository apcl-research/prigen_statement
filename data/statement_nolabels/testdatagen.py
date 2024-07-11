# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets
from datasets import Dataset

import pickle
import random
import argparse

random.seed(1337)

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num-proc', type=int, default=4)
    parser.add_argument('--testfids-file', type=str, default='/nublar/datasets/prigen/prigen_statement/new_data/testfid.pkl')
    parser.add_argument('--statement-file', type=str, default='/nublar/datasets/prigen/prigen_statement/new_data/duplicates.pkl')
    parser.add_argument('--data-dir', type=str, default='testset/')
    

    args = parser.parse_args()

    num_proc = outdir = args.num_proc
    testfids_file = args.testfids_file
    statement_file = args.statement_file
    data_dir = args.data_dir

    if(not os.path.exists(data_dir)):
        os.mkdir(data_dir)


    testfids = pickle.load(open(testfids_file, 'rb'))
    statements = pickle.load(open(statement_file, 'rb'))
    
    for res in tqdm(statements):
        fid = res['fid']
        if(fid not in testfids):
            continue

        with open(f'{data_dir}{fid}.txt', 'w') as f:
            first_hop = "FIRST HOP:\t" + res['code'] + "\n"
            #label = f"TAG:\t" + res['label'] + "\n"
            s = first_hop + "STATEMENT:<s>\t"
            f.write(s)
