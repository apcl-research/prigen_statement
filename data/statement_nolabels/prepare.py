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
import bincomb

random.seed(1337)

number_to_english = {0:'ZERO', 1:'ONE', 2:'TWO', 3:'THREE', 4:'FOUR', 5:'FIVE', 6:'SIX', 7:'SEVEN'}
# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num-proc', type=int, default=4)
    parser.add_argument('--testfids-file', type=str, default='/nublar/datasets/prigen/prigen_statement/new_data/testfid.pkl')
    parser.add_argument('--valfids-file', type=str, default='/nublar/datasets/prigen/prigen_statement/new_data/valfids.pkl')
    parser.add_argument('--statement-file', type=str, default='/nublar/datasets/prigen/prigen_statement/new_data/unique_response.pkl')
    parser.add_argument('--data-dir', type=str, default='bins/')

    args = parser.parse_args()

    num_proc = outdir = args.num_proc
    testfids_file = args.testfids_file
    valfids_file = args.valfids_file
    statement_file = args.statement_file
    data_dir = args.data_dir
    
    all_first_hop_methods = {}
    all_second_hop_methods = {}
    all_third_hop_methods = {}

    statements = pickle.load(open(statement_file, 'rb'))
    statement_fids = list(statements.keys())


    pt = int(len(statement_fids) * 1.0)

    testfids = pickle.load(open(testfids_file, 'rb'))
    valfids = pickle.load(open(valfids_file, 'rb'))
    
    enc = tiktoken.get_encoding("gpt2")
    count_train = 0
    count_val = 0
    count_test = 0
    for partnum in range(0, 1):

        print(f'starting part {partnum}')

        txtfiles = list()
        txtfiles_val = list()
        bin_file_path = data_dir + f'/val_2pt_p{partnum}.bin'

        if os.path.isfile(bin_file_path):
            continue

        start_pt = (partnum * pt)
        end_pt = ((partnum+1) * pt)

        fundats_fids_2pt_px = statement_fids[start_pt:end_pt]
        for fid in tqdm(fundats_fids_2pt_px):
            if fid in testfids:
                count_test += 1
                continue
            elif fid in valfids:
                #try:
                with open(f'tmp/{fid}', 'w') as f:
                    count_val += 1   
                    statement = statements[fid] 
                    first_hop = "FIRST HOP:\t" + statement['code'] + "\n"
                    statement1 = statement['first']
                    statement2 = statement['second']
                    statement3 = statement['third']
                    prompt = first_hop + f"STATEMENT:<s>\t{statement1}\t{statement2}\t{statement3}</s>"
                    f.write(prompt)
                txtfiles_val.append(f'tmp/{fid}')
            else:
                count_train += 1
                with open(f'tmp/{fid}', 'w') as f:
                    statement = statements[fid] 
                    first_hop = "FIRST HOP:\t" + statement['code'] + "\n"
                    statement1 = statement['first']
                    statement2 = statement['second']
                    statement3 = statement['third']

                    prompt = first_hop + f"STATEMENT:<s>\t{statement1}\t{statement2}\t{statement3}</s>"
                    f.write(prompt)

                txtfiles.append(f'tmp/{fid}')
        if(txtfiles == []):
            dataset = load_dataset('text', data_files={'val':txtfiles_val}, sample_by="document")
        elif(txtfiles_val==[]): 
            dataset = load_dataset('text', data_files={'train': txtfiles}, sample_by="document")
        else:
            dataset = load_dataset('text', data_files={'train': txtfiles, 'val':txtfiles_val}, sample_by="document")

        pickle.dump(dataset, open(f'pkls/dataset_funcom_2pt_p{partnum}.pkl', 'wb'))


        # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
        def process(example):
            #print(example['text'])
            ids = enc.encode(example['text']) # encode_ordinary ignores any special tokens
            ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
            # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
            out = {'ids': ids, 'len': len(ids)}
            return out

        # tokenize the dataset
        tokenized = dataset.map(
            process,
            remove_columns=['text'],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset['len'])
            filename = os.path.join(data_dir, f'{split}_2pt_p{partnum}.bin')
            dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

            print(f"writing {filename}...")
            idx = 0
            for example in tqdm(dset):
                arr[idx : idx + example['len']] = example['ids']
                idx += example['len']
            arr.flush()
    
    bincomb.main('bins/')
    print(f'number of training samples: {count_train}')
    print(f'number of val samples: {count_val}')
    print(f'number of test samples: {count_test}')
    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
