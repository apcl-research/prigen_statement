# Code for replication of What Code Statements Affect Privacy?

## To-do list
- To set up your local environment, run the following command. We recommend the use of a virtual environment for running the experiments.

```
pip install -r requirements.txt
```
- Please download the dataset from [link]()

## Compiling dataset
We also release all of our raw datasets for the experiments in [link]() and the scripts for compiling the raw data to bin files in this Github repo. Before running the command, please create three dir: pkls, bins, and tmp. Then, you can simply run the following command to generate train.bin and val.bin.

```
python3 data/statement_labels/prepare.py
```
- Note that you will need to place ``testfid.pkl``, ``valfids.pkl`` and  ``unique_response_filter.pkl`` on ``/nublar/datasets/prigen/prigen_statement/new_data/`` or you will need to change the related parameters in the script.
- Related parameters are as follows:
```
  --testfids-file: file lcation of function id on testset
  --valfids-file: file location of function id on valset
  --statement-file: file location of statements
```

## Finetuning
