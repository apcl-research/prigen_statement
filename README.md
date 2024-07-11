# Code for replication of Which Code Statements Implement Privacy Behaviors in Android Applications?

## Quick link
- [To-do list](#to-do-list)
- [Compiling dataset](#compiling-dataset)
- [Finetuning](#finetuning)
- [Inference](#inference)
- [Metrics](#metrics)

## To-do list
- To set up your local environment, run the following command. We recommend the use of a virtual environment for running the experiments.

```
pip install -r requirements.txt
```
- Please download the dataset from [link](https://drive.google.com/drive/folders/17Us2z-_Qfe20B4APrO1Ta4WB39HBSUsw?usp=drive_link)


## Compiling dataset
We also release all of our raw datasets for the experiments in ``raw_data.tar.gz`` [link](https://drive.google.com/drive/folders/17Us2z-_Qfe20B4APrO1Ta4WB39HBSUsw?usp=drive_link) and the scripts for compiling the raw data to bin files in this Github repo. Before running the command, please create three dir: ``pkls``, ``bins``, and ``tmp``. Then, you can simply run the following command to generate ``train.bin`` and ``val.bin``.

```
python3 data/statement_labels/prepare.py
```
- Note that you will need to place ``testfid.pkl``, ``valfids.pkl`` and  ``train.pkl`` on ``/nublar/datasets/prigen/prigen_statement/new_data/`` or you will need to change the related parameters in the script.
- Related parameters are as follows:
```
  --testfids-file: file lcation of function id on testset
  --valfids-file: file location of function id on valset
  --statement-file: file location of statements
```

## Finetuning
These steps will show you how to fine-tune the model for statement prediction.

### Step 1: Download the finetuning dataset
You can download all of the datasets in our paper in the [link](https://drive.google.com/drive/folders/17Us2z-_Qfe20B4APrO1Ta4WB39HBSUsw?usp=drive_link). Please place ``train.bin`` and ``val.bin`` to the same dir as ``--dataset`` in ``config/finetune_statement_advert.py``.

### Step 2: Download the models for finetuning
Please download the checkpoint files named ``ckpt_pretrain.pt`` in the [link](https://drive.google.com/drive/folders/17Us2z-_Qfe20B4APrO1Ta4WB39HBSUsw?usp=drive_link) for finetuning and place the checkpoint to the same dir as ``--out_dir`` in ``config/finetune_statement_advert.py``.

### Step 3: Finetuning model
```
CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='0,1' OMP_NUM_THREADS=2 time torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:4000 --nnodes=1 --nproc_per_node=2 train.py config/finetune_statement_advert.py --outfilename=ckpt_pretrain.pt --gradient_accumulation_steps=16
```

## Inference
After you download the test set named ``testset.tar.gz`` in the [link](https://drive.google.com/drive/folders/17Us2z-_Qfe20B4APrO1Ta4WB39HBSUsw?usp=drive_link), you can simiply run command below for inference.
```
CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='0' python3 sample_statement.py config/finetune_statement_advert.py --outfilename=ckpt_pretrain.pt --prediction_filename=statement.txt --testdir=data/statement_labels/testset/ --max_new_tokens=256
```
The script for inference with some parameters that you can change:
```
--outfilename: file name of the model
--prediction_filename: file name of the prediction file
--testdir: directory of the test set
```
## Metrics
We also provide the script for computing the metrics that we report in the paper.

### Accuracy without order
```
python3 accuracy_statement.py --input={filename_of_your_prediction_file}
```

### Accuracy without order (at least one statement matching on two different participants)
```
python3 accuracy_statement_overall.py --input={filename_of_your_prediction_file}
```
### Accuracy in order
```
python3 accuracy_statement_order.py --input={filename_of_your_prediction_file}
```
### Accuracy in order (at least one statement matching on two different participants)
```
python3 accuracy_statement_order_overall.py --input={filename_of_your_prediction_file}
```
Those scripts come with some parameters that you can change:
```
--input: file name of the prediction file
--prigen-file: file name of reference file
```
