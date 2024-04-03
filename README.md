# jam_prigen




## RQ2: How closely does the machine prediction match human judgement?

### Compiling training data
```
python3 data/prigen_statement/purpose_advertisement/prepare.py
```

### Compiling test data
```
python3 data/prigen_statement/testdatagen.py
```

### Finetuning
```
CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='2' OMP_NUM_THREADS=2 time torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:4222 --nnodes=1 --nproc_per_node=1 train.py config/finetune_statement_advert.py --outfilename=ckpt_stat_advert.pt
```

### Prediction
```
CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='2' python3 sample_statement.py config/finetune_statement_advert.py --outfilename=ckpt_stat_advert.pt --prediction_filename=statement_advert.txt --testdir=data/prigen_statement/purpose_advertisement/testset/ --max_new_tokens=1024
```
### Metrics
```
python3 accuracy_statement.py
```

### Training prompt
```
CODE:\t<code>\nLABEL:\t<label> STATEMENT:<s>\t<statement1>\t<statement2\>\t<statement3></s>
```

## RQ3: Ablation study

### Compiling training data
```
python3 prepare.py --prigen-file=/nublar/datasets/prigen/prigen_statement/purpose_advertisement/prigen_advertisement_all_allseq.pkl
```
### Compiling test data
```
python3 testdatagen.py --prigendats-file=/nublar/datasets/prigen/prigen_statement/purpose_advertisement/prigen_advertisement_all_allseq.pkl
```
### finetuning
```
CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='2, 3' OMP_NUM_THREADS=2 time torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:4222 --nnodes=1 --nproc_per_node=2 train.py config/finetune_prigen_advert.py --outfilename=ckpt_advert.pt --gradient_accumulation_steps=16
```
### Prediction
```
CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='2' python3 sample_prigen.py config/finetune_prigen_advert.py --outfilename=ckpt_advert.pt --prediction_filename=prigen_advert.txt --testdir=data/advert/testset/ --max_new_tokens=1024 --temperature=0.8
```
### Metrics
```
python3 accuracy_prigen.py 
```
