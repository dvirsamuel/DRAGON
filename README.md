# DRAGON: Long tail learning with attributes
[Paper](http://arxiv.org/abs/2004.02235)  
[Project Website](https://chechiklab.biu.ac.il/~dvirsamuel/DRAGON/)

## Overview
`DRAGON` is a novel modular architecture for long-tail learning, designed to address the "familiarity bias", as described in our paper:
Samuel, Atzmon and Chechik, ["Long tail learning with attributes"](http://arxiv.org/abs/2004.02235).

## Requirements
- numpy  1.15.4
- pandas 0.25.3
- scipy 1.1.0
- tensorflow 1.14.0
- keras 2.2.5

Quick installation under Anaconda:
```
conda env create -f requirements.yml
```

## Data Preparation
Datasets: CUB, SUN and AWA.  
Download `data.tar` from [here](https://chechiklab.biu.ac.il/~dvirsamuel/DRAGON/data.tar), untar it and place it under the **project root directory**.
```
DRAGON
| data
   |--CUB
   |--SUN
   |--AWA1
| attribute_expert
| dataset_handler
| fusion
...
```

## Train Experts and Fusion Module
**Reproduce results for `DRAGON` and its modules (Table 1, rows 7-9 in our paper):**  
Training and evaluation should be according to the training protocol described in our paper (Section 5 - *training*):
1. First, train each expert without the hold-out set (partial training set) by executing the following commands:
    - CUB:
        ```
        # Visual-Expert training
        PYTHONPATH="./" python visual_expert/main.py --base_train_dir=./checkpoints/CUB --dataset_name=CUB --transfer_task=DRAGON --train_dist=dragon --data_dir=data --batch_size=64 --max_epochs=100 --initial_learning_rate=0.0003 --l2=0.005
        # Attribute-Expert training 
        PYTHONPATH="./" python attribute_expert/main.py --base_train_dir=./checkpoints/CUB --dataset_name=CUB --transfer_task=DRAGON --data_dir=data --train_dist=dragon --batch_size=64 --max_epochs=100 --initial_learning_rate=0.001 --LG_beta=1e-7 --LG_lambda=0.0001 --SG_gain=3 --SG_psi=0.01 --SG_num_K=-1
        ```
    - SUN:
        ```
        # Visual-Expert training
        PYTHONPATH="./" python visual_expert/main.py --base_train_dir=./checkpoints/SUN --dataset_name=SUN --transfer_task=DRAGON --train_dist=dragon --data_dir=data --batch_size=64 --max_epochs=100 --initial_learning_rate=0.0001 --l2=0.01
        # Attribute-Expert training 
        PYTHONPATH="./" python attribute_expert/main.py --base_train_dir=./checkpoints/SUN --dataset_name=SUN --transfer_task=DRAGON --data_dir=data --train_dist=dragon --batch_size=64 --max_epochs=100 --initial_learning_rate=0.001 --LG_beta=1e-6 --LG_lambda=0.001 --SG_gain=10 --SG_psi=0.01 --SG_num_K=-1
        ```
    - AWA:
        ```
        # Visual-Expert training
        PYTHONPATH="./" python visual_expert/main.py --base_train_dir=./checkpoints/AWA1 --dataset_name=AWA1 --transfer_task=DRAGON --train_dist=dragon --data_dir=data --batch_size=64 --max_epochs=100 --initial_learning_rate=0.0003 --l2=0.1
        # Attribute-Expert training 
        PYTHONPATH="./" python attribute_expert/main.py --base_train_dir=./checkpoints/AWA1 --dataset_name=AWA1 --transfer_task=DRAGON --data_dir=data --train_dist=dragon --batch_size=64 --max_epochs=100 --initial_learning_rate=0.001 --LG_beta=0.001 --LG_lambda=0.001 --SG_gain=1 --SG_psi=0.01 --SG_num_K=-1
        ```
2. Then, re-train each expert, with the hold-out set (full train set) by executing above commands with the `--test_mode` flag as a parameter.
3. Rename `Visual-lr=0.0003_l2=0.005` to `Visual` and `LAGO-lr=0.001_beta=1e-07_lambda=0.0001_gain=3.0_psi=0.01` to `LAGO` (this is essential since the `FusionModule` finds trained experts by their names, without extensions).
4. Train the fusion-module on trained experts by running the following commands:

    - CUB:
      ```
      PYTHONPATH="./" python fusion/main.py --base_train_dir=./checkpoints/CUB --dataset_name=CUB --data_dir=data --initial_learning_rate=0.005 --batch_size=64 --max_epochs=50 --sort_preds=1 --freeze_experts=1 --nparams=2
      ```
    - SUN:
      ```
      PYTHONPATH="./" python fusion/main.py --base_train_dir=./checkpoints/SUN --dataset_name=SUN --data_dir=data --initial_learning_rate=0.0005 --batch_size=64 --max_epochs=50 --sort_preds=1 --freeze_experts=1 --nparams=4
      ```
    - AWA:
      ```
      PYTHONPATH="./" python fusion/main.py --base_train_dir=./checkpoints/AWA1 --dataset_name=AWA1 --data_dir=data --initial_learning_rate=0.005 --batch_size=64 --max_epochs=50 --sort_preds=1 --freeze_experts=1 --nparams=4
      ```
5. Finally, evaluate the fusion-module with fully-trained experts, by executing step 4 commands with the `--test_mode` flag as a parameter.

## Pre-trained Models and Checkpoints
Download `checkpoints.tar` from [here](https://chechiklab.biu.ac.il/~dvirsamuel/DRAGON/checkpoints.tar), untar it and place it under the **project root directory**.
```
checkpoints
  |--CUB
      |--Visual
      |--LAGO
      |--Dual2ParametricRescale-lr=0.005_freeze=1_sort=1_topk=-1_f=2_s=(2, 2)
  |--SUN
      |--Visual
      |--LAGO
      |--Dual4ParametricRescale-lr=0.0005_freeze=1_sort=1_topk=-1_f=2_s=(2, 2)
  |--AWA1
      |--Visual
      |--LAGO
      |--Dual4ParametricRescale-lr=0.005_freeze=1_sort=1_topk=-1_f=2_s=(2, 2)
```

## Cite Our Paper
If you find our paper and repo useful, please cite:
```
@inproceedings{samuel2020longtail,
  title={Long-tail learning with attributes},
  author={Dvir Samuel and Yuval Atzmon and Gal Chechik},
  booktitle={arXiv preprint arXiv:2004.02235}
  year={2020}}
```