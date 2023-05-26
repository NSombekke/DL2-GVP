# DL2-GVP
- Luka Wong Chung
- Wouter Haringhuizen
- Erencan Tatar

Read the report on [our Blogpost](./Blogpost.md).


-----

<code to run files and jobs >  .job files

    
    ## Introduction

This repo uses the original Geometric Vector Perceptrons [GVP](https://github.com/drorlab/gvp-pytorch/tree/main), a rotation-equivariant GNN, and combines a TransformerConv layer attenmpting to further improve the the baseline model.
Scripts for training / testing / sampling on protein design and training / testing on all ATOM3D tasks are provided.
Trained model are provided in the foler "TODO".
We provide a blogpost.md and GVP.ipynb walking through all the experiments.

## Table Of Content
- [Installation](#installation)
    - [Requirements](#composer)
    - [Environment](#Environment)
    - [Download Atom3D dataset](#download-dataset)
- [Run experiments](#Run-experiments)
    - [aa](#database-setup)
- [Links](#links)

## Installation
For general usage and deeper understand of individual files we refer back to the original readme [GVP](https://github.com/drorlab/gvp-pytorch/tree/main)
### Requirements


```bash
conda create -n gvp
source activate gvp

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg
conda install pytorch-scatter -c pyg
conda install pytorch-cluster -c pyg
conda install pytorch-sparse -c pyg
conda install tqdm numpy scikit-learn
pip install atom3d  
```
***Note***: The original atom3d module might contain a error regarding 'get_subunits' and 'get_negatives' from the subpackage `neighbors`

### Environment
In order to test a correct install run  `test_equivariance.py`

### Download-dataset
In order to download the ATOM3d datasets with splits we use the `download_atomdataset.job`. Running this file downloads the necesarry datasets by running `download_atom3d.py` and saves them into the right directories.

## Run-experiments
### In order to train the model on the ATOM3D datasets, we used the `run_atom3d.job` file in which we called `run_atom3d.py` for the different datasets and on different seeds. The usage of `run_atom3d.py` is as follows:
```
 $ python run_atom3d.py -h

usage: run_atom3d.py [-h] [--num-workers N] [--smp-idx IDX]
                     [--lba-split SPLIT] [--batch SIZE] [--train-time MINUTES]
                     [--val-time MINUTES] [--epochs N] [--test PATH]
                     [--lr RATE] [--load PATH]
                     TASK

positional arguments:
  TASK                  {RES, MSP, SMP, LBA, LEP}

```


## Links




* [Web site](https://aimeos.org/integrations/typo3-shop-extension/)
* [Documentation](https://aimeos.org/docs/TYPO3)
* [Forum](https://aimeos.org/help/typo3-extension-f16/)
* [Issue tracker](https://github.com/aimeos/aimeos-typo3/issues)
* [Source code](https://github.com/aimeos/aimeos-typo3)
