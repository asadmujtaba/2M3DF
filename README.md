# 2M3DF
## Introduction

This repository contains the official source **Advanсing 3D Industrial Dеfeсt Dеtесtiоn with Multi реrsреctive Multimodal Fusiоn Nеtwоrk**.

## Requirement

- Ubuntu 18.04

## Environment

- Python >= 3.8

## Packages

- Use requirement.txt file

Other required package(s) can be installed via:

### Install pointnet2_ops_lib
```bash
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```
## Data Download and Preprocess

### Dataset

The **MVTec-3D AD dataset** can be downloaded from the [Official Website of MVTec-3D AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad).

The **Eyecandies dataset** can be downloaded from the [Official Website of Eyecandies](https://eyecan-ai.github.io/eyecandies/).

After downloading, move the dataset to the `data` folder.

### Data Preprocessing

To run the preprocessing for background removal and resizing, run the following command:

```bash
python data_preprocess/preprocessing_mvtec3d.py
```
```bash
python data_preprocess/preprocessing_eyecandies.py
```
To generate multiview RGB images dataset, run the following command:
```bash
python data_preprocess/RGB_multiview_images.py
```

## RUN Model

```bash
ython twom3df_main
```


## Citation

If you find this repository useful for your research, please use the following citation:

```bibtex
@article{asad20252m3df,
  title={2M3DF: Advancing 3D industrial defect detection with multi perspective multimodal fusion network},
  author={Asad, Mujtaba and Azeem, Waqar and Jiang, He and Mustafa, Hafiz Tayyab and Yang, Jie and Liu, Wei},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025},
  publisher={IEEE}
}
```

## Acknowledgement

This repo is based on [CPMF](https://github.com/caoyunkang/CPMF) and [Padim](https://github.com/taikiinoue45/PaDiM), and we thank them for their great work.
