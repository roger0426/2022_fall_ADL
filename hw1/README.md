
# Code for Homework 1 ADL NTU

## Envirement preparing
You can download the code directiry or use git clone the repo.
Instsall requirements.txt in a Python=3.9.0 environment.
```shell
# Download from github
git clone https://github.com/roger0426/2022_fall_ADL.git
cd 2022_fall_ADL/hw1
pip install -r requirement.in
```

**You could pip install specific torch version from Pytorch if you have a newer edition of GPU and get error when execute the code.**
- More details: https://pytorch.org/get-started/locally/
- Eg: pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116

## Downloading
Download necessary file before execute the program
```shell
bash download.sh
```

## Preprocessing
~~Preprocess intent classification and slot tagging datasets.~~
~~bash preprocess.sh~~

## Training
- **Intent detection**
```shell
# set hyperparameters in .py file
python3.9 train_intent.py
```
- **Sequence tagging**
```shell
# set hyperparameters in .py file
python3.9 train_slot.py
```

## Inference
- **Intent detection**
```shell
bash ./intent_cls.sh [test_data_path] [output_pred_path]
```
- **Sequence tagging**
```shell
bash ./slot_tag.sh [test_data_path] [output_pred_path]
```
