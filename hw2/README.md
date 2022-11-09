
# Code for Homework 2 ADL NTU

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
- Eg: for linux, pip
  ```
  pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116
  ```
## Downloading
Download necessary file before execute the program
```shell
bash download.sh
```

## Training
- **Context Selection**
```shell
# set hyperparameters in .sh file
bash run_swag_no_trainer.sh
```
- **Span Selection**
```shell
# set hyperparameters in .sh file
bash run_qa_no_trainer.sh
```

## Inference
- **End-to-End**
```shell
bash run.sh [context json file] [test json file] [output file]
```
