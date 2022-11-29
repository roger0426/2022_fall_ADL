# Code for Homework 3 ADL NTU

## Envirement preparing
You can download the code directiry or use git clone the repo.
Instsall requirements.txt in a Python=3.8.0 environment.
```shell
# Download from github (private repo)
git clone https://github.com/roger0426/2022_fall_ADL.git
cd 2022_fall_ADL/hw3
pip install -r requirement.txt
```

**You could pip install specific torch version from Pytorch if you have a newer edition of GPU and get error when execute the code.**
- More details: https://pytorch.org/get-started/locally/
- Eg: for linux, pip
  ```
  pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116
  ```
## Downloading
Download necessary file before execute the program, including model
```shell
bash download.sh
```

## Training
```shell
# set hyperparameters in .sh file
bash hw3_train.sh
```

## Inference
```shell
bash run.sh [input context jsonl file] [output summarization file]
```


## Reference
[cccntu/tw_rouge](https://github.com/cccntu/tw_rouge)  
[Huggingdace mT5_small](https://huggingface.co/google/mt5-small)
[Huggungface sample](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization)  
[Summarization tutorial](https://xiaosheng.run/2022/03/29/transformers-note-8.html)  
