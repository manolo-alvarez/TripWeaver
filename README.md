# TripWeaver

## Setup

Install dependencies:
```sh
pip install -r requirements.txt
```

## Training
SFT Training:
```sh
python model/sft.py
```
DPO Training:
```sh
python model/dpo.py
```
You can monitor the results via tensorboard:
```sh
tensorboard --logdir=experiments
```

## Debug in VSCode
Start debug session with launch.json configuration file in .vscode/
