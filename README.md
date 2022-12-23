# ArcFace on Labelled Faces in the Wild (LFW) Dataset 😶 🌐 😱

Related paper: https://arxiv.org/abs/1801.07698

Table of Contents
* [Installation](#installation)
* [Scripts](#scripts)
  * [Training Experiment](#training)
  * [download_data.sh](#download)

## Installation
```bash
pip3 install -r requirements.txt
```

## Scripts 

### Training 

```bash
python3 train.py --config experiment_0.yaml --data_root_path data/lfw-deepfunneled --wandb
```


