# ArcFace on Labelled Faces in the Wild (LFW) Dataset 😶 🌐 😱


<p align="center">
  <a href="https://github.com/faridrashidi/kaggle-solutions/blob/gh-pages/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg">

  <a href="https://guides.github.com/features/mastering-markdown/">
    <img src="https://img.shields.io/badge/Language-Markdown-green.svg">
  </a>
</p>

<hr />

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


