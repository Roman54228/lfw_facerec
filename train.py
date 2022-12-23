import os, shutil, sys
from benedict import benedict
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as TF
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image
from distutils.dir_util import copy_tree
import timm
import torch.nn.functional as F
from torch.nn import Parameter
import math
from tqdm import tqdm
import time
import torchmetrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from arcface import ArcMarginProduct
from utils import testing, get_loss, get_optimizer, get_scheduler
import wandb
import argparse
plt.style.use('ggplot')
torch.manual_seed(42)


def get_backbone(embedding_size):
	backbone = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
	backbone.fc = nn.Linear(2048, embedding_size, bias=True)
	return backbone


def init_dataset(data_root_path, img_size):

	source_path = data_root_path

	pwd = os.getcwd()
	path_to_train = os.path.join(pwd,'train_dir')
	path_to_test = os.path.join(pwd,'test_dir')
	if not os.path.exists(path_to_train):
		os.mkdir(path_to_train)
	if not os.path.exists(path_to_test):
		os.mkdir(path_to_test)

	for dir_path in os.listdir(source_path)[:-1000]:
	    global_path = os.path.join(source_path, dir_path)
	    copy_tree(global_path, f'{path_to_train}/{dir_path}')
	    
	for dir_path in os.listdir(source_path)[-1000:]:
	    global_path = os.path.join(source_path, dir_path)
	    copy_tree(global_path, f'{path_to_test}/{dir_path}')


	train_transform = TF.Compose([TF.ToTensor(), TF.Resize((img_size, img_size))])


	train_dataset = ImageFolder(path_to_train, transform=train_transform)
	train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)

	return train_loader, len(train_dataset.classes)
	


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_root_path', required=True)
	parser.add_argument('--config_path', required=True)             
	parser.add_argument('--wandb', action='store_true')
	args = parser.parse_args()

	path = 'my.yaml'
	cfg = benedict.from_yaml(args.config_path)

	if args.wandb:
		wandb.init(config=cfg)		

	train_loader, number_of_classes = init_dataset(data_root_path=args.data_root_path, 
												   img_size=cfg['img_size'])

	device = torch.device("cuda" if torch.cuda.is_available else "cpu")
	device = 'cpu'

	backbone = get_backbone(cfg['embedding_size'])
	face_head = ArcMarginProduct(cfg['embedding_size'], number_of_classes, s=30, m=0.5)

	backbone.to(device)
	face_head.to(device)


	criterion = get_loss(loss_type=cfg['loss'])
	optimizer = get_optimizer(backbone=backbone,
							  head=face_head,
							  optimizer_type=cfg['optimizer'],
							  lr=cfg['lr'],
							  weight_decay=cfg['weight_decay'],
							  )

	scheduler = get_scheduler(scheduler_type=cfg['scheduler']['scheduler_type'],
							  optimizer=optimizer,
							  schedul_step=cfg['scheduler']['scheduler_step'])

	metric = torchmetrics.Accuracy(num_classes=number_of_classes).to(device)

	txt_pairs = 'lfw_test_matches.txt'
	for epoch in range(cfg['epochs']):
		print(f"Start epoch {epoch}")

		if scheduler is not None:
			scheduler.step()
	    
		running_loss = 0.0
		backbone.train()
		for i, data in enumerate(tqdm(train_loader)):
				data_input, label = data
				data_input = data_input.to(device)
				label = label.to(device).long()
				feature = backbone(data_input)
				output = face_head(feature, label)
				loss = criterion(output, label)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
	            
				running_loss += loss.item()
				train_acc = metric(output, label)
	    
		epoch_loss = running_loss / len(train_loader)        
		train_acc = metric.compute()
		print(f"Accuracy on all data: {train_acc}")
		print(f"Loss on all data: {epoch_loss}")
		wandb.log({'Train/accuracy': train_acc, 'Train/loss': epoch_loss})

		metric.reset()
	    
		if (epoch + 1) % 2 == 0:
			backbone.eval()   
			eer, frr, fpr = testing(backbone, compair_list=txt_pairs, source_path=source_path)
			wandb.log({'Val/eer': eer})
			save_best_model(eer, epoch, backbone, optimizer, criterion)




		

		
		




