import torch
import torch.nn as nn
import numpy as np 
import os
import cv2 
from sklearn.metrics import roc_curve

def get_lfw_list(pair_list, source_path):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_dist = {}
    i = 0
    for pair in pairs:
        to_append = []
        path1, path2, label = pair.split()
        path1, path2 = [os.path.join(source_path, each) for each in (path1, path2)]
        
        new_pair = [path1, path2, int(label)]
        data_dist[i] = new_pair
        
        i += 1
        
    return data_dist


def load_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (IMAGE_SIZE, IMAGE_SIZE))
    image = torch.tensor(image).permute(2,0,1).unsqueeze(0) / 255
    
    if image is None:
        return None

    return image


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def test_performance(fe_dict, pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        fe_1 = fe_dict[splits[0]]
        fe_2 = fe_dict[splits[1]]
        label = int(splits[2])
        sim = cosin_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th


def lfw_test(model, img_paths, identity_list, compair_list, batch_size):
    s = time.time()
    features, cnt = get_featurs(model, img_paths, batch_size=batch_size)
    print(features.shape)
    t = time.time() - s
    print('total time is {}, average time is {}'.format(t, t / cnt))
    fe_dict = get_feature_dict(identity_list, features)
    acc, th = test_performance(fe_dict, compair_list)
    print('lfw face verification accuracy: ', acc, 'threshold: ', th)
    return acc


def testing(model, compair_list, source_path):
    cos = nn.CosineSimilarity()
    device = next(model.parameters()).device
    
    test_dict = get_lfw_list(compair_list, source_path)
    results_dict = {}
    
    scores = []
    targets = []
    
    for idx in tqdm(test_dict):
        pair = test_dict[idx]
        img1 = load_image(pair[0])
        img2 = load_image(pair[1])
        label = pair[2]
        
        
        tensr = torch.cat((img1, img2), 0).to(device)
        
        with torch.no_grad():
            pred = model(tensr)
            embed1, embed2 = pred
            
        score = cos(embed1.unsqueeze(0), embed2.unsqueeze(0))
        
        scores.append(score.cpu().item())
        targets.append(label)
        
    fpr, tpr, thresholds = roc_curve(targets, scores, pos_label=1)
    frr = 1 - tpr
    eerv = np.argmin(np.abs(frr - fpr))
    eer = np.mean([frr[eerv], fpr[eerv]])
    
               
    return eer, frr, fpr


def get_loss(loss_type):
    if loss_type == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif loss_type == 'FocalLoss':
        return nn.FocalLoss()
    else:
        raise ValueError("Wrong LOSS NAME in config. Choose from 'CrossEntropyLoss', 'FocalLoss'")


def get_optimizer(backbone, head, optimizer_type, lr, weight_decay):
    if optimizer_type == 'SGD':
        return torch.optim.SGD([{'params': backbone.parameters()}, {'params': head.parameters()}], 
                               lr=float(lr),
                               weight_decay=float(weight_decay))
    if optimizer_type == 'Adam':
        return torch.optim.SGD([{'params': backbone.parameters()}, {'params': head.parameters()}], 
                               lr=float(lr),
                               weight_decay=float(weight_decay))
    else:
        raise ValueError("Wrong OPIMIZER NAME in config. Choose from 'SGD', 'Adam'")


def get_scheduler(scheduler_type, optimizer, schedul_step):
    if scheduler_type == 'linear':
        return None#torch.optim.lr_scheduler.LinearLR(optimizer, 
                #                                      step_size=int(schedul_step), 
                 #                                     gamma=0.1)
    else: 
        return None







