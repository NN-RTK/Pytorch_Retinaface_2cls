from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import logging
import torch.utils.data as data
import numpy as np
from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from data.ssd.ssd import MatchPrior
from data.ssd.data_preprocessing import TrainAugmentation, TestTransform
from data.open_images import OpenImagesDataset
from data.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.retinaface import RetinaFace

parser = argparse.ArgumentParser(description='Retinaface Training')
#parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', help='Training dataset directory')#
#parser.add_argument('--training_dataset', default='./data/widerperson/train/label_person_050324.txt', help='Training dataset directory')#
parser.add_argument('--training_dataset', default='./data/wider_OIpersonface1103/train/wider_train_gt.txt', help='Training dataset directory')#
#parser.add_argument('--training_dataset', default="/home/ruchi/data/open_images/", help='Training dataset directory')


parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50

rgb_mean = (104, 117, 123) # bgr order
num_classes = 3#2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']
variance = cfg['variance']
num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
print("load training dataset", training_dataset)
save_folder = args.save_folder

net = RetinaFace(cfg=cfg)
def make_custom_object_detection_model_fcos(num_classes):
    model = torchvision.models.detection.fcos_resnet50_fpn(pretrained=True)  # load an object detection model pre-trained on COCO
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.num_classes = num_classes
    out_channels = model.head.classification_head.conv[9].out_channels
    cls_logits = torch.nn.Conv2d(out_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
    torch.nn.init.normal_(cls_logits.weight, std=0.01)
    torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))

    model.head.classification_head.cls_logits = cls_logits

    return model

print("Printing net...")
print(net)

if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net).cuda()
else:
    net = net.cuda()

cudnn.benchmark = True


optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
#optimizer = optim.Adam(net.parameters(), lr=initial_lr)#, momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.45, True, 0, True, 7, 0.30, False)
priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()

def train():
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...', training_dataset)
    #******************
    if 'wider' in training_dataset:      
      dataset = WiderFaceDetection( training_dataset,preproc(img_dim, rgb_mean))
      train_loader = data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate)
    #******************  
    elif 'open_images' in training_dataset:
      image_mean = np.array([127, 127, 127])  # RGB layout
      train_transform = TrainAugmentation(img_dim, image_mean, 128)
      #target_transform = MatchPrior(priors, variance[0],variance[1], 0.5)
      datasets = []
      for dataset_path in training_dataset:
        dataset = OpenImagesDataset(dataset_path,transform=train_transform, target_transform=None, dataset_type="train", balance_data=True)
        label_file = os.path.join(save_folder, "open-images-model-labels.txt")
        store_labels(label_file, dataset.class_names)
        logging.info(dataset)
        num_classes = len(dataset.class_names)
        #print(num_classes)
        datasets.append(dataset)
      train_dataset = data.ConcatDataset(datasets)
      train_loader = data.DataLoader(train_dataset, batch_size,num_workers=num_workers,shuffle=True,  collate_fn=detection_collate)
    #******************

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size#1#

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size) 
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):#1):#
        if iteration % epoch_size == 0:
            # create batch iterator
            #batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
            batch_iterator = iter(train_loader)
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                torch.save(net.state_dict(), save_folder + cfg['name']+ '_2cls6anchors_130324_epoch_' + str(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        #print("images*********************\n", images)
        #print("targets*********************\n")#, targets)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]
        
        # forward
        out = net(images)
        
        # backprop
        optimizer.zero_grad()
        #loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss_l, loss_c = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c# + loss_landm
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        #print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
        #      .format(epoch, max_epoch, (iteration % epoch_size) + 1,
        #      epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), lr, batch_time, str(datetime.timedelta(seconds=eta)))) 
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f}  || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
              .format(epoch, max_epoch, (iteration % epoch_size) + 1,
              epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(),  lr, batch_time, str(datetime.timedelta(seconds=eta))))

    torch.save(net.state_dict(), save_folder + cfg['name'] + '_2cls6anchors_130324.pth')
    # torch.save(net.state_dict(), save_folder + 'Final_Retinaface.pth') 


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    train()
