"""
WiderFace evaluation code
author: wondervictor
mail: tianhengcheng@gmail.com
copyright@wondervictor
"""

import os
import tqdm
import pickle
import argparse
import numpy as np
from scipy.io import loadmat
from bbox import bbox_overlaps
from IPython import embed
import glob

def get_gt_boxes(gt_dir):
    """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list']
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list


def get_gt_boxes_from_txt(gt_path, cache_dir):

    #cache_file = os.path.join(cache_dir, 'gt_cache.pkl')
    #if os.path.exists(cache_file):
    #    f = open(cache_file, 'rb')
    #    boxes = pickle.load(f)
    #    f.close()
    #    return boxes

    f = open(gt_path, 'r')
    state = 0
    lines = f.readlines()
    lines = list(map(lambda x: x.rstrip('\r\n'), lines))
    boxes = {}
    faces = {}
    persons = {}
    file_list = []
    #boxes = dict
    box_list = []
    event_list = []
    face_gt_list = []#np.zeros(len(lines))#
    person_gt_list = []# np.zeros(len(lines))#[]#
    
    
    print(len(lines))
    f.close()
    current_boxes = []
    current_name = None
    image_name = []
    count = 0
    for line in lines:
        if line.startswith('#'):
          
          image_name = line[2:]
          path = gt_path.replace('wider_val_gt_cleanedBoundary_1203.txt','images/') + image_name
          #path = gt_path.replace('wider_val_gt.txt','images/') + image_name
          file_list.append(path)
          box_list = []
          face_gt_list = []#np.zeros(len(lines))#
          person_gt_list = []# np.zeros(len(lines))#[]#
          continue
        else:
          line = line.split(' ')
          label = [float(x) for x in line]
          if label[0] == 1:#face
            face_gt_list.append(1)
            person_gt_list.append(0)
          else:
            person_gt_list.append(1)
            face_gt_list.append(0)
          event_list.append(label[0])
          box_list.append(label[1:])
          boxes[image_name.split('.')[0]] = box_list#label[1:]#label[1:]#box_list
          faces[image_name.split('.')[0]] = face_gt_list#
          persons[image_name.split('.')[0]] = person_gt_list#
        count = count +1
        #print(image_name)#, label[1:])
        #print(boxes[image_name.split('.')[0]])
        #break
       
    #f = open(cache_file, 'wb')
    #pickle.dump(boxes, f)
    #f.close()
    return boxes, event_list, file_list, faces, persons


def read_pred_file(filepath):

    #print(filepath)
    f = open(filepath,'r')
    lines = f.readlines()
    isFirst = True
    labels = []
    box_list = []
    event_list = []
    for line in lines:
        line = line.rstrip()
        if isFirst is True:
          isFirst = False
          image_name = line
          #print(image_name)
        else:
          line = line.split(' ')
          if len(line)>1:
            label = [float(x) for x in line]
            #print(label)
            event_list.append(label[5])
            box_list.append(label[:5])
            #box_list.append(label[0])
    return image_name, box_list, event_list


def get_preds(pred_dir):
    events = ['1.0', '2.0']#os.listdir(pred_dir)
    
    #pbar = tqdm.tqdm(events)
    event_images = os.listdir(pred_dir)
    current_event = dict()
    #count = 3
    for imgtxt in event_images:
      imgname, _boxes, event_list = read_pred_file(os.path.join(pred_dir, imgtxt))
      #print(imgname)#, _boxes, event_list)
      #count = count -1
      boxes = dict()
      i = 0
      face_list = []
      person_list = []
    
      if len(event_list)>0:
        for event in event_list:
          if event == 1:
            face_list.append(_boxes[i])
            #print("\nface_list", _boxes[i])
          else:
            person_list.append(_boxes[i])
            #print("\nperson_list", _boxes[i])
          
          i = i + 1
        boxes[events[0]]=face_list
        boxes[events[1]]=person_list
        
      else:
        boxes[events[0]]=[]
        boxes[events[1]]=[]
      #print(boxes)
      current_event[imgname.rstrip('.jpg')] = boxes#_boxes
      #print("current_event", current_event, "\n")
      #if count==0:
      #  break
      
    return current_event#boxes#


def norm_score(pred):
    """ norm score
    pred {key: [[x1,y1,x2,y2,s]]}
    """
    #print(pred)
    max_score = 0
    min_score = 1

    for _, k in pred.items():
        #print(k, "\n")
        for _, v in k.items():
            #print(v, "\n")
            if len(v) == 0:
                continue
            _min = np.min(v[:, -1])
            _max = np.max(v[:, -1])
            max_score = max(_max, max_score)
            min_score = min(_min, min_score)

    diff = max_score - min_score
    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            v[:, -1] = (v[:, -1] - min_score)/diff


def image_eval(pred, gt, ignore, iou_thresh):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """
    #print(pred, gt)
    _pred = pred.copy()
    _gt = np.array(gt.copy())
    pred_recall = np.zeros(len(_pred))#.shape[0])
    recall_list = np.zeros(len(_gt))#.shape[0])
    proposal_list = np.ones(len(_pred))#.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:,1]
    _gt[:, 2] = _gt[:, 2] + _gt[:,0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    for h in range(_pred.shape[0]):

        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):

        thresh = 1 - (t+1)/thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[:r_index+1] == 1)[0]
            pr_info[t, 0] = len(p_index)
            pr_info[t, 1] = pred_recall[r_index]
    return pr_info


def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve


def voc_ap(rec, prec):

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluation(pred_path, gt_path, iou_thresh=0.5):
    print("predictions", pred_path, "\nground truth", gt_path)
    pred = get_preds(pred_path) 
    #print(pred)
    #norm_score(pred)
    #facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)
    facebox_list, event_list, file_list, face_gt_list, person_gt_list = get_gt_boxes_from_txt(gt_path, pred_path)
    event_num = len(file_list)
    thresh_num = 1000
    settings = ['face', 'person']#'easy', 'medium', 'hard']
    setting_gts = [face_gt_list, person_gt_list]#, hard_gt_list]
    aps = []
    for setting_id in range(2):                                                         # for each class
        # different setting
        gt_list = setting_gts[setting_id]
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2)).astype('float')
        # [hard, medium, easy]
        pbar = tqdm.tqdm(range(event_num))
        for i in pbar:                                                                   # for each image, get name, get class boxes mask,gt boxes list
            pbar.set_description('Processing {}'.format(settings[setting_id]))
            img_list = file_list[i]#[0]# get name
            image_name = img_list.split('/')[-1].split('.')[0]
            #print(img_list, image_name)
            sub_gt_list = gt_list[image_name]#[i][0]# get 
            # img_pr_info_list = np.zeros((len(img_list), thresh_num, 2))
            gt_bbx_list = facebox_list[image_name]#[0]
            ignore = np.zeros(len(gt_bbx_list))#.shape[0])
            pred_info = np.array(pred[image_name][str(float(setting_id+1))])#pred_list[str(img_list[j][0][0])]
            #print(image_name,  len(gt_bbx_list), pred[image_name], len(pred_info))#gt_bbx_list[0],
            
            #for j in range(len(gt_bbx_list)):                                           # for each gt box, get all predoctions 
                
            #gt_boxes = gt_bbx_list[j]#.astype('float')#[0].astype('float')
            #keep_index = sub_gt_list[j]#[0]
            count_face += np.sum(sub_gt_list)#keep_index#

            if len(gt_bbx_list) == 0 or len(pred_info) == 0:
              continue
                
            #if keep_index != 0:
            #  ignore[keep_index-1] = 1
            #print(len(pred_info), len(gt_bbx_list), len(sub_gt_list))
            pred_recall, proposal_list = image_eval(pred_info, gt_bbx_list, sub_gt_list, iou_thresh)

            _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)

            pr_curve += _img_pr_info
        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)

        propose = pr_curve[:, 0]
        recall = pr_curve[:, 1]

        ap = voc_ap(recall, propose)
        aps.append(ap)

    print("==================== Results ====================")
    print("Face   Val AP: {}".format(aps[0]))
    print("Person Val AP: {}".format(aps[1]))
    #print("Hard   Val AP: {}".format(aps[2]))
    print("=================================================")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred', default="/home/ruchi/work/models/Pytorch_Retinaface-master/wider_openimagespersonface_evaluate/results_wider_OIpersonface_150_1803txt/")#results_widerOIpersonface_3002txt/") #wider_openimagespersonface_2k_rescaletxt/")##results_widerOIpersonface_2902_320txt
    parser.add_argument('-g', '--gt', default="/home/ruchi/work/models/Pytorch_Retinaface-master/data/wider_OIpersonface1103/val/wider_val_gt_cleanedBoundary_1203.txt")
    #parser.add_argument('-g', '--gt', default="/home/ruchi/work/models/Pytorch_Retinaface-master/data/wider_openimages_all/val/wider_val_gt.txt")

    args = parser.parse_args()
    evaluation(args.pred, args.gt)












