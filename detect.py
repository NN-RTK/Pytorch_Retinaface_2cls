from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.0001, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.4, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(unused_pretrained_keys))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    resize = 1

    # testing begin
    for i in range(1):
        image_path ="./curve/personface_easy1.jpg"#e678caf843ffad4e.jpg"#test.jpg"#a5604603e6e77427.jpg"#5fa9047cf0e2b423.jpg"#65cfc179fad56234.jpg"#
        # "/home/ruchi/work/models/Pytorch_Retinaface-master/data/widerperson/train/images/013709.jpg"
        #
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        #loc, conf, landms = net(img)  # forward pass
        #print(loc, conf, landms)
        loc, conf = net(img)  # forward pass
        print("loc", loc, "\nconf", conf, conf.size())
        conf = F.softmax(conf, dim=-1)#F.sigmoid(conf)##
        #torch.set_printoptions(profile="full")
        print("softmax conf", conf)
        print('net forward time: {:.4f}'.format(time.time() - tic))
        priorbox = PriorBox(cfg, image_size=(im_height, im_width)) 
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        colors = [(0, 0, 255), (0, 255, 255)]
        for class_num in range(1,conf.size(2)):
          scores = conf.squeeze(0).data.cpu().numpy()[:, class_num]
          print(class_num,  conf.squeeze(0).data.cpu().numpy()[:, class_num])
          scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                 img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                 img.shape[3], img.shape[2]])
          scale1 = scale1.to(device)
  
          # ignore low scores
          inds = np.where(scores > args.confidence_threshold)[0]
          print(class_num, "scores", inds, scores[np.argmax(scores)], len(scores), len(inds))
          boxes2 = boxes[inds]
          scores2 = scores[inds]
  
          # keep top-K before NMS
          order = scores2.argsort()[::-1][:args.top_k]
          boxes3 = boxes2[order]
          scores3 = scores2[order]
  
          # do NMS
          dets = np.hstack((boxes3, scores3[:, np.newaxis])).astype(np.float32, copy=False)
          keep = py_cpu_nms(dets, args.nms_threshold)
          # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
          dets = dets[keep, :]
  
          # keep top-K faster NMS
          dets = dets[:args.keep_top_k, :]
          #print(dets)#.shape)
          #dets_fin = np.concatenate(dets, axis=1)

          # show image
          
          for b in dets:
              if b[4] < args.vis_thres:
                  continue
              text = "{:.4f}".format(b[4])#class_num)#
              b = list(map(int, b))
              print("final", b, text)
              cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), colors[class_num-1], 2)
              cx = b[0]
              cy = b[1] + 12
              cv2.putText(img_raw, text, (cx, cy),
                          cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                
          if args.save_image:
            # save image
            name = "test"+ str(class_num) + ".jpg"
            cv2.imwrite(name, img_raw)


#        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
#        priors = priorbox.forward()
#        priors = priors.to(device)
#        prior_data = priors.data
#        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
#        boxes = boxes * scale / resize
#        boxes = boxes.cpu().numpy()
#        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
#        #landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
#        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
#                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
#                               img.shape[3], img.shape[2]])
#        scale1 = scale1.to(device)
#        #landms = landms * scale1 / resize
#        #landms = landms.cpu().numpy()
#
#        # ignore low scores
#        inds = np.where(scores > args.confidence_threshold)[0]
#        boxes = boxes[inds]
#        #landms = landms[inds]
#        scores = scores[inds]
#
#        # keep top-K before NMS
#        order = scores.argsort()[::-1][:args.top_k]
#        boxes = boxes[order]
#        #landms = landms[order]
#        scores = scores[order]
#
#        # do NMS
#        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
#        keep = py_cpu_nms(dets, args.nms_threshold)
#        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
#        dets = dets[keep, :]
#        #landms = landms[keep]
#
#        # keep top-K faster NMS
#        dets = dets[:args.keep_top_k, :]
#        #landms = landms[:args.keep_top_k, :]
#
#        #dets = np.concatenate((dets, landms), axis=1)
#
#        # show image
#        if args.save_image:
#            for b in dets:
#                if b[4] < args.vis_thres:
#                    continue
#                text = "{:.4f}".format(b[4])
#                b = list(map(int, b))
#                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
#                cx = b[0]
#                cy = b[1] + 12
#                cv2.putText(img_raw, text, (cx, cy),
#                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
#
#                # landms
#                #cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
#                #cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
#                #cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
#                #cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
#                #cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
#            # save image
#
#            name = "test.jpg"
#            cv2.imwrite(name, img_raw)

