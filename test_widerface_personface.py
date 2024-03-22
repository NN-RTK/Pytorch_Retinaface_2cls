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
from utils.timer import Timer


parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')  
parser.add_argument('--save_folder', default='./wider_openimagespersonface_evaluate/results_wider_OIpersonface_1303txt/', type=str, help='Dir to save txt results') #_openimagesperson
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')

#parser.add_argument('--dataset_folder', default='./data/widerface/val/images', type=str, help='dataset path') #"./data/wider_openimages/val/images/" 
#parser.add_argument('--dataset_folder', default="./data/wider_openimages_all/val/images", type=str, help='dataset path') #
parser.add_argument('--dataset_folder', default="./data/wider_OIpersonface1103/val/images", type=str, help='dataset path') #

parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold') 
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
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
    #print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # testing dataset
    testset_folder = args.dataset_folder
    testset_list = args.dataset_folder[:-6] + "wider_val_list_cleanedBoundary_1203.txt"#"wider_val.txt" #args.dataset_folder[:-11] + "/wider_val.txt"#
    print(testset_list)
    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    # testing begin
    for i, img_name in enumerate(test_dataset):
        image_path = testset_folder + img_name
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)
        print(image_path)
        # testing scale
        target_size = 1600
        max_size = 2150
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if args.origin_size:
            resize = 1

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR) 
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        #_t['forward_pass'].tic()
        #loc, conf, landms = net(img)  # forward pass
        #_t['forward_pass'].toc()
        #_t['misc'].tic()

        _t['forward_pass'].tic()
        loc, conf = net(img)  # forward pass
        conf = F.softmax(conf, dim=-1)#F.sigmoid(conf)##
        _t['forward_pass'].toc()
        _t['misc'].tic()
        #print('net forward time: {:.4f}'.format(time.time() - tic))
        priorbox = PriorBox(cfg, image_size=(im_height, im_width)) 
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        colors = [(0, 0, 255), (0, 255, 255)]
        
        save_name = args.save_folder + img_name[:-4] + ".txt"
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
          os.makedirs(dirname)          
        fd= open(save_name, "w")             
        file_name = os.path.basename(save_name)[:-4] + "\n"          
        fd.write(file_name)
        bboxs_num = str(len(boxes)) + "\n"
        fd.write(bboxs_num)  
        
        for class_num in range(1,conf.size(2)):
          scores = conf.squeeze(0).data.cpu().numpy()[:, class_num]
          #print(class_num,  conf.squeeze(0).data.cpu().numpy()[:, class_num])
          scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                 img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                 img.shape[3], img.shape[2]])
          scale1 = scale1.to(device)
  
          # ignore low scores
          inds = np.where(scores > args.confidence_threshold)[0]
          #print(class_num, "scores", inds, scores[np.argmax(scores)], len(scores), len(inds))
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
          
          # --------------------------------------------------------------------
          # show image          
          #bboxs_num = str(len(dets)) + "\n"
          #fd.write(bboxs_num)
            
          for b in dets:
                if b[4] < args.vis_thres:
                  continue                
                x = int(b[0]) / im_shape[0]
                y = int(b[1])/ im_shape[1]
                w = (int(b[2]) - int(b[0]) )/ im_shape[0]
                h = (int(b[3]) - int(b[1]) )/ im_shape[1]
                confidence = str(b[4])
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " " + str(class_num) + " \n"
                
                fd.write(line)
                
                
                text = "{:.4f}".format(b[4])#class_num)#
                b = list(map(int, b))
                #print("final", b, text)
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), colors[class_num-1], 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))


        # save image
        if not os.path.exists("./results_wider_OIpersonface_1303/"):#_OIperson
            os.makedirs("./results_wider_OIpersonface_1303/") #_openimagesperson#_OIperson
        name = "./results_wider_OIpersonface_1303/" + str(i) + ".jpg"   
        cv2.imwrite(name, img_raw)
    print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))

    # save image
        
            

