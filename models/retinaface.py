import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

from models.net import MobileNetV1 as MobileNetV1
from models.net import FPN as FPN
from models.net import SSH as SSH



class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        num_classes = 2
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*(num_classes+1),kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        num_classes = 2
        return out.view(out.shape[0], -1, (num_classes+1))

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace,self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                #checkpoint = torch.load("./weights/mb1-ssd-Epoch-99-Loss-4.6495876759290695.pth", map_location=torch.device('cpu'))
                checkpoint = torch.load("./weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                
#                for k, v in checkpoint.items():#checkpoint['state_dict'].items():
#                    #print(k)
#                    k_split = k.split('.')
#                    if 'base_net' in k_split[0]:
#                      if int(k_split[1])<6:
#                        name = k.replace('base_net', 'stage1')#
#                        new_state_dict[name] = v
#                        #print( name)
#                      elif int(k_split[1])<12:
#                        k_modified = k_split[0] + '.' + str(int(k_split[1])-6) + '.' + k_split[2] + '.' + k_split[3]
#                        name = k_modified.replace('base_net', 'stage2')# 
#                        new_state_dict[name] = v
#                        #print(name)
#                      elif int(k_split[1])<15:
#                        k_modified = k_split[0] + '.' + str(int(k_split[1])-12) + '.' + k_split[2] + '.' + k_split[3]
#                        name = k_modified.replace('base_net', 'stage3')#
#                        new_state_dict[name] = v
#                        #print(name)
#                    #elif 'extras.0.0' in k:
#                    #    name = k.replace('extras.0.0', 'fc')
#                    #    new_state_dict[name] = v
#                    else:
#                      name = k
                    
                for k, v in checkpoint['state_dict'].items():  
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                    #print(k, name)
                # load params
                #print(new_state_dict)
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        #if self.phase == 'train':
        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        #self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])
        #else:
          #self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
          #self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
          
    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=6):#10):#
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=6):#10):#2):#6
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self,inputs):
        out = self.body(inputs)
        #print("inputs and backboneouts" )
        # FPN
        fpn = self.fpn(out)
        #print("fpn finished test")
        # SSH
        feature1 = self.ssh1(fpn[0])
        #print("features1 finished test")
        feature2 = self.ssh2(fpn[1])
        #print("features2 finished test")
        feature3 = self.ssh3(fpn[2])
        #print("features3 finished test")
        features = [feature1, feature2, feature3]
        #print("features", [feature1[:,0], feature2[:,0], feature3[:,0]])
        if self.phase == 'train':
          bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
          classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
          #ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)
          #print("bbx and clasifications finished ldm finished")
        else:
          bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
          classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
          #ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)
          #print("bbx and clasifications finished")
        if self.phase == 'train':
            
            output = (bbox_regressions, classifications)#, ldm_regressions)
        else:
            #output = (bbox_regressions, F.softmax(classifications, dim=-1))#, ldm_regressions)
            output = (bbox_regressions, classifications)
        #print("classifications", classifications)
        return output