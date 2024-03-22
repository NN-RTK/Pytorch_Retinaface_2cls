import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from data.utils import box_utils #import hard_negative_mining


class MultiboxLoss(nn.Module):
    def __init__(self, priors, iou_threshold, neg_pos_ratio,
                 center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss, self).__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)

    
    
    
    
    
    
    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            pos_mask, mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)
        print(confidence)  
        masked_confidence = confidence[mask,:]
        labels_t = np.array([label.tolist() for label in labels])
        masked_labels = torch.tensor(labels_t[pos_mask])#.cpu().numpy().astype(int)])#pos_mask])
        print("mask",  mask.size(), masked_confidence.shape)
        
        print("labels",  masked_labels.size(), masked_confidence.shape)
        #print(np.array(labels_t)[pos_mask])#, torch.tensor(labels))#.size())#[pos_mask.astype(int)])#mask.type(torch.uint8)])
        #classification_loss = F.cross_entropy(masked_confidence.reshape(-1, num_classes), masked_labels, size_average=False)
        classification_loss = F.cross_entropy(confidence.reshape(32, num_classes, -1), masked_labels, size_average=False)
        
        #pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
        num_pos = gt_locations.size(0)
        return smooth_l1_loss/num_pos, classification_loss/num_pos





#        pos = conf_t != zeros
#        conf_t[pos] = 1
#        
#        # Localization Loss (Smooth L1)
#        # Shape: [batch,num_priors,4]
#        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
#        loc_p = loc_data[pos_idx].view(-1, 4)
#        loc_t = loc_t[pos_idx].view(-1, 4)
#        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
#
#        # Compute max conf across batch for hard negative mining
#        batch_conf = conf_data.view(-1, self.num_classes)
#        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
#
#        # Hard Negative Mining
#        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
#        loss_c = loss_c.view(num, -1)
#        _, loss_idx = loss_c.sort(1, descending=True)
#        _, idx_rank = loss_idx.sort(1)
#        num_pos = pos.long().sum(1, keepdim=True)
#        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
#        neg = idx_rank < num_neg.expand_as(idx_rank)
#        # Confidence Loss Including Positive and Negative Examples
#        #print(pos.shape, conf_data.size(), conf_t.size()) 
#        
#        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
#        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
#        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
#        targets_weighted = conf_t[(pos+neg).gt(0)]
#        #print("predicted", conf_p,conf_p.size(), "\ntarget", targets_weighted,targets_weighted.size() )
#        
#        ##** cross entropy loss - original
#        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')
#        ##** Negative likelyhood - RD
#        #m = nn.LogSoftmax(dim=1)
#        #nll_loss = nn.NLLLoss( reduction='sum')
#        #loss_c = nll_loss(m(conf_p), targets_weighted)
#        ##**
#
#        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
#        N = max(num_pos.data.sum().float(), 1)
#        loss_l /= N
#        loss_c /= N
#        #loss_landm /= N1
#        #print(loss_l, loss_c)
#        return loss_l, loss_c#, loss_landm

#def forward(self, predictions, priors, targets):
#        """Multibox Loss
#        Args:
#            predictions (tuple): A tuple containing loc preds, conf preds,
#            and prior boxes from SSD net.
#                conf shape: torch.size(batch_size,num_priors,num_classes)
#                loc shape: torch.size(batch_size,num_priors,4)
#                priors shape: torch.size(num_priors,4)
#
#            ground_truth (tensor): Ground truth boxes and labels for a batch,
#                shape: [batch_size,num_objs,5] (last idx is the label).
#        """
#
#        #loc_data, conf_data, landm_data = predictions
#        loc_data, conf_data = predictions
#        priors = priors
#        num = loc_data.size(0)
#        num_priors = (priors.size(0))
#        # match priors (default boxes) and ground truth boxes
#        loc_t = torch.Tensor(num, num_priors, 4)
#        #landm_t = torch.Tensor(num, num_priors, 10)
#        conf_t = torch.LongTensor(num, num_priors)
#        for idx in range(num):
#            truths = targets[idx*2][:, :4].data
#            labels = targets[idx*2 + 1][:].data
#            ##
#            #truths = targets[idx][:, :4].data
#            #labels = targets[idx][:, -1].data
#            ##
#            #landms = targets[idx][:, 4:14].data
#            defaults = priors.data
#            #match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)
#            match(self.threshold, truths, defaults, self.variance, labels,  loc_t, conf_t,  idx)
#        if GPU:
#            loc_t = loc_t.cuda()
#            conf_t = conf_t.cuda()
#            #landm_t = landm_t.cuda()
#
#        zeros = torch.tensor(0).cuda()
#        # landm Loss (Smooth L1)
#        # Shape: [batch,num_priors,10]
##        pos1 = conf_t > zeros
##        num_pos_landm = pos1.long().sum(1, keepdim=True)
##        N1 = max(num_pos_landm.data.sum().float(), 1)
##        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
##        landm_p = landm_data[pos_idx1].view(-1, 10)
##        landm_t = landm_t[pos_idx1].view(-1, 10)
##        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')
#
#
#        pos = conf_t != zeros
#        conf_t[pos] = 1
#        
#        # Localization Loss (Smooth L1)
#        # Shape: [batch,num_priors,4]
#        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
#        loc_p = loc_data[pos_idx].view(-1, 4)
#        loc_t = loc_t[pos_idx].view(-1, 4)
#        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
#
#        # Compute max conf across batch for hard negative mining
#        batch_conf = conf_data.view(-1, self.num_classes)
#        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
#
#        # Hard Negative Mining
#        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
#        loss_c = loss_c.view(num, -1)
#        _, loss_idx = loss_c.sort(1, descending=True)
#        _, idx_rank = loss_idx.sort(1)
#        num_pos = pos.long().sum(1, keepdim=True)
#        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
#        neg = idx_rank < num_neg.expand_as(idx_rank)
#        # Confidence Loss Including Positive and Negative Examples
#        #print(pos.shape, conf_data.size(), conf_t.size()) 
#        
#        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
#        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
#        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
#        targets_weighted = conf_t[(pos+neg).gt(0)]
#        #print("predicted", conf_p,conf_p.size(), "\ntarget", targets_weighted,targets_weighted.size() )
#        
#        ##** cross entropy loss - original
#        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')
#        ##** Negative likelyhood - RD
#        #m = nn.LogSoftmax(dim=1)
#        #nll_loss = nn.NLLLoss( reduction='sum')
#        #loss_c = nll_loss(m(conf_p), targets_weighted)
#        ##**
#
#        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
#        N = max(num_pos.data.sum().float(), 1)
#        loss_l /= N
#        loss_c /= N
#        #loss_landm /= N1
#        #print(loss_l, loss_c)
#        return loss_l, loss_c#, loss_landm
#
#    
    