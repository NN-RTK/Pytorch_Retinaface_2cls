import torch
from itertools import product as product
import numpy as np
from math import ceil
import math

class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"
        
    def forward(self):
        anchors = []
        aspect_ratios = [2.5,3.33 ]
        count = 0
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
                        #print("count", count, "i, j", i,j, "anchors", anchors[-1], "k,f,min_sizes", k,f,min_sizes)
                        count = count +1
                        # change h/w ratio of the small sized box
                        for ratio in aspect_ratios:
                            ratio = math.sqrt(ratio)
#                            anchors += [cx, cy, s_kx*ratio, s_ky/ratio]
#                            print("count", count, "i, j", i,j, "anchors", anchors[-1], "k,f,min_sizes", k,f,min_sizes)
#                            count = count +1
                            anchors += [cx, cy, s_kx/ratio, s_ky*ratio]
                            #print("count", count, "i, j", i,j, "anchors", anchors[-1], "k,f,min_sizes", k,f,min_sizes)
                            count = count +1
                        

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        #print(output.shape)
        return output
