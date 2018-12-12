import numpy as np
import cv2
from collections import OrderedDict
import torch
import torch.nn.functional as F
import time
import warnings

from torch.autograd import Variable

from .net import SiamFC
from .config import config
from .transforms import Compose, ToTensor
from .utils import get_exemplar_image, get_pyramid_instance_image, get_instance_image

torch.set_num_threads(1) # otherwise pytorch will take all cpus

class SiamFCTracker:
    def __init__(self, model_path):
        self.model = SiamFC()
        # self.model.load_state_dict(torch.load(model_path))
         
        weight = torch.load(model_path)
        new_state_dict = OrderedDict()
        for k, v in weight.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        
        self.model = self.model.cuda()
        self.model.eval() 
        self.transforms = Compose([
            ToTensor()
        ])

    def _cosine_window(self, size):
        """
            get the cosine window
        """
        cos_window = np.hanning(int(size[0]))[:, np.newaxis].dot(np.hanning(int(size[1]))[np.newaxis, :])
        cos_window = cos_window.astype(np.float32)
        cos_window /= np.sum(cos_window)
        return cos_window

    def init(self, frame, bbox):
        """ initialize siamfc tracker
        Args:
            frame: an RGB image
            bbox: one-based bounding box [x, y, width, height]
        """
        self.bbox = (bbox[0]-1, bbox[1]-1, bbox[0]-1+bbox[2], bbox[1]-1+bbox[3]) # zero based
        self.pos = np.array([bbox[0]-1+(bbox[2]-1)/2, bbox[1]-1+(bbox[3]-1)/2])  # center x, center y, zero based
        self.target_sz = np.array([bbox[2], bbox[3]])                            # width, height
        # get exemplar img
        self.img_mean = tuple(map(int, frame.mean(axis=(0, 1))))
        exemplar_img, scale_z, s_z = get_exemplar_image(frame, self.bbox,
                config.exemplar_size, config.context_amount, self.img_mean)

        # get exemplar feature
        exemplar_img = self.transforms(exemplar_img)[None,:,:,:]
        exemplar_img = exemplar_img.cuda()
        self.exemplar_feats = self.model.init_template(exemplar_img)

        self.penalty = np.ones((config.num_scale)) * config.scale_penalty
        self.penalty[config.num_scale//2] = 1

        # create cosine window
        self.interp_response_sz = config.response_UP * config.response_sz
        self.cosine_window = self._cosine_window((self.interp_response_sz, self.interp_response_sz))

        # create scalse
        self.scales = config.scale_step ** np.arange(np.ceil(config.num_scale/2)-config.num_scale,
                np.floor(config.num_scale/2)+1)

        # create s_x
        self.s_x = s_z + (config.instance_size-config.exemplar_size) / scale_z

        # arbitrary scale saturation
        self.min_s_x = 0.2 * self.s_x
        self.max_s_x = 5 * self.s_x

    def update(self, frame):
        """track object based on the previous frame
        Args:
            frame: an RGB image

        Returns:
            bbox: tuple of 1-based bounding box(xmin, ymin, xmax, ymax)
        """
        size_x_scales = self.s_x * self.scales
        # pyramid = get_pyramid_instance_image(frame, self.pos, config.instance_size, size_x_scales, self.img_mean)
        pyramid = get_pyramid_instance_image(frame, self.pos, size_x_scales, config.num_scale, config.instance_size)
        instance_imgs = torch.cat([self.transforms(x)[None,:,:,:] for x in pyramid], dim=0)
        instance_imgs = instance_imgs.cuda()
        with torch.no_grad():
            response_maps = self.model.forward_corr(self.exemplar_feats, instance_imgs)
        response_maps = response_maps.data.cpu().numpy().squeeze()
        response_maps_up = [cv2.resize(x, (self.interp_response_sz, self.interp_response_sz), cv2.INTER_CUBIC)
             for x in response_maps]
        # get max score
        max_score = np.array([x.max() for x in response_maps_up]) * self.penalty

        # penalty scale change
        scale_idx = max_score.argmax()
        response_map = response_maps_up[scale_idx]
        response_map -= response_map.min()
        response_map /= response_map.sum()
        response_map = (1 - config.w_influence) * response_map + \
                config.w_influence * self.cosine_window
        max_r, max_c = np.unravel_index(response_map.argmax(), response_map.shape)
        # displacement in interpolation response
        disp_response_interp = np.array([max_c, max_r]) - (self.interp_response_sz-1) / 2.
        # displacement in input
        disp_response_input = disp_response_interp * config.total_stride / config.response_UP
        # displacement in frame
        scale = self.scales[scale_idx]
        disp_response_frame = disp_response_input * (self.s_x * scale) / config.instance_size
        # position in frame coordinates
        self.pos += disp_response_frame
        # scale damping and saturation
        self.s_x *= ((1 - config.scale_LR) + config.scale_LR * scale)
        self.s_x = max(self.min_s_x, min(self.max_s_x, self.s_x))
        self.target_sz = ((1 - config.scale_LR) + config.scale_LR * scale) * self.target_sz
        bbox = (self.pos[0] - self.target_sz[0]/2 + 1, # xmin   convert to 1-based
                self.pos[1] - self.target_sz[1]/2 + 1, # ymin
                self.pos[0] + self.target_sz[0]/2 + 1, # xmax
                self.pos[1] + self.target_sz[1]/2 + 1) # ymax
        return bbox
