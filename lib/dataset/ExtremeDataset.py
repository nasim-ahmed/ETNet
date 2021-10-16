# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Hanbin Dai (daihanbin.ac@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplrtb_points


logger = logging.getLogger(__name__)

class ExtremeDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_points = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set
        
        self.categories = 80
        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP

        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_points_weight = cfg.LOSS.USE_DIFFERENT_POINTS_WEIGHT
        self.points_weight = 1

        self.transform = transform
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def __len__(self, ):
        return len(self.db)

    def __getitem__(self, idx):

        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']

        #print('THIS IS THE IMAGE INDEX {0}'.format(image_file))
       # print(image_file)

        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        points = db_rec['points_4p']
        points_vis = db_rec['points_4p_vis']

        c = db_rec['center']
        s = db_rec['scale']
        obj_cat = db_rec['category'].astype(int)
        #print('THIS IS THE CATEGORY: {0}'.format(obj_cat))

        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0
           
            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                points, points_vis = fliplrtb_points(
                    points, points_vis, data_numpy.shape[1], data_numpy.shape[0], self.flip_pairs)
        
        points_heatmap = points.copy()
        trans = get_affine_transform(c, s, r, self.image_size)
        trans_heatmap = get_affine_transform(c, s, r, self.heatmap_size)

        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)

        for i in range(self.num_points):
            if points_vis[i, 0] > 0.0:
                points[i, 0:2] = affine_transform(points[i, 0:2], trans)
                points_heatmap[i, 0:2] = affine_transform(points_heatmap[i, 0:2], trans_heatmap)

        target_heat, target_weight, target_cat = self.generate_target(points_heatmap, points_vis, obj_cat)

        target_cat = torch.from_numpy(target_cat)
        target_cat = torch.flatten(target_cat)

        target_heat = torch.from_numpy(target_heat)
        target_weight = torch.from_numpy(target_weight)
        #print('THIS IS THE DIFFERENT POINTS:{0}'.format(points))

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'points': points,
            'points_vis': points_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score,
            'category': obj_cat,
           }

        return input, target_heat, target_weight, target_cat, meta

    def generate_target(self, points, points_vis, category):
        '''
        :param points:  [num_points, 3]
        :param points_vis: [num_points, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_points, 1), dtype=np.float32)
        target_weight[:, 0] = points_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target_heat = np.zeros((self.num_points,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            #one-hot-encoding

            # target_cat = np.eye(self.categories)[category-1]
            target_cat = category

            tmp_size = self.sigma * 3

            for point_id in range(self.num_points):
                target_weight[point_id] = \
                    self.adjust_target_weight(points[point_id], target_weight[point_id], tmp_size)

                # if target_weight[point_id] == 0:
                #     continue

                mu_x = points[point_id][0]
                mu_y = points[point_id][1]

                x = np.arange(0, self.heatmap_size[0], 1, np.float32)
                y = np.arange(0, self.heatmap_size[1], 1, np.float32)
                y = y[:, np.newaxis]
                
                #category = points[point_id][2].astype(int)
                
                v = target_weight[point_id]

                if v > 0.5:
                    target_heat[point_id] = np.exp(- ((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * self.sigma ** 2))

        if self.use_different_points_weight:
             target_weight = np.multiply(target_weight, self.points_weight)

        return target_heat, target_weight, target_cat

    def adjust_target_weight(self, point, target_weight, tmp_size):
        # feat_stride = self.image_size / self.heatmap_size
        mu_x = point[0]
        mu_y = point[1]
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight = 0

        return target_weight










