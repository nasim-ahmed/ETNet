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

        c = db_rec['center']
        s = db_rec['scale']
        obj_cat = db_rec['category'].astype(int)

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
                points = fliplrtb_points(
                    points, data_numpy.shape[1], data_numpy.shape[0], self.flip_pairs)

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
            # if points_vis[i, 0] > 0.0:
            points[i, 0:2] = affine_transform(points[i, 0:2], trans)
            points_heatmap[i, 0:2] = affine_transform(points_heatmap[i, 0:2], trans_heatmap)

        target_heat, target_cat = self.generate_target(points_heatmap, obj_cat)

        target_cat = torch.from_numpy(target_cat)
        # target_cat = torch.flatten(target_cat)

        target_heat = torch.from_numpy(target_heat)
        # target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'points': points,
            'score': score,
            'center': c,
            'scale': s,
            'category': obj_cat,
        }

        return input, target_heat, target_cat, meta

    def generate_target(self, points, category):
        '''
        :param points:  [num_points, 3]
        :param points_vis: [num_points, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target_heat = np.zeros((self.num_points,
                                    self.heatmap_size[1],
                                    self.heatmap_size[0]),
                                   dtype=np.float32)

            target_cat = category

            for point_id in range(self.num_points):

                mu_x = points[point_id][0]
                mu_y = points[point_id][1]

                x = np.arange(0, self.heatmap_size[0], 1, np.float32)
                y = np.arange(0, self.heatmap_size[1], 1, np.float32)
                y = y[:, np.newaxis]

                target_heat[point_id] = np.exp(- ((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * self.sigma ** 2))

        return target_heat, target_cat










