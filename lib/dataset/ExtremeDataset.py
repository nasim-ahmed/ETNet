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
from utils.transforms import fliplr_points


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
        self.num_points_half_body = cfg.DATASET.NUM_points_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_points_weight = cfg.LOSS.USE_DIFFERENT_points_WEIGHT
        self.points_weight = 1

        self.transform = transform
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def center_scale_det(self, points):
        center = points.mean(axis=0)[:2]

        left_top = np.amin(points, axis=0)
        right_bottom = np.amax(points, axis=0)

        w = right_bottom[0] - left_top[0] + 1
        h = right_bottom[1] - left_top[1] + 1

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ], dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def __len__(self,):
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

        points = db_rec['points_3d']

        c = db_rec['center']
        s = db_rec['scale']
        obj_cat = db_rec['category'].astype(int)
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            c_points, s_points = self.center_scale_det(points)

            if c_points is not None and s_points is not None:
                c, s = c_points, s_points

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                points = fliplr_points(
                    points, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1
                
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

            points[i, 0:2] = affine_transform(points[i, 0:2], trans)
            points_heatmap[i, 0:2] = affine_transform(points_heatmap[i, 0:2], trans_heatmap)

        target, target_cat = self.generate_target(points_heatmap, obj_cat)

        target_cat = torch.from_numpy(target_cat).long()
        target = torch.from_numpy(target)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'points': points,
            'category': obj_cat,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }

        return input, target, target_cat, meta

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            points_x = 0.0
            points_y = 0.0
            for point in zip(rec['points_3d']):

                num_vis += 1

                points_x += point[0]
                points_y += point[1]

            points_x, points_y = points_x / 4, points_y / 4

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            points_center = np.array([points_x, points_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((points_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * 4 + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected


    def generate_target(self, points, category):
        '''
        :param points:  [num_points, 3]
        :param points_vis: [num_points, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_points,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)
            #one-hot
            target_cat = np.eye(self.categories)[category]

            for point_id in range(self.num_points):

                mu_x = points[point_id][0]
                mu_y = points[point_id][1]
                
                x = np.arange(0, self.heatmap_size[0], 1, np.float32)
                y = np.arange(0, self.heatmap_size[1], 1, np.float32)
                y = y[:, np.newaxis]

                target[point_id] = np.exp(- ((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * self.sigma ** 2))

        return target, target_cat


