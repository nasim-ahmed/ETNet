from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class KeypointMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(KeypointMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_points = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_points, -1)).split(1, 1)

        heatmaps_gt = target.reshape((batch_size, num_points, -1)).split(1, 1)

        loss = 0

        for idx in range(num_points):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_points

class CategoryLoss(nn.Module):
    def __init__(self):
        super(CategoryLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, output, target):
        cat_pred = output
        cat_gt = target

        loss = 0

        loss += 0.5 * self.criterion(cat_pred, cat_gt)

        return loss




