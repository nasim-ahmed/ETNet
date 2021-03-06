# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class pointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(pointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target):

        batch_size = output.size(0)
        num_points = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_points, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_points, -1)).split(1, 1)
        loss = 0

        for idx in range(num_points):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_points


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, output, target):
        cat_pred = output
        cat_gt = target

        loss = 0

        loss += 0.5 * self.criterion(cat_pred, cat_gt)

        return loss

# class pointsOHKMMSELoss(nn.Module):
#     def __init__(self, use_target_weight, topk=8):
#         super(pointsOHKMMSELoss, self).__init__()
#         self.criterion = nn.MSELoss(reduction='none')
#         self.use_target_weight = use_target_weight
#         self.topk = topk
#
#     def ohkm(self, loss):
#         ohkm_loss = 0.
#         for i in range(loss.size()[0]):
#             sub_loss = loss[i]
#             topk_val, topk_idx = torch.topk(
#                 sub_loss, k=self.topk, dim=0, sorted=False
#             )
#             tmp_loss = torch.gather(sub_loss, 0, topk_idx)
#             ohkm_loss += torch.sum(tmp_loss) / self.topk
#         ohkm_loss /= loss.size()[0]
#         return ohkm_loss
#
#     def forward(self, output, target):
#         batch_size = output.size(0)
#         num_points = output.size(1)
#         heatmaps_pred = output.reshape((batch_size, num_points, -1)).split(1, 1)
#         heatmaps_gt = target.reshape((batch_size, num_points, -1)).split(1, 1)
#
#         loss = []
#         for idx in range(num_points):
#             heatmap_pred = heatmaps_pred[idx].squeeze()
#             heatmap_gt = heatmaps_gt[idx].squeeze()
#             loss.append(
#                     0.5 * self.criterion(heatmap_pred, heatmap_gt)
#                 )
#
#         loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
#         loss = torch.cat(loss, dim=1)
#
#         return self.ohkm(loss)
