# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Hanbin Dai (daihanbin.ac@gmail.com) and Feng Zhang (zhangfengwcy@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.vis import save_debug_images
from utils.transforms import flip_back

logger = logging.getLogger(__name__)

def _topk_class(scores, K=1):
    #same as our cat_out dimension
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.reshape(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()
    
    return topk_clses, topk_scores 

def train(config, train_loader, model, criterion, criterion2, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    point_acc = AverageMeter()


    # switch to train mode
    model.train()
    #cat_criterion = CrossEntropyLoss()

    end = time.time()
    for i, (input, target_heat, target_weight, target_cat, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        heat_out, cat_out = model(input)
        target_heat = target_heat.cuda(non_blocking=True)
        target_cat = target_cat.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(heat_out, list):
            loss = criterion(heat_out[0], target_heat, target_weight)
            for output in heat_out[1:]:
                loss += criterion(output, target_heat, target_weight)
        else:
            output_heat = heat_out
            #output_cat = classes.type(torch.cuda.FloatTensor)

            #target_new = torch.max(target_cat, 1)[1]
            loss_h = criterion(output_heat, target_heat, target_weight)
            #print('This is target cat:{}'.format(target_cat.shape))
            loss_c = criterion2(cat_out, torch.squeeze(target_cat))
            #loss_c = cat_criterion(cat_out, torch.squeeze(target_cat))
            #loss = loss_h
            loss = loss_h + loss_c

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc_h, cnt_h, pred_h = accuracy(output_heat.detach().cpu().numpy(),
                                         target_heat.detach().cpu().numpy())



        correct = 0
        total = 0

        total = total+1
        correct += (cat_out == target_cat).sum().item()
        avg_acc_c = correct / total
        #avg_acc= (avg_acc_c + avg_acc_h)/2

        point_acc.update(avg_acc_h, cnt_h)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'point Accuracy {point_acc.val:.3f} ({point_acc.avg:.3f})\t'\
                  'cls acc {point_acc.avg:.3f}\t'\
                  .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=input.size(0) / batch_time.val,
                data_time=data_time, loss=losses, point_acc=point_acc, cls_acc=avg_acc_c )
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', point_acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            # save_debug_images(config, input, meta, target, pred*4, output,
            #                   prefix)


def validate(config, val_loader, val_dataset, model, criterion, criterion2, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)

    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_POINTS, 3),
        dtype=np.float32
    )

    all_boxes = np.zeros((num_samples, 7))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    meta_data ={}
    with torch.no_grad():
        end = time.time()
        for i, (input, target_heat, target_weight, target_cat, meta) in enumerate(val_loader):
        #for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # print("target_l")
            # print(target)
            # compute output
            heat_out, cat_out = model(input)
            if isinstance(heat_out, list):
                output = heat_out[-1]
            else:
                output = heat_out

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                heat_out_flipped, cls = model(input_flipped)

                if isinstance(heat_out_flipped, list):
                    output_flipped = heat_out_flipped[-1]
                else:
                    output_flipped = heat_out_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                output_heat = (output + output_flipped) * 0.5


            #print(output.shape) #shape([19, 4, 64, 48])
            cls, score = _topk_class(cat_out, K=1)
            # print('clasess')
            # print(classes)
            output_cat = cls.type(torch.cuda.FloatTensor)

            target_heat = target_heat.cuda(non_blocking=True)
            target_cat = target_cat.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            target_new = torch.max(target_cat, 1)[1]
            loss_h = criterion(output_heat, target_heat, target_weight)
            loss_c = criterion2(output_cat, target_new)
            loss = loss_h + loss_c

            #loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            #measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc_h, cnt_h, pred_h = accuracy(output_heat.detach().cpu().numpy(),
                                               target_heat.detach().cpu().numpy())

            acc.update(avg_acc_h, cnt_h)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()
            cat = cls.detach().cpu().numpy().flatten()
            

            preds, maxvals = get_final_preds(
                config, output_heat.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            all_boxes[idx:idx + num_images, 6] = cat
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target_heat, target_cat, pred_h*4, output_heat, output_cat,
                                  prefix)
                meta_data = meta
        

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums, meta
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0