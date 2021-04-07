import os
import time
import torch
import torchvision
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from lib.models.utils import rand_bbox
from lib.utils.helpers import AverageMeter, accuracy, reduce_tensor
from lib.utils.loss import LabelSmoothingCrossEntropy, CrossEntropy


def train_epoch(
        epoch, model, loader, optimizer, cfg, args, loss_fn = None,
        lr_scheduler=None, saver=None, output_dir='', use_amp=False, model_ema=None, logger=None, writer=None):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()
    model.train()
    k = cfg.num_head

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    optimizer.zero_grad()
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)

        if not cfg.prefetcher:
            input, target = input.cuda(), target.cuda()
        outputs = model(input)

        loss_fns = nn.ModuleList([loss_fn.cuda() for i in range(k)])
        loss = 0
        for i in range(k):
            loss += loss_fns[i](outputs[i], target)

        if cfg.loss_scale:
            loss = loss / cfg.accumulation_steps

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data, args.world_size)
        else:
            reduced_loss = loss.data

        loss.backward()

        if ((batch_idx + 1) % cfg.accumulation_steps == 0) or last_batch:
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        losses_m.update(reduced_loss.item(), input.size(0))


        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % cfg.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.local_rank == 0:
                logger.info(
                    'Train: {} [{:>4d}/{}] '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f}) '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                    'LR: {lr:.3e}'
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                writer.add_scalar('Loss/train', prec1_m.avg, epoch * len(loader) + batch_idx)
                writer.add_scalar('Accuracy/train', prec1_m.avg, epoch * len(loader) + batch_idx)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch * len(loader) + batch_idx)

        if saver is not None and cfg.recovery_interval and (
                last_batch or (batch_idx + 1) % cfg.recovery_interval == 0):
            saver.save_recovery(
                model, optimizer, cfg, epoch, model_ema=model_ema, use_amp=use_amp, batch_idx=batch_idx)

        if lr_scheduler is not None:
            if cfg.sched in ['cyclic', 'onecycle']:
                lr_scheduler.step()
            else:
                lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)


        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])