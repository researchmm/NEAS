import argparse
import torch.nn as nn
import numpy as np
import _init_paths

from datetime import datetime

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    from torch.nn.parallel import DistributedDataParallel as DDP
    has_apex = False

from lib.dataset import Dataset, create_loader, resolve_data_config
from lib.core.train_function import train_epoch
from lib.core.valid_function import validate
from lib.utils.helpers import *
from lib.utils.EMA import ModelEma
from lib.utils.saver import CheckpointSaver
from lib.utils.optimizer import create_optimizer
from lib.utils.scheduler import create_scheduler
from torch.utils.tensorboard import SummaryWriter
from timm.loss import LabelSmoothingCrossEntropy
from lib.models.NEAS_subnet import _gen_NEAS

from configs.config import cfg, update_config_from_file
from tools.utils import init_distributed_mode


def decode_arch(arch_tuple, sta_num):
    split_num = arch_tuple[0]
    num_path = arch_tuple[1]
    split_cand = list()
    split_len = sum(sta_num) - split_num
    p = 2
    sharing_cand = list(arch_tuple[p:p+split_num+1])
    p = p+split_num+1
    for i in range(num_path):
        split_cand.append(list(arch_tuple[p:p+split_len+1]))
        p = p + split_len + 1
    return sharing_cand, split_cand


def main():

    parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='../configs/subnets/314.yaml', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    parser.add_argument('--output', default='', type=str, metavar='PATH',
                        help='path to output folder (default: none, current dir)')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')


    args = parser.parse_args()
    update_config_from_file(args.config)
    init_distributed_mode(args)

    arch_tuple = tuple(cfg.arch_list)

    sharing_cand, split_cand = decode_arch(arch_tuple, sta_num=[4,4,4,4,4])
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        # stage 1, 112x112 in
        ['ir_r1_k3_s2_e4_c24_se0.25', 'ir_r1_k3_s1_e4_c24_se0.25', 'ir_r1_k3_s1_e4_c24_se0.25',
         'ir_r1_k3_s1_e4_c24_se0.25'],
        # stage 2, 56x56 in
        ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r1_k5_s1_e4_c40_se0.25', 'ir_r1_k5_s1_e4_c40_se0.25',
         'ir_r1_k5_s1_e4_c40_se0.25'],
        # stage 3, 28x28 in
        ['ir_r1_k3_s2_e6_c80_se0.25', 'ir_r1_k3_s1_e4_c80_se0.25', 'ir_r1_k3_s1_e4_c80_se0.25',
         'ir_r1_k3_s1_e4_c80_se0.25'],
        # stage 4, 14x14in
        ['ir_r1_k3_s1_e6_c112_se0.25', 'ir_r1_k3_s1_e6_c112_se0.25', 'ir_r1_k3_s1_e6_c112_se0.25',
         'ir_r1_k3_s1_e6_c112_se0.25'],
        # stage 5, 14x14in
        ['ir_r1_k5_s2_e6_c160_se0.25', 'ir_r1_k5_s1_e6_c160_se0.25', 'ir_r1_k5_s1_e6_c160_se0.25',
         'ir_r1_k5_s1_e6_c160_se0.25'],
        # stage 6, 7x7 in
        ['cn_r1_k1_s1_c960_se0.25'],
    ]
    model = _gen_NEAS(
        sharing_cand,
        split_cand,
        arch_def,
        num_head=cfg.num_head,
        model=cfg.model,
        num_classes=cfg.num_classes,
        drop_rate=cfg.drop,
        drop_path_rate=cfg.drop_path,
        global_pool=cfg.gp,
        zero_gamma=cfg.zero_gamma)


    data_config = resolve_data_config(cfg, model=model, verbose=args.local_rank == 0)

    eval_metric = cfg.eval_metric
    best_metric = None
    saver = None
    output_dir = ''

    if args.local_rank == 0:
        output_base = args.output if args.output else './experiments'
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = get_outdir(output_base, timestamp)
        logger = get_logger(os.path.join(output_dir, 'retrain.log'))
        writer = SummaryWriter(os.path.join(output_dir, 'runs'))
        decreasing = True if eval_metric == 'loss' else False
        logger.info("Save model at: {}\n".format(output_dir))
        saver = CheckpointSaver(checkpoint_dir=output_dir, decreasing=decreasing, max_history=4)
    else:
        writer = None
        logger = None

    cfg.prefetcher = not cfg.no_prefetcher
    seed = cfg.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device)
    model.to(device)

    if "initial_checkpoint" in cfg and os.path.exists(cfg.initial_checkpoint):
        load_checkpoint(model, cfg.initial_checkpoint)

    # optionally resume from a checkpoint
    resume_state = {}
    resume_epoch = None

    if os.path.exists(os.path.join(output_dir, 'last.pth.tar')):
        cfg.resume = os.path.join(output_dir, 'last.pth.tar')

    model_ema = None
    if cfg.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=cfg.model_ema_decay,
            device='cpu' if cfg.model_ema_force_cpu else '',
            resume=cfg.resume if not args.eval else cfg.initial_checkpoint)
    model_without_ddp = model

    if args.distributed:
        if has_apex:
            model = DDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                logger.info("Using torch DistributedDataParallel. Install NVIDIA Apex for Apex DDP.")
            model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module

        # NOTE: EMA model does not need to be wrapped by DDP

    optimizer = create_optimizer(cfg, model_without_ddp)

    if cfg.resume:
        resume_state, resume_epoch = resume_checkpoint(model_without_ddp, cfg.resume)
    if resume_state and not args.no_resume_opt:
        if 'optimizer' in resume_state:
            if args.local_rank == 0:
                logging.info('Restoring Optimizer state from checkpoint')
            optimizer.load_state_dict(resume_state['optimizer'])

    del resume_state

    train_dir = os.path.join(cfg.data, 'train')
    eval_dir = os.path.join(cfg.data, 'val')

    if args.local_rank == 0:
        print("train_dir at: {}".format(train_dir))
    if not os.path.exists(train_dir) and args.local_rank == 0:
        logger.error('Training folder does not exist at: {}'.format(train_dir))
        exit(1)

    if not os.path.exists(eval_dir) and args.local_rank == 0:
        logger.error('Validation folder does not exist at: {}'.format(eval_dir))
        exit(1)

    dataset_train = Dataset(train_dir)
    dataset_eval = Dataset(eval_dir)

    if args.local_rank == 0:
        print("current img size is {}\n".format(data_config['input_size']))
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=cfg.batch_size,
        is_training=True,
        use_prefetcher=cfg.prefetcher,
        re_prob=cfg.reprob,
        re_mode=cfg.remode,
        re_count=cfg.recount,
        re_split=cfg.resplit,
        color_jitter=cfg.color_jitter,
        auto_augment=cfg.aa,
        num_aug_splits=0,
        interpolation=cfg.train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=cfg.workers,
        distributed=args.distributed,
        collate_fn=None,
        pin_memory=cfg.pin_mem,
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=cfg.validation_batch_size_multiplier * cfg.batch_size,
        is_training=False,
        use_prefetcher=cfg.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=cfg.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=cfg.pin_mem,
    )

    validate_loss_fn = nn.CrossEntropyLoss().cuda()
    if cfg.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=cfg.smoothing)
    else:
        train_loss_fn = torch.nn.CrossEntropyLoss()

    cfg.lr_noise = None
    lr_scheduler, num_epochs = create_scheduler(cfg, optimizer)

    start_epoch = 0
    if cfg.start_epoch is not None:
        start_epoch = cfg.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        logger.info('Scheduled epochs: {}'.format(num_epochs))

    if args.eval:
        eval_metrics = validate(0, model_ema.ema, loader_eval, validate_loss_fn, cfg, args, log_suffix=' (EMA)',
                                    logger=logger, writer=writer)
        eval_metrics = eval_metrics
        best_record = eval_metrics[eval_metric]
        if args.local_rank == 0:
            logger.info(f"Accuracy of the network on the {len(dataset_eval)} test images: {best_record:.1f}%")
        return

    try:
        best_record = 0
        best_ep = 0
        for epoch in range(start_epoch, num_epochs):
            if args.distributed:
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_epoch(
                epoch, model, loader_train, optimizer, cfg, args, loss_fn=train_loss_fn,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                model_ema=model_ema, logger=logger, writer=writer)

            eval_metrics = validate(epoch, model, loader_eval, validate_loss_fn, cfg, args, logger=logger, writer=writer)

            if model_ema is not None and not cfg.model_ema_force_cpu:
                ema_eval_metrics = validate(epoch, model_ema.ema, loader_eval, validate_loss_fn, cfg, args, log_suffix=' (EMA)',
                                            logger=logger, writer=writer)
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None and args.sched not in ['cyclic', 'onecycle']:
                    lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            update_summary(
                epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                write_header=best_metric is None)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(
                    model, optimizer, args,
                    epoch=epoch, model_ema=model_ema, metric=save_metric, use_amp=cfg.amp)

            if best_record < eval_metrics[eval_metric]:
                best_record = eval_metrics[eval_metric]
                best_ep = epoch

            if args.local_rank == 0:
                logger.info('*** Best metric: {0} (epoch {1})'.format(best_record, best_ep))

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

if __name__ == '__main__':
    main()
