import sys
import argparse
import torch.nn as nn
import yaml
from torch import optim as optim
from thop import profile, clever_format
from torch.utils.tensorboard import SummaryWriter
from timm.utils import *
from lib.utils.helpers import get_logger
from datetime import datetime

def get_logger_writer(args, data_config):

    output_base = args.output if args.output else './experiments'
    if args.platform == 'PAI' and not args.output:
        output_base = '/mnt/configblob/users/v-miche2/CKP'
    if args.platform == 'philly' and not args.output:
        output_base = '/philly/rr1/resrchvc/v-miche2/checkpoint'
    if args.timestamp == 'now':
        args.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.timestamp = args.timestamp + '_' + args.name
    output_dir = get_outdir(output_base, args.model, str(data_config['input_size'][-1]), args.timestamp)
    logger = get_logger(os.path.join(output_dir, 'train.log'))
    writer = SummaryWriter(os.path.join(output_dir, 'runs'))
    return logger, writer

def decode_arch(arch_tuple, sta_num):
    split_num = arch_tuple[0]
    num_path = arch_tuple[1]
    sharing_cand = list()
    split_cand = list()
    sharing_cand.append([0])
    p = 3
    for i in range(split_num):
        end = p + sta_num[i]
        sharing_cand.append(list(arch_tuple[p:end]))
        p = end
    for i in range(num_path):
        cur_path = list()
        for j in range(split_num, len(sta_num)):
            end = p + sta_num[j]
            cur_path.append(list(arch_tuple[p:end]))
            p = end
        cur_path.append([0])
        p += 1
        split_cand.append(cur_path)
    return sharing_cand, split_cand


def encode_arch(sharing_cand, split_cand):
    # sharing_cand [[], [], []], split_cand [[path1], [path1]], path [[], [], []]
    arch_all = [len(sharing_cand) - 1, len(split_cand)] + sum(sharing_cand, []) + sum(sum(split_cand, []), [])
    return tuple(arch_all)
