import torch
import torch.nn as nn
from torch.nn import functional as F
import copy
from lib.models.NEAS_builder import *

DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


_DEBUG = False
class NEAS(nn.Module):

    def __init__(self, sharing_args, split_args, num_classes=1000, in_chans=3, stem_size=16, num_features=1280, head_bias=True,
                 channel_multiplier=1.0, pad_type='', act_layer=nn.ReLU, drop_rate=0., drop_path_rate=0., num_head=2,
                 se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None, global_pool='avg', pool_bn=False, zero_gamma=False,
                 add_dw=False, block_type='ir', add_last_conv=False, change_se=False, change_ir_se=False, dilation_method=None):
        super(NEAS, self).__init__()
        # custom DIY
        self.dilation_method = dilation_method
        self.add_last_conv = add_last_conv
        self.change_se = change_se
        self.block_type = block_type
        self.add_dw = add_dw
        self.change_ir_se = change_ir_se
        self.k = num_head

        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate
        self._in_chs = in_chans
        self.pool_bn = pool_bn

        # Stem
        stem_size = round_channels(stem_size, channel_multiplier)
        self.conv_stem = create_conv2d(self._in_chs, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        self._in_chs = stem_size

        # Middle stages (IR/ER/DS Blocks)
        builder = GENetBuilder(
            channel_multiplier, 8, None, 32, pad_type, act_layer, se_kwargs,
            norm_layer, norm_kwargs, drop_path_rate, verbose=_DEBUG, add_dw=self.add_dw,
            block_type=self.block_type, change_se=self.change_se, change_ir_se=self.change_ir_se, dilation_method=self.dilation_method)

        k = self.k
        self.split_in_chs = sharing_args[-1][-1]['out_chs']
        self.sharing_blocks = nn.Sequential(*builder(self._in_chs, sharing_args))
        self.head_in_chs = sharing_args[-1][-1]['out_chs']
        self.blocks = list()
        for i in range(k):
            self.blocks.append(nn.Sequential(*builder(self.head_in_chs, split_args[i])))
        self.blocks = nn.ModuleList(self.blocks)

        # self.blocks = builder(self._in_chs, block_args)
        self._in_chs = builder.in_chs

        # Head + Pooling
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.conv_heads = nn.ModuleList([create_conv2d(self._in_chs, self.num_features, 1, padding=pad_type, bias=head_bias) for i in range(k)])
        self.acts = nn.ModuleList([act_layer(inplace=True) for i in range(k)])


        # Classifier
        self.classifiers = nn.ModuleList([nn.Linear(self.num_features * self.global_pool.feat_mult(), self.num_classes) for i in range(k)])
        if pool_bn:
            self.pool_bns = nn.ModuleList([nn.BatchNorm1d(1) for i in range(self.k)])

        efficientnet_init_weights(self, zero_gamma=zero_gamma)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_classes = num_classes
        self.classifier = nn.Linear(
            self.num_features * self.global_pool.feat_mult(), num_classes) if self.num_classes else None

    def forward_features(self, x):
        # architecture = [[0], [], [], [], [], [0]]
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x_sharing = self.sharing_blocks(x)
        x_sharing = GradDivHeads.apply(x_sharing, self.k)
        xs = list()
        for i in range(self.k):
            xs.append(self.blocks[i](x_sharing))
            xs[i] = self.global_pool(xs[i])
            xs[i] = self.conv_heads[i](xs[i])
            xs[i] = self.acts[i](xs[i])
        return xs

    def forward_features_layers(self, x, layer_number):
        # architecture = [[0], [], [], [], [], [0]]
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        cnt = 0
        for idx, block in enumerate(self.sharing_blocks):
            for layer_idx, layer in enumerate(block):
                if cnt <= layer_number:
                    x = layer(x)
                    cnt += 1

        for idx, block in enumerate(self.blocks[0]):
            for layer_idx, layer in enumerate(block):
                if cnt <= layer_number:
                    x = layer(x)
                    cnt += 1

        return x

    def forward(self, x):
        xs = self.forward_features(x)

        for i in range(self.k):
            xs[i] = xs[i].flatten(1)
            if self.drop_rate > 0.:
                xs[i] = F.dropout(xs[i], p=self.drop_rate, training=self.training)
            xs[i] = self.classifiers[i](xs[i])
            if self.pool_bn:
                xs[i] = torch.unsqueeze(xs[i], 1)
                xs[i] = self.pool_bns[i](xs[i])
                xs[i] = torch.squeeze(xs[i])

        return xs


class GradDivHeads(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_heads):
        ctx.num_heads = num_heads
        return x
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output / ctx.num_heads, None

def modify_block_args(block_args, kernel_size, exp_ratio):
    # kernel_size: 3,5,7
    # exp_ratio: 4,6
    block_type = block_args['block_type']
        # each type of block has different valid arguments, fill accordingly
    if block_type == 'cn':
        block_args['kernel_size'] = kernel_size
    elif block_type == 'er':
        block_args['exp_kernel_size'] = kernel_size
    else:
        block_args['dw_kernel_size'] = kernel_size

    if block_type == 'ir' or block_type == 'er':
        block_args['exp_ratio'] = exp_ratio
    return block_args


def _gen_NEAS(sharing_cand, split_cand, arch_def, num_classes, **kwargs):
    # arch_list = [[0], [], [], [], [], [0]]
    choices = {'kernel_size': [3, 5, 7], 'exp_ratio': [4, 6]}
    choices_list = [[x, y] for x in choices['kernel_size'] for y in choices['exp_ratio']]

    num_features = 1280

    act_layer = Swish
    arch_def = sum(arch_def, [])
    sharing_arch = []
    sharing_num =  len(sharing_cand)

    for i, (layer_choice, layer_arch) in enumerate(zip(sharing_cand, arch_def[:sharing_num])):
        if i == 0:
            sharing_arch.append(layer_arch)
            continue
        else:
            if layer_choice == -1:
                continue
            kernel_size, exp_ratio = choices_list[layer_choice]
            elements = layer_arch.split('_')
            layer_arch = layer_arch.replace(elements[2], 'k{}'.format(str(kernel_size)))
            layer_arch = layer_arch.replace(elements[4], 'e{}'.format(str(exp_ratio)))

            sharing_arch.append(layer_arch)

    sharing_args = decode_arch_def([sharing_arch])
    split_args = []
    for path in split_cand:
        path_arch = []
        for i, (layer_choice, layer_arch) in enumerate(zip(path, arch_def[sharing_num:])):
            if i == len(path)-1:
                path_arch.append(layer_arch)
                continue
            else:
                if layer_choice == -1:
                    continue
                kernel_size, exp_ratio = choices_list[layer_choice]
                elements = layer_arch.split('_')
                layer_arch = layer_arch.replace(elements[2], 'k{}'.format(str(kernel_size)))
                layer_arch = layer_arch.replace(elements[4], 'e{}'.format(str(exp_ratio)))

                path_arch.append(layer_arch)
        else:
            path_arch = path_arch
        split_args.append(decode_arch_def([path_arch]))
    kwargs.pop('model')

    model_kwargs = dict(
        sharing_args = sharing_args,
        split_args = split_args,
        num_features=num_features,
        stem_size=16,
        num_classes=num_classes,
        norm_kwargs=resolve_bn_args(kwargs),
        act_layer=act_layer,
        se_kwargs=dict(act_layer=nn.ReLU, gate_fn=hard_sigmoid, reduce_mid=True, divisor=8),
        **kwargs,
    )
    model = NEAS(**model_kwargs)
    return model



