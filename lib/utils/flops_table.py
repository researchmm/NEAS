import torch
from ptflops import get_model_complexity_info

class LatencyEst(object):
    def __init__(self, model, input_shape=(1, 3, 224, 224), device='cpu'):
        self.block_num = len(model.blocks)
        self.choice_num = len(model.blocks[0])
        self.latency_dict = {}
        self.flops_dict = {}
        self.params_dict = {}
        self.num_path = len(model.conv_heads)
        if device == 'cpu':
            model = model.cpu()
        else:
            model = model.cuda()

        self.params_fixed = 0
        self.flops_fixed = 0

        input = torch.randn((2, 3, 224, 224))

        flops, params = get_model_complexity_info(model.conv_stem, (3, 224, 224), as_strings=False, print_per_layer_stat=False)
        self.params_fixed += params / 1e6
        self.flops_fixed += flops / 1e6

        input = model.conv_stem(input)

        for block_id, block in enumerate(model.blocks):
            self.flops_dict[block_id] = {}
            self.params_dict[block_id] = {}
            for module_id, module in enumerate(block):
                self.flops_dict[block_id][module_id] = {}
                self.params_dict[block_id][module_id] = {}
                for choice_id, choice in enumerate(module):
                    flops, params = get_model_complexity_info(choice, tuple(input.shape[1:]),  as_strings=False, print_per_layer_stat=False)
                    self.flops_dict[block_id][module_id][choice_id] = flops / 1e6 # M
                    self.params_dict[block_id][module_id][choice_id] = params /1e6 # M

                input = choice(input)

        # conv_last
        flops, params = get_model_complexity_info(model.global_pool, tuple(input.shape[1:]), as_strings=False, print_per_layer_stat=False)
        self.params_fixed += params / 1e6
        self.flops_fixed += flops / 1e6

        input = model.global_pool(input)

        # globalpool
        flops, params = get_model_complexity_info(model.conv_heads[0], tuple(input.shape[1:]), as_strings=False, print_per_layer_stat=False)
        self.params_fixed += self.num_path * params / 1e6
        self.flops_fixed += self.num_path * flops / 1e6

        input = model.conv_heads[0](input).flatten(1)

        flops, params = get_model_complexity_info(model.classifiers[0], tuple(input.shape[1:]), as_strings=False, print_per_layer_stat=False)
        self.params_fixed += self.num_path * params / 1e6
        self.flops_fixed += self.num_path * flops / 1e6


    # return params (M)
    def get_params(self, sharing_cand, split_cand):
        params = 0

        for block_id, block in enumerate(sharing_cand):
            for module_id, choice in enumerate(block):
                if choice == -1:
                    continue
                params += self.params_dict[block_id][module_id][choice]

        for path_id, path in enumerate(split_cand):
            for block_id, block in enumerate(path):
                for module_id, choice in enumerate(block):
                    if choice == -1:
                        continue
                    params += self.params_dict[block_id + len(sharing_cand)][module_id][choice]
        return params + self.params_fixed

    # return flops (M)
    def get_flops(self, sharing_cand, split_cand):
        flops = 0
        for block_id, block in enumerate(sharing_cand):
            for module_id, choice in enumerate(block):
                if choice == -1:
                    continue
                flops += self.flops_dict[block_id][module_id][choice]

        for path_id, path in enumerate(split_cand):
            for block_id, block in enumerate(path):
                for module_id, choice in enumerate(block):
                    if choice == -1:
                        continue
                    flops += self.flops_dict[block_id + len(sharing_cand)][module_id][choice]

        return flops + self.flops_fixed

if __name__ == '__main__':
    from lib.models.hypernet import _gen_supernet
    model = _gen_supernet()
    est = LatencyEst(model)
    print(est.get_flops([[0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0]]))