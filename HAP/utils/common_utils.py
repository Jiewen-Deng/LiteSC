"""
@Author: JW Deng
@Time: 2026/7/6 17:00
"""
import os
import time
import json
import logging
import torch
from pprint import pprint
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter
import torch.nn as nn

running_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

def get_logger(name, logpath, filepath, package_files=[],
               displaying=True, saving=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    log_path = logpath + name.replace(':', '_')
    makedirs(log_path)
    if saving:
        info_file_handler = logging.FileHandler(log_path)
        info_file_handler.setLevel(logging.INFO)
        logger.addHandler(info_file_handler)
    logger.info(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, 'r', encoding='utf-8') as package_f:
            logger.info(package_f.read())
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    return logger


def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


def str_to_list(src, delimiter, converter):
    """Conver a string to list.
    """
    src_split = src.split(delimiter)
    res = [converter(_) for _ in src_split]
    return res


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    config = edict(config_dict)

    return config, config_dict

def try_contiguous(x):
    if not x.is_contiguous():
        x = x.contiguous()

    return x


# =====================================================
# For learning rate schedule
# =====================================================
class StairCaseLRScheduler(object):
    def __init__(self, start_at, interval, decay_rate):
        self.start_at = start_at
        self.interval = interval
        self.decay_rate = decay_rate

    def __call__(self, optimizer, iteration):
        start_at = self.start_at
        interval = self.interval
        decay_rate = self.decay_rate
        if (start_at >= 0) \
                and (iteration >= start_at) \
                and (iteration + 1) % interval == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_rate
                print('[%d]Decay lr to %f' % (iteration, param_group['lr']))

    @staticmethod
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            return lr


class PresetLRScheduler(object):
    """Using a manually designed learning rate schedule rules.
    """
    def __init__(self, decay_schedule):
        # decay_schedule is a dictionary
        # which is for specifying iteration -> lr
        self.decay_schedule = decay_schedule
        print('=> Using a preset learning rate schedule:')
        pprint(decay_schedule)
        self.for_once = True

    def __call__(self, optimizer, iteration):
        for param_group in optimizer.param_groups:
            lr = self.decay_schedule.get(iteration, param_group['lr'])
            param_group['lr'] = lr

    @staticmethod
    def get_lr(optimizer, alpha=False):
        lr_list = []
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            lr_list.append(lr)
        if alpha:
            return lr_list[0], lr_list[1]
        else:
            return lr_list[0]


    def update_lr(self, optimizer, iteration):
        for i, p in enumerate(optimizer.param_groups):
            lr = self.decay_schedule.get(iteration, p['lr'])
        for i, p in enumerate(optimizer.param_groups):
            if i == 0:  # quantization paragmeter group
                p['lr'] = lr / 10
            else:
                p['lr'] = lr
# =======================================================
# For math computation
# =======================================================
def init_summary_writer(config):
    makedirs(config.summary_dir)
    makedirs(config.checkpoint_dir)
    print(config.checkpoint, os.path.exists(config.checkpoint))
    if not os.path.exists(config.checkpoint):
        os.makedirs(config.checkpoint)

    # set logger
    path = 'C:\\Users\\dengjiewen\\PycharmProjects\\LDSC_github'
    # path = os.path.dirname(os.path.abspath(__file__))
    path_model = 'C:\\Users\\dengjiewen\\PycharmProjects\\LDSC_github\\N2UQ\\transformer_constellation.py'
    # path_model = os.path.join(path, 'models/%s.py' % config.network)
    path_main = os.path.join(path, 'main_HAP.py')
    path_pruner = os.path.join(path, 'HAP/pruner/%s.py' % config.pruner)

    logger = get_logger(f'log{running_time}.log_time', logpath=config.saving_log,
                        filepath=path_model, package_files=[path_main, path_pruner])
    logger.info(dict(config))
    writer = SummaryWriter(config.summary_dir)

    return logger, writer

def save_model(config, iteration, pruner, cfg, stat):
    network = config.network
    depth = config.depth
    dataset = config.dataset
    path = os.path.join(config.checkpoint, '%s_%s%s_%d.pth.tar' % (dataset, network, depth, iteration))
    save = {
        'config': config,
        'net': pruner.model,
        'cfg': cfg,
        'stat': stat
    }
    torch.save(save, path)

def fetch_mat_weights(layer, use_patch=False):
    # -> output_dium * input_dim (kh*kw*in_c + [1 if with bias])
    if isinstance(layer, nn.Conv2d):
        if use_patch:
            weight = layer.weight.transpose(1, 2).transpose(2, 3)  # n_out * kh * kw * inc
            n_out, k_h, k_w, in_c = weight.size()
            weight = try_contiguous(weight)
            weight = weight.view(-1, weight.size(-1))
            bias = 0
            if layer.bias is not None:
                copied_bias = torch.cat([layer.bias.unsqueeze(1) for _ in range(k_h*k_w)], 1).view(-1, 1)
                weight = torch.cat([weight, copied_bias], 1)  # layer.bias.unsqueeze(1)], 1)
                bias = 1
            weight = weight.view(n_out, k_h*k_w, in_c+bias)
        else:
            weight = layer.weight  # n_filters * in_c * kh * kw
            # weight = weight.transpose(1, 2).transpose(2, 3).contiguous()
            weight = weight.view(weight.size(0), -1)
            if layer.bias is not None:
                weight = torch.cat([weight, layer.bias.unsqueeze(1)], 1)
    elif isinstance(layer, nn.Linear):
        weight = layer.weight
        if layer.bias is not None:
            weight = torch.cat([weight, layer.bias.unsqueeze(1)], 1)
    elif isinstance(layer, nn.Embedding) or isinstance(layer, nn.LayerNorm):
        weight = layer.weight
    else:
        raise NotImplementedError

    return weight