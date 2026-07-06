"""
Perform Hessian-aware pruning on pretrained digital semantic system.
@Author: JW Deng
@Time: 2026/7/6 12:00
"""
import os
import argparse
import json
import time
import tqdm
import torch
import re
import torch.utils.model_zoo as model_zoo
running_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

from HAP.utils.common_utils import *
from HAP.utils.data_utils import *
from HAP.utils.network_utils import *
from HAP.utils.compute_flops import *
from HAP.pruner.hessian_pruner import *
from pretrain_utils import setup_seed

parser = argparse.ArgumentParser()
parser.add_argument("--gpu",type=str, default="2")
parser.add_argument('--config', type=str, default='HAP/transformer_prune.json', required=False, help='could choose configs/imagenet_exps/hessian_trace.json')
parser.add_argument('--network' , type=str, default='Transformer', required=False, help='Please choose DeepSC, Transformer, QuantTransformer, N2UQTransformer')
parser.add_argument('--channel' , type=str, default='Rician', required=False, help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--ratio', type=str, default="0.5", required=False, help='0.1, 0.3, 0.5, 0.7, 0.9, 0.95')
parser.add_argument('--hessian-mode', type=str, default='trace', required=False, help='choose random, magnitude or trace')
parser.add_argument('--bits', default=4, type=int, help='Please choose 1, 2, 4, 6, 8, 10bits')
parser.add_argument('--quant', default='N2UQ_Symmetric', type=str, help='Please choose  N2UQ_Symmetric, N2UQ_Asymmetric, quant')
parser.add_argument('--epochs', type=str, default='40', required=False)
parser.add_argument('--learning-rate', type=str, default='1e-3', required=False)
parser.add_argument('--weight-decay', type=str, default='1e-4', required=False)
parser.add_argument("--data_distributed",type=int, default=0)
parser.add_argument('--seed',type=int, default=10, help='Setup seed')
args = parser.parse_args()
setup_seed(args.seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

print('Using config!')
config, _ = get_config_from_json(args.config)
# print(config)

config.bits                 = args.bits
config.network              = args.network
config.quant                = args.quant
config.channel              = args.channel
config.ratio                = args.ratio
config.hessian_mode = args.hessian_mode
config.epochs                = args.epochs
config.learning_rate        = args.learning_rate
config.weight_decay         = args.weight_decay
config.load_checkpoint = f"./runs/pretrain/{args.channel}_{args.quant}/{args.bits}bits_ep200_lr0.001/checkpoint/"

config.exp_name = f"{config.channel}_{config.quant}/pr{config.ratio}_{config.bits}bits_{config.hessian_mode}_ep{config.epochs}_lr{config.learning_rate}_wd{config.weight_decay}"
config.summary_dir = f"./runs/pruning/{config.exp_name}/summary/"
config.checkpoint = f"./runs/pruning/{config.exp_name}/checkpoint/"
config.saving_log = f"./runs/pruning/{config.exp_name}/logs/"
# config.checkpoint = f"./HAP/out/{args.channel}_{args.quant}/pr_{args.ratio}_{args.bits}bits_{args.hessian_mode}"
if args.data_distributed:
    torch.cuda.set_device(args.local_rank)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

stats = {}
""" preparing the dataset """
vocab = json.load(open(config.vocab_file, 'rb'))
token_to_idx = vocab['token_to_idx']
num_vocab = len(token_to_idx)
pad_idx = token_to_idx["<PAD>"]
start_idx = token_to_idx["<START>"]
end_idx = token_to_idx["<END>"]
if config.data_distributed:
    torch.distributed.init_process_group(backend="nccl")
device = torch.device(f'cuda:{args.gpu}')
# device = torch.device('cuda:0,1')
criterion = torch.nn.CrossEntropyLoss(reduction='none')
# criterion = torch.nn.CrossEntropyLoss()

logger, writer = init_summary_writer(config)
trainloader, testloader = init_dataloader(config)

if config.data_distributed:
    trainset, testset = trainloader, testloader
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
    trainloader = torch.utils.data.DataLoader(trainset, config.batch_size_train, False, num_workers=config.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
    testloader = torch.utils.data.DataLoader(testset, config.batch_size_test, False, num_workers=config.num_workers, pin_memory=True, drop_last=True, sampler=test_sampler)

hessianloader = get_hessianloader(config.dataset, config.hessian_batch_size)
hess_data = hessianloader
net, bottleneck_net = init_network(config, logger, device)

pruner = init_pruner(net, config, writer, logger)

# total FLOPs calculation
total_flops, _ = compute_transformer_flops(pruner.model, 31, cuda=True)

# start pruning
epochs         = str_to_list(config.epochs, ',', int)
learning_rates = str_to_list(config.learning_rate, ',', float)
weight_decays  = str_to_list(config.weight_decay, ',', float)
ratios         = str_to_list(config.ratio, ',', float)

fisher_type = config.fisher_type  # empirical|true
fisher_mode = config.fisher_mode  # eigen|full|diagonal
normalize = config.normalize
prune_mode = config.prune_mode  # one-pass | iterative
fix_rotation = config.get('fix_rotation', True)

assert (len(epochs) == len(learning_rates) and
        len(learning_rates) == len(weight_decays) and
        len(weight_decays) == len(ratios))

total_parameters = count_parameters_embedding(net)
# total_parameters = count_parameters(net)
for it in range(len(epochs)):
    epochs = epochs[it]
    lr = learning_rates[it]
    wd = weight_decays[it]
    ratio = ratios[it]
    logger.info('-'*120)
    logger.info('** [%d], Ratio: %.2f, epochs: %d, lr: %.4f, wd: %.4f' % (it, ratio, epochs, lr, wd))
    logger.info('Reinit: %s, Fisher_mode: %s, fisher_type: %s, hessian_mode: %s, normalize: %s, fix_rotation: %s.' %
        (config.re_init, fisher_mode, fisher_type, config.hessian_mode, normalize, fix_rotation))
    pruner.fix_rotation = fix_rotation

    # test pretrained model
    if config.init_test:
        train_loss_pruned, train_acc_pruned, top5_acc = pruner.test_model(trainloader, criterion, config.channel, device)
        test_loss_pruned, test_acc_pruned, top5_acc = pruner.test_model(testloader, criterion, config.channel, device)
        logger.info('Pretrain: Accuracy: %.2f%%(train), %.2f%%(test).' % (train_acc_pruned, test_acc_pruned))
        logger.info('          Loss:     %.2f  (train), %.2f  (test).' % (train_loss_pruned, test_loss_pruned))

    # conduct pruning
    if 'hessian' not in config.fisher_mode:
        cfg = pruner.make_pruned_model(trainloader,pad_idx,
                                    criterion=criterion,
                                    device=device,
                                    fisher_type=fisher_type,
                                    prune_ratio=ratio,
                                    normalize=normalize,
                                    re_init=config.re_init)
    else:
        cfg = pruner.make_pruned_model(hess_data,pad_idx,
                                    criterion=criterion,
                                    device=device,
                                    fisher_type=fisher_type,
                                    prune_ratio=ratio,
                                    normalize=normalize,
                                    re_init=config.re_init,
                                    n_v=config.nv)
    print(pruner.model)
    # for tracking the best accuracy
    compression_ratio, unfair_ratio, all_numel, rotation_numel = compute_ratio(pruner.model, total_parameters,
                                                                               fix_rotation, logger)
    remained_flops, rotation_flops = compute_transformer_flops(pruner.model, 31, cuda=True)
    logger.info('  + Remained FLOPs: %.4fM(%.2f%%), Total FLOPs: %.4fM' % (remained_flops / 1e6, 100.*remained_flops/total_flops ,total_flops / 1e6) )

    logger.info(f"Total Flops: {remained_flops}")

    test_loss_pruned, test_acc_pruned, top5_acc = pruner.test_model(testloader, criterion, config.channel, device)
    if 'N2UQ' in config.network or 'N2UQ' in config.quant:
        test_loss_finetuned, test_acc_finetuned = pruner.fine_tune_N2UQ(trainloader=trainloader,
                                                                     testloader=testloader,
                                                                     criterion=criterion,
                                                                     learning_rate=lr,
                                                                     weight_decay=wd,
                                                                     channel=config.channel,
                                                                     nepochs=epochs,
                                                                     device=device)
    else:
        test_loss_finetuned, test_acc_finetuned = pruner.fine_tune_model(trainloader=trainloader,
                                                                     testloader=testloader,
                                                                     criterion=criterion,
                                                                     learning_rate=lr,
                                                                     weight_decay=wd,
                                                                     channel=config.channel,
                                                                     nepochs=epochs,
                                                                     device=device)

    train_loss_finetuned, train_acc_finetuned, top5_acc = pruner.test_model(trainloader, criterion, config.channel, device)# 训练集上的测试结果
    logger.info(f'After {config.dataset, config.network}:  Accuracy: %.2f%%(train), %.2f%%.' % (train_acc_finetuned, test_acc_finetuned))
    logger.info('        Loss:     %.2f  (train), %.2f  .' % (train_loss_finetuned, test_loss_finetuned))

    stat = {
        'total_flops': total_flops,
        'rotation_flops': rotation_flops,
        'flops_remained': float(100.*remained_flops / total_flops),
        'it': it,
        'prune_ratio': ratio,
        'cr': compression_ratio,
        'unfair_cr': unfair_ratio,
        'all_params': all_numel,
        'rotation_params': rotation_numel,
        'prune/test_loss': test_loss_pruned,
        'prune/test_acc': test_acc_pruned,
        'finetune/train_loss': train_loss_finetuned,
        'finetune/test_loss': test_loss_finetuned,
        'finetune/train_acc': train_acc_finetuned,
        'finetune/test_acc': test_acc_finetuned
    }

    print('saving checkpoint')
    save_model(config, it, pruner, cfg, stat)

    stats[it] = stat
    if prune_mode == 'one_pass':
        print('one_pass')
        del net
        del pruner
        net, bottleneck_net = init_network(config, logger, device)
        pruner = init_pruner(net, config, writer, logger)
        pruner.iter = it
    with open(os.path.join(config.saving_log, f'stats_{running_time}.json'.replace(':', '_')), 'w') as f:
        json.dump(stats, f)
    if prune_mode != 'one_pass':
        with open(os.path.join(config.saving_log, f'stats{it}.json'), 'w') as f:
            json.dump(stats, f)