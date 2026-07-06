# -*- coding: utf-8 -*-
"""
Train quantization-based digital semantic communication.
@Author: JW Deng
@Time: 2026/7/6 11:45
"""
import argparse
# import os
# import time
import json
# import numpy as np
# import torch
# import torch.nn as nn
from dataset import EurDataset, collate_data
from torch.utils.data import DataLoader
from tqdm import tqdm
from pretrain_utils import *
from N2UQ.transformer_constellation import Transformer
from HAP.utils.common_utils import get_logger, makedirs
from tensorboardX import SummaryWriter

running_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

def validate(epoch, args, net, writer=None):
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    net.eval()
    pbar = tqdm(test_iterator)
    total = 0
    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)
            loss = val_step_quant(net, sents, sents, 0.1, pad_idx,
                             criterion, args.channel)

            total += loss
            pbar.set_description(
                'Epoch: {}; Type: VAL; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )

    avg_loss = total/len(test_iterator)

    if writer is not None:
        writer.add_scalar('Validation/loss', avg_loss, epoch)

    return avg_loss


def train(epoch, args, net, mi_net, writer=None):
    train_eur = EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator)
    total = 0
    if args.channel == 'AWGN':
        noise_std = np.random.uniform(SNR_to_noise(0), SNR_to_noise(10), size=(1))[0]
    else:
        noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))[0]

    batch_count = 0
    for sents in pbar:
        sents = sents.to(device)
        loss = train_step_quant(net, sents, sents, noise_std, pad_idx,
                          optimizer, criterion, args.channel)
        pbar.set_description(
            'Epoch: {};  Type: Train; Loss: {:.5f}'.format(epoch, loss))
        total += loss

        if writer is not None:
            global_step = epoch * len(train_iterator) + batch_count
            writer.add_scalar('Training/batch_loss', loss, global_step)

        batch_count += 1

    avg_loss = total/len(train_iterator)

    if writer is not None:
        writer.add_scalar('Training/epoch_loss', avg_loss, epoch)

    return avg_loss

parser = argparse.ArgumentParser()
parser.add_argument('--vocab-file',
                    default='C:/Users/dengjiewen/PycharmProjects/Digital_SC/europarl/europarl_origin_cut/vocab.json', type=str)
parser.add_argument('--bits', default=4, type=int, help='Please choose 1, 2, 4, 8,16bits')
parser.add_argument('--quant', default='N2UQ_Symmetric', type=str, help='Please choose  N2UQ_Symmetric, N2UQ_Asymmetric, quant')
parser.add_argument('--channel', default='Rayleigh', type=str, help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--learning_rate', default=1e-3, type=float)# default=1e-3
parser.add_argument('--seed',type=int, default=10, help='Setup seed')
args = parser.parse_args()
setup_seed(args.seed)

# Build experiment name
args.exp_name = f"{args.channel}_{args.quant}/{args.bits}bits_ep{args.epochs}_lr{args.learning_rate}"

# Setup paths
base_path = f"./runs/pretrain/{args.exp_name}"
summary_dir = os.path.join(base_path, "summary/")
checkpoint_dir = os.path.join(base_path, "checkpoint/")
log_dir = os.path.join(base_path, "logs/")

makedirs(summary_dir)
makedirs(checkpoint_dir)
makedirs(log_dir)

# Initialize logger and writer
path_main = os.path.abspath(__file__)
path_model = 'C:\\Users\\dengjiewen\\PycharmProjects\\LDSC_github\\N2UQ\\transformer_constellation.py'
path_utils = 'C:\\Users\\dengjiewen\\PycharmProjects\\LDSC_github\\pretrain_utils.py'

logger = get_logger(f'log{running_time}.log_time', logpath=log_dir,
                    filepath=path_main, package_files=[path_model, path_utils], displaying=True, saving=True)
writer = SummaryWriter(summary_dir)

checkpoint_save_path = checkpoint_dir

logger.info('='*120)
logger.info(f'Experiment Name: {args.exp_name}')
logger.info('Starting Pretraining with following configuration:')
logger.info(f'Channel: {args.channel}')
logger.info(f'Quantization: {args.quant}')
logger.info(f'Bits: {args.bits}')
logger.info(f'Model Dimension (d_model): {args.d_model}')
logger.info(f'Feed Forward Dimension (dff): {args.dff}')
logger.info(f'Number of Layers: {args.num_layers}')
logger.info(f'Number of Heads: {args.num_heads}')
logger.info(f'Batch Size: {args.batch_size}')
logger.info(f'Total Epochs: {args.epochs}')
logger.info(f'Learning Rate: {args.learning_rate}')
logger.info(f'Checkpoint Path: {checkpoint_dir}')
logger.info('='*120)

start = time.time()
""" preparing the dataset """
vocab = json.load(open(args.vocab_file, 'rb'))
token_to_idx = vocab['token_to_idx']
num_vocab = len(token_to_idx)
pad_idx = token_to_idx["<PAD>"]
start_idx = token_to_idx["<START>"]
end_idx = token_to_idx["<END>"]

""" define optimizer and loss function """
DSC = Transformer(args.num_layers, num_vocab, num_vocab,
                         num_vocab, num_vocab, args.d_model, args.num_heads,
                         args.dff, 0.1, args.quant, args.bits).to(device)
criterion = nn.CrossEntropyLoss(reduction='none')
model_paths = []
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
else:
    for fn in os.listdir(checkpoint_dir):
        if not fn.endswith('.pth'): continue
        idx = int(os.path.splitext(fn)[0].split('_')[-1])  # read the idx of image
        model_paths.append((os.path.join(checkpoint_dir, fn), idx))
    model_paths.sort(key=lambda x: x[1])  # sort the image by the idx

total_params = sum(p.numel() for p in DSC.parameters())
trainable_params = sum(p.numel() for p in DSC.parameters() if p.requires_grad)
logger.info(f"Total parameters: {total_params:,}")
logger.info(f"Trainable parameters: {trainable_params:,}")

record_acc = 6
train_loss_record = []
validate_loss_record = []
best_val_loss = float('inf')
best_epoch = 0

if model_paths:
    model_path, epoch_origin = model_paths[-1]
    checkpoint = torch.load(model_path)
    DSC.load_state_dict(checkpoint)
    epoch_range = [epoch_i for epoch_i in range((epoch_origin+1),epoch_origin+args.epochs+1)]
    logger.info(f'Model loaded from: {model_path}')
    logger.info(f'Resuming from epoch: {epoch_origin + 1}')
else:
    initNetParams(DSC)
    logger.info('Model initialized with random parameters')
    epoch_range = range(args.epochs)

    # update constellation quantizer parameters
    logger.info('Updating constellation quantizer parameters (calibration)...')
    train_eur = EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator)
    for sents in pbar:
        sents = sents.to(device)
        DSC = update_quant(DSC, sents, sents, pad_idx)
    logger.info('Calibration completed')

all_parameters = DSC.parameters()
alpha_parameters = []
for pname, p in DSC.named_parameters():
    if 'quant_constellation.a' in pname or 'start' in pname:
    # if 'quant_constellation.a' in pname or 'start' in pname or 'scale' in pname:
        # print('alpha_param:', pname)
        alpha_parameters.append(p)
alpha_parameters_id = list(map(id, alpha_parameters))
other_parameters = list(filter(lambda p: id(p) not in alpha_parameters_id, all_parameters))
lr = args.learning_rate
optimizer = torch.optim.Adam(
        [{'params' : alpha_parameters, 'lr': lr/10, 'name': 'quantization_params'},
        {'params' : other_parameters, 'lr': lr, 'weight_decay': 0.0005, 'name': 'other_params'}],
        betas=(0.9,0.999), eps=1e-9)

logger.info('-'*120)
logger.info('Optimizer Configuration:')
for i, param_group in enumerate(optimizer.param_groups):
    group_name = param_group.get('name', f'group_{i}')
    cur_lr = param_group['lr']
    weight_decay = param_group.get('weight_decay', 0)
    logger.info(f'  Group {i} ({group_name}): lr={cur_lr}, weight_decay={weight_decay}')

# Initial validation
logger.info('Running initial validation...')
avg_acc = validate(0, args, DSC, writer)
logger.info(f'Initial validation loss: {avg_acc:.6f}')

for epoch in epoch_range:
    start_epoch = time.time()

    # Log quantizer parameters at the beginning of each epoch
    for pname, p in DSC.named_parameters():
        if 'quant_constellation.a' in pname or 'start' in pname:
        # if 'quant_constellation.a' in pname or 'start' in pname or 'scale' in pname:
            logger.info(f'Epoch {epoch} - Quantizer parameter {pname}: {p}')
            break

    # Training phase
    logger.info(f'\nEpoch {epoch}/{epoch_range[-1]} - Training Phase')
    tloss = train(epoch, args, DSC, mi_net=None, writer=writer)
    train_loss_record.append(tloss)
    logger.info(f'Epoch {epoch} - Average training loss: {tloss:.6f}')

    # Validation phase
    logger.info(f'Epoch {epoch} - Validation Phase')
    avg_acc = validate(epoch, args, DSC, writer)
    validate_loss_record.append(avg_acc)
    logger.info(f'Epoch {epoch} - Average validation loss: {avg_acc:.6f}')

    # Save best model
    if avg_acc < record_acc:
        save_model_weight(DSC, checkpoint_save_path, epoch, max_files=2)
        logger.info(f'Epoch {epoch} - New best model saved! Validation loss: {avg_acc:.6f}')
        record_acc = avg_acc

    if avg_acc < best_val_loss:
        best_val_loss = avg_acc
        best_epoch = epoch

    # Adjust learning rate every 40 epochs
    if (epoch-epoch_range[0]+1)%40 == 0:
        logger.info(f'Epoch {epoch} - Adjusting learning rate')
        load_model(DSC, checkpoint_save_path)
        lr = adjust_learning_rate(optimizer, lr)
        for i, param_group in enumerate(optimizer.param_groups):
            group_name = param_group.get('name', f'group_{i}')
            cur_lr = param_group['lr']
            logger.info(f'Adjusted learning rate {i} ({group_name}): {cur_lr}')

    ##########################################################################
    elapsed = time.time() - start_epoch
    logger.info(f'Epoch {epoch} completed in {elapsed:.2f} seconds')
    logger.info(f'Current best validation loss: {best_val_loss:.6f} (epoch {best_epoch})')
    logger.info(f'{"="*60}')

logger.info('-'*120)
logger.info(f'Training Finished! Minimum Validation Loss: {record_acc:.6f} at epoch {best_epoch}')

np.save(checkpoint_save_path + '/train_loss_record.npy', train_loss_record)
np.save(checkpoint_save_path + '/validate_loss_record.npy', validate_loss_record)

# Log final results to TensorBoard
writer.add_scalar('Final/best_validation_loss', best_val_loss, best_epoch)
writer.add_hparams({
    'channel': args.channel,
    'quant': args.quant,
    'bits': args.bits,
    'd_model': args.d_model,
    'dff': args.dff,
    'num_layers': args.num_layers,
    'num_heads': args.num_heads,
    'batch_size': args.batch_size,
    'epochs': args.epochs,
    'learning_rate': args.learning_rate
}, {
    'best_validation_loss': best_val_loss,
    'final_train_loss': train_loss_record[-1],
    'final_validation_loss': validate_loss_record[-1]
})

end = time.time()
td = end-start
hours = int(td // 3600)
minutes = int((td % 3600) // 60)
seconds = int(td % 60)
runtime_str = f'{hours}hours {minutes}minutes {seconds}seconds'
logger.info(f'Total runtime: {runtime_str}')

writer.close()
logger.info('TensorBoard writer closed.')
logger.info(f'All results saved to: {base_path}')
