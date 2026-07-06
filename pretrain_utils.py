# -*- coding: utf-8 -*-
"""
Utils for main_pretrain.py and main_HAP.py.
@Author: JW Deng
@Time: 2026/7/6 11:50
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import math
import torch
import time
import torch.nn as nn
import numpy as np
from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu
import re
import random

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class BleuScore():
    def __init__(self, w1, w2, w3, w4):
        self.w1 = w1 # 1-gram weights
        self.w2 = w2 # 2-grams weights
        self.w3 = w3 # 3-grams weights
        self.w4 = w4 # 4-grams weights
    
    def compute_blue_score(self, real, predicted):
        score = []
        for (sent1, sent2) in zip(real, predicted):
            sent1 = remove_tags(sent1).split()
            sent2 = remove_tags(sent2).split()
            score.append(sentence_bleu([sent1], sent2, 
                          weights=(self.w1, self.w2, self.w3, self.w4)))
        return score

class SeqtoText:
    def __init__(self, vocb_dictionary, end_idx):
        self.reverse_word_map = dict(map(reversed, vocb_dictionary))
        self.end_idx = end_idx

    def sequence_to_text(self, list_of_indices):
        # Looking up words in dictionary
        words = []
        for letter in list_of_indices:
            if letter == self.end_idx:
                break
            else:
                words.append(self.reverse_word_map.get(letter))
        words = ' '.join(words)
        return (words)

class Channels():

    def AWGN(self, Tx_sig, n_var):
        Rx_sig = Tx_sig + torch.normal(0., n_var, size=Tx_sig.shape).to(device)# n_var[0]
        return Rx_sig

    def Rayleigh(self, Tx_sig, n_var):
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

    def Rician(self, Tx_sig, n_var, K = 1):# default:K = 1
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (2 * (K + 1)))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(0, std, size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

def initNetParams(model):
    '''Init net parameters.'''
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
         
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    # 产生下三角矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)

def create_masks(src, trg, padding_idx):

    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]

    trg_mask = (trg == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]
    look_ahead_mask = subsequent_mask(trg.size(-1)).type_as(trg_mask.data)
    combined_mask = torch.max(trg_mask, look_ahead_mask)
    
    return src_mask.to(device), combined_mask.to(device)

def loss_function(x, trg, padding_idx, criterion):
    
    loss = criterion(x, trg)
    mask = (trg != padding_idx).type_as(loss.data)
    # a = mask.cpu().numpy()
    loss *= mask
    
    return loss.mean()

def PowerNormalize(x):
    
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
    # if power > 0:
        x = torch.div(x, power)
    
    return x

def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)

    return noise_std

def train_step_quant(model, src, trg, n_var, pad, opt, criterion, channel):
    model.train()

    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    channels = Channels()
    opt.zero_grad()

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    channel_enc_output_quant = model.quant_constellation(channel_enc_output)
    Tx_sig = PowerNormalize(channel_enc_output_quant)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")
    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred = model.dense(dec_output)
    ntokens = pred.size(-1)

    loss = loss_function(pred.contiguous().view(-1, ntokens),
                         trg_real.contiguous().view(-1),
                         pad, criterion)

    loss.backward()
    opt.step()

    return loss.item()

def update_quant(model, src, trg, pad, channel='AWGN', n_var=0.1):
    model.train()
    trg_inp = trg[:, :-1]
    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    model.quant_constellation.update_params(channel_enc_output)

    return model

def val_step_quant(model, src, trg, n_var, pad, criterion, channel):
    channels = Channels()
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    channel_enc_output_quant = model.quant_constellation(channel_enc_output)
    Tx_sig = PowerNormalize(channel_enc_output_quant)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")
    if hasattr(model, 'quant_RX'):
        if model.quant_RX:
            Rx_sig = model.quant_signaly(Rx_sig)
    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred = model.dense(dec_output)

    # pred = model(src, trg_inp, src_mask, look_ahead_mask, n_var)
    ntokens = pred.size(-1)
    loss = loss_function(pred.contiguous().view(-1, ntokens),
                         trg_real.contiguous().view(-1),
                         pad, criterion)
    # loss = loss_function(pred, trg_real, pad)

    return loss.item()
    
def greedy_decode_quant(model, src, n_var, max_len, padding_idx, start_symbol, channel):
    """
    Here we use greedy decoder, if better performance is needed, beam search decode can be used
    """
    # create src_mask
    channels = Channels()
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device) #[batch, 1, seq_len]

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    channel_enc_output_quant = model.quant_constellation(channel_enc_output)
    Tx_sig = PowerNormalize(channel_enc_output_quant)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    if hasattr(model, 'quant_RX'):
        if model.quant_RX:
            Rx_sig = model.quant_signaly(Rx_sig)
    memory = model.channel_decoder(Rx_sig)

    outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1):
        # create the decode mask
        trg_mask = (outputs == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]
        look_ahead_mask = subsequent_mask(outputs.size(1)).type(torch.FloatTensor)
#       print(look_ahead_mask)
        combined_mask = torch.max(trg_mask, look_ahead_mask)
        combined_mask = combined_mask.to(device)

        # decode the received signal
        dec_output = model.decoder(outputs, memory, combined_mask, None)
        pred = model.dense(dec_output)

        # predict the word
        prob = pred[: ,-1:, :]  # (batch_size, 1, vocab_size)
        #prob = prob.squeeze()

        # return the max-prob index
        _, next_word = torch.max(prob, dim = -1)
        #next_word = next_word.unsqueeze(1)

        #next_word = next_word.data[0]
        outputs = torch.cat([outputs, next_word], dim=1)

    return outputs

def greedy_decode_HAP(model, src, n_var, max_len, padding_idx, start_symbol, channel):
    # create src_mask
    channels = Channels()
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device) #[batch, 1, seq_len]

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    if hasattr(model, 'quant_constellation'):
        channel_enc_output = model.quant_constellation(channel_enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    memory = model.channel_decoder(Rx_sig)

    outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1):
        # create the decode mask
        trg_mask = (outputs == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]
        look_ahead_mask = subsequent_mask(outputs.size(1)).type(torch.FloatTensor)
#        print(look_ahead_mask)
        combined_mask = torch.max(trg_mask, look_ahead_mask)
        combined_mask = combined_mask.to(device)

        # decode the received signal
        dec_output = model.decoder(outputs, memory, combined_mask, None)
        pred = model.dense(dec_output)
        ntokens = model.ntokens
        # Create a new tensor with dimensions [batch_size, seq_len, ntokens], initialized with zeros
        expanded_pred = torch.zeros(pred.shape[:-1] + (ntokens,), device=pred.device, dtype=pred.dtype)

        # Distribute the pruned prediction values to specified positions
        expanded_pred[:, :, model.dense.out_indices] = pred

        # Update pred to the expanded tensor
        pred = expanded_pred
        # predict the word
        prob = pred[: ,-1:, :]  # (batch_size, 1, vocab_size)
        #prob = prob.squeeze()

        # return the max-prob index
        _, next_word = torch.max(prob, dim = -1)
        #next_word = next_word.unsqueeze(1)

        #next_word = next_word.data[0]
        outputs = torch.cat([outputs, next_word], dim=1)

    return outputs

def adjust_learning_rate(optimizer, lr):
    lr = lr * 0.1
    # Assume the first group is the quantization parameter group, which should use a smaller learning rate
    for i, p in enumerate(optimizer.param_groups):
        if i == 0:  # Quantization parameter group
            p['lr'] = lr / 10  # Maintain 1/10 ratio
        else:  # Other parameter groups
            p['lr'] = lr
    return lr

def save_model_weight(model, folder_path, epoch, max_files=4):
    """
    Save model weights with epoch info in the file name and manage file count.

    :param model: PyTorch model
    :param folder_path: Folder to save the model
    :param epoch: Current epoch number
    :param max_files: Maximum number of files to keep
    """
    # Ensure folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Get the list of all files in the folder
    files = os.listdir(folder_path)

    # Extract the epoch numbers from filenames and sort the files by epoch
    epoch_pattern = re.compile(r"checkpoint_(\d+)\.pth")  # Assuming filename format is 'epoch_X.pt'
    epoch_files = []

    for file in files:
        match = epoch_pattern.search(file)
        if match:
            epoch_files.append((int(match.group(1)), os.path.join(folder_path, file)))

    # Sort files by the extracted epoch number (ascending order, oldest epoch first)
    epoch_files.sort(key=lambda x: x[0])

    # If there are more files than the max_files, delete the one with the smallest epoch number
    if len(epoch_files) >= max_files:
        # Delete the file with the smallest epoch (oldest one)
        os.remove(epoch_files[0][1])  # Remove the file
        print(f"Deleted old file: {epoch_files[0][1]}")

    # Save the new model with the epoch number in the filename
    model_filename = f"checkpoint_{epoch}.pth"
    model_path = os.path.join(folder_path, model_filename)

    # Save the model's state_dict (weights)
    with open(model_path, 'wb') as f:
        torch.save(model.state_dict(), f)

    print(f"Model saved to {model_path}")

def load_model(transformer, PATH):

    model_paths = []
    for fn in os.listdir(PATH):
        if not fn.endswith('.pth'): continue
        match = re.search(r'checkpoint_(\d+)\.pth', fn)
        if match:
            idx = int(match.group(1))
            model_paths.append((os.path.join(PATH, fn), idx))
    model_paths.sort(key=lambda x: x[1])  # sort the image by the idx
    model_path, _ = model_paths[-1]# -1
    checkpoint = torch.load(model_path)
    transformer.load_state_dict(checkpoint)
    print(f'model loaded from  {model_path}')

def save_results_to_file(args, SNR, bleu_score1, bleu_score2):
    """
    save performance results to .txt file
    """
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    results_path = os.path.join(args.checkpoint_path, 'performance_results.txt')

    with open(results_path, 'w') as f:
        f.write('=' * 50 + '\n')
        f.write('Performance Results\n')
        f.write('=' * 50 + '\n')
        f.write(f'Channel: {args.channel}\n')
        f.write(f'Quantization: {args.quant}\n')
        f.write(f'Bits: {args.bits}\n')
        f.write(f'Test_epochs: {args.Test_epochs}\n')
        f.write(f'Test Time: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('=' * 50 + '\n\n')

        f.write('SNR values: [{}]\n'.format(','.join(str(x) for x in SNR)))
        f.write('BLEU-1 scores: [{}]\n'.format(','.join('{:.5f}'.format(x) for x in bleu_score1)))
        f.write('BLEU-4 scores: [{}]\n'.format(','.join('{:.5f}'.format(x) for x in bleu_score2)))

        f.write('\nDetailed Results:\n')
        f.write('-' * 30 + '\n')
        for i, snr in enumerate(SNR):
            f.write('SNR {:2d} dB: BLEU-1 = {:.5f}, BLEU-4 = {:.5f}\n'.format(
                snr, bleu_score1[i], bleu_score2[i]))

        avg_bleu1 = np.mean(bleu_score1)
        avg_bleu2 = np.mean(bleu_score2)
        f.write('-' * 30 + '\n')
        f.write('Average BLEU-1: {:.5f}\n'.format(avg_bleu1))
        f.write('Average BLEU-4: {:.5f}\n'.format(avg_bleu2))

    print(f'Results saved to {results_path}')
    return results_path