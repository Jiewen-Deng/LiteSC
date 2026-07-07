"""
@Author: Jiewen Deng
@Time: 2026/7/6 16:48
"""
import re
import json
import torch
import argparse
import time
import numpy as np
from dataset import EurDataset, collate_data
from torch.utils.data import DataLoader
from tqdm import tqdm

from pretrain_utils import *
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

def performance(args, SNR, net,epochs):
    bleu_score_1gram = BleuScore(1, 0, 0, 0)
    bleu_score_4gram = BleuScore(0.25, 0.25, 0.25, 0.25)
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
   # test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0)

    StoT = SeqtoText(token_to_idx, end_idx)
    score1 = []
    score2 = []
    # score3 = []
    net.eval()
    with torch.no_grad():
        for epoch in epochs:#range(args.epochs):
            Tx_word = []
            Rx_word = []

            for snr in SNR:
                print('test SNR（dB）:    ', snr)
            # for snr in tqdm(SNR):
                word = []
                target_word = []
                noise_std = SNR_to_noise(snr)

                for sents in tqdm(test_iterator):

                    sents = sents.to(device)
                    # src = batch.src.transpose(0, 1)[:1]
                    target = sents

                    out = greedy_decode_HAP(net, sents, noise_std, args.MAX_LENGTH, pad_idx,
                                        start_idx, args.channel)

                    sentences = out[:,1:].cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, sentences))
                    word = word + result_string

                    target_sent = target[:,1:].cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, target_sent))
                    target_word = target_word + result_string

                Rx_word.append(word)
                Tx_word.append(target_word)
                valid_path = os.path.join(args.checkpoint_path, 'valid')
                if not os.path.exists(valid_path):
                    os.makedirs(valid_path)
                # valid_path = os.path.join(args.valid_path,'trans{}_{}.txt'.format(snr,epoch))
                with open(os.path.join(valid_path,'trans{}_{}.txt'.format(snr,epoch)), 'w') as f:
                    for line in word:
                        f.write('%s\n' % line)

            bleu_score1 = []
            bleu_score2 = []
            sim_score = []
            for sent1, sent2 in zip(Tx_word, Rx_word):
                # 1-gram
                bleu_score1.append(bleu_score_1gram.compute_blue_score(sent2, sent1)) # 7*num_sent
                bleu_score2.append(bleu_score_4gram.compute_blue_score(sent2, sent1))
                # sim_score.append(similarity.compute_similarity(sent1, sent2)) # 7*num_sent
            bleu_score1 = np.array(bleu_score1)
            bleu_score1 = np.mean(bleu_score1, axis=1)
            score1.append(bleu_score1)

            bleu_score2 = np.array(bleu_score2)
            bleu_score2 = np.mean(bleu_score2, axis=1)
            score2.append(bleu_score2)

            # sim_score = np.array(sim_score)
            # sim_score = np.mean(sim_score, axis=1)
            # score3.append(sim_score)

    score1 = np.mean(np.array(score1), axis=0)
    score2 = np.mean(np.array(score2), axis=0)
    # score3 = np.mean(np.array(score3), axis=0)

    return score1, score2#, score3

def load_network(PATH):
    model_paths = []
    for fn in os.listdir(PATH):
        if not fn.endswith('.pth.tar'): continue
        match = re.search(r'checkpoint_(\d+)\.pth\.tar', fn)
        if match:
            idx = int(match.group(1))
            model_paths.append((os.path.join(PATH, fn), idx))
    model_paths.sort(key=lambda x: x[1])  # sort the image by the idx
    model_path, _ = model_paths[-1]
    model_tar = torch.load(model_path)
    DSC_HAP = model_tar['net'].to(device)
    print(f'model loaded from  {model_path}')

    return DSC_HAP

if __name__ == '__main__':
    data_dir = 'C:/Users/dengjiewen/PycharmProjects/Digital_SC/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab-file', default='europarl/europarl_origin_cut/vocab.json', type=str)
    parser.add_argument('--bits', default=4, type=int, help='Please choose 1, 2, 4, 8,16bits')
    parser.add_argument('--quant', default='N2UQ_Symmetric', type=str, help='Please choose  N2UQ_Symmetric, N2UQ_Asymmetric, quant')
    parser.add_argument('--channel', default='AWGN', type=str, help='Please choose AWGN, Rayleigh, and Rician')
    parser.add_argument('--network', type=str, default='Transformer', required=False,
                        help='Please choose DeepSC, Transformer, QuantTransformer, N2UQTransformer')
    parser.add_argument('--ratio', type=str, default="0.5", required=False)
    parser.add_argument('--hessian-mode', type=str, default='trace', required=False,
                        help='choose trace, random or magnitude')

    parser.add_argument('--MAX-LENGTH', default=30, type=int)
    parser.add_argument('--d-model', default=128, type=int)
    parser.add_argument('--dff', default=512, type=int)
    parser.add_argument('--num-layers', default=4, type=int)
    parser.add_argument('--num-heads', default=8, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--Test-epochs', default=5, type=int)
    setup_seed(10)
    start = time.time()

    args = parser.parse_args()
    if args.network == 'Transformer':
        args.checkpoint_path = f"./HAP/out/{args.channel}_{args.quant}/pr_{args.ratio}_{args.bits}bits_{args.hessian_mode}/"# _{args.bits}bits
    else:
        args.checkpoint_path = f"./HAP/out/{args.channel}_{args.network}/pr_{args.ratio}_{args.hessian_mode}/"# snr_-4_10/
    DSC_HAP = load_network(args.checkpoint_path)
    SNR = [-8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14]
    if args.channel == 'AWGN':
        args.Test_epochs = 2
        # SNR = [-4, -2, 0, 2, 4, 6, 8, 10]
    print('SRN: ',SNR)
    epochs = range(args.Test_epochs)

    args.vocab_file = data_dir + args.vocab_file
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    """ define optimizer and loss function """
    bleu_score1,bleu_score2= performance(args, SNR, DSC_HAP, epochs) #, sim_score
    print('[{}]'.format(','.join('{:.5f}'.format(x) for x in bleu_score1)))# bleu_score1gram =
    print('[{}]'.format(','.join('{:.5f}'.format(x) for x in bleu_score2)))# bleu_score4gram =
    #similarity.compute_similarity(sent1, real)
    save_results_to_file(args, SNR, bleu_score1, bleu_score2)

    end = time.time()
    td = end-start
    hours = int(td // 3600)
    minutes = int((td % 3600) // 60)
    seconds = int(td % 60)
    print(f'runtime: {hours}hours {minutes}minutes {seconds}seconds')
