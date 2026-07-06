"""
@Author: Jiewen Deng
@Time: 2026/7/6 16:48
"""
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import argparse
from tqdm import tqdm

# model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('C:\\Users\\dengjiewen\\PycharmProjects\\LDSC_github\\bert\\bert-base-uncased',
                            device=device)

def save_results_to_file(PATH_FOLDER, SNR, mean_bertsimilarity):
    """
    save performance results to .txt file
    """
    results_path = os.path.join(PATH_FOLDER, 'performance_results.txt')

    with open(results_path, 'a') as f:
        f.write('\n' + '=' * 50 + '\n')
        import time
        f.write(f'Test Time: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('SNR values: [{}]\n'.format(','.join(str(x) for x in SNR)))
        f.write('epoch values: [{}]\n'.format(','.join(str(x) for x in epoch_range)))
        f.write('bert-similarity scores: [{}]\n'.format(','.join('{:.5f}'.format(x) for x in mean_bertsimilarity)))

        # calculate the average
        avg_bs = np.mean(mean_bertsimilarity)
        f.write('-' * 30 + '\n')
        f.write('Average bert-similarity scores: {:.5f}\n'.format(avg_bs))

    print(f'Results saved to {results_path}')
    return results_path

SNR_list = [-4, -2, 0, 2, 4, 6, 8, 10, 12, 14]
# SNR_list = [-8, -6, -4, -2]
# epoch_range = [0,1,2]
# SNR_list = [0]
epoch_range = [0]
# epoch_range = [0, 1, 2, 3, 4]
bertsimilarity = np.zeros((len(epoch_range),len(SNR_list)))
parser = argparse.ArgumentParser()
# parser.add_argument('--parent_dir', default='/media/bcng/hdd/LiuYueling/DeepSC-master_context/', type=str)
parser.add_argument('--BASE-PATH', default='C:/Users/dengjiewen/PycharmProjects/Traditional Method/', type=str)#Digital_SC
parser.add_argument('--transmit-text', default="C:/Users/dengjiewen/PycharmProjects/Digital_SC/europarl/europarl_origin_cut/test_data_batchsize64.txt", type=str)
# parser.add_argument('--transmit-text', default="C:\\Users\\dengjiewen\\PycharmProjects\\Traditional Method\\transmitter\\europarl\\transmit.txt", type=str)
parser.add_argument('--valid-path', default="Huffman_LDPC_8PSK_forword/europarl/AWGN", type=str)
parser.add_argument('--model-name', default='bert/bert-base-uncased', type=str)
args = parser.parse_args()

# checkpoint_paths = []
# for checkpoint_path in checkpoint_paths:
path_folder = os.path.join(args.BASE_PATH, args.valid_path)
args.receive_text = os.path.join(path_folder, 'trans{}_{}.txt')
# path_folder = 'C:\\Users\\dengjiewen\\PycharmProjects\\Traditional Method\\Huffman_LDPC_8PSK_forword\\europarl\\Rayleigh\\'
# args.receive_text = os.path.join(path_folder, 'trans{}_{}.txt')
print('=' * 60)
print('Configuration:')
print(f"BASE-PATH:        {args.BASE_PATH}")
print(f"transmit-text:    {args.transmit_text}")
print(f"valid-path:       {args.valid_path}")
print(f"model-name:       {args.model_name}")
print(f"receive-text:     {args.receive_text}")
print(f"SNR_list:         {SNR_list}")
print(f"epoch_range:      {epoch_range}")
print('=' * 60)

for epoch_j, epoch in enumerate(tqdm(epoch_range)):
    for SNR_i, SNR_db in enumerate(tqdm(SNR_list)):
        # with open(args.transmit_text, mode='r') as fr:
        with open(args.transmit_text, mode='r') as fr:
            textm = fr.readlines()
        # with open(args.receive_text.format(SNR_db), mode='r') as fwb:
        with open(args.receive_text.format(SNR_db,epoch), mode='r') as fwb:
        # with open(args.receive_text.format(SNR_db), mode='r') as fwb:
            ansm_raw = fwb.readlines()
            ansm = [ansm_raw[i][:] for i in range(len(ansm_raw))]
            # ansm = [ansm_raw[i][8:] for i in range(len(ansm_raw))] # exclude the beginning '<START> '
        sim_score = []
        for i in range(len(ansm)):
            #predictions = web_model.predict([(textm[i], ansm[i])])
            sentences = [textm[i], ansm[i]]
            embeddings = model.encode(sentences)
            predictions = util.cos_sim(embeddings[0], embeddings[1])
            sim_score.append(predictions[0][0])
        sim_score_mean = sum(sim_score) / len(sim_score)
        bertsimilarity[epoch_j, SNR_i] = sim_score_mean

        print({f'SNR_{SNR_db}dB': f'{sim_score_mean:.4f}'})
    # Summary information
    print(f"Epoch {epoch} finished | SNR results:")
    for snr_idx, snr_val in enumerate(SNR_list):
        print(f"  SNR {snr_val:>4}dB: {bertsimilarity[epoch_j, snr_idx]:.4f}")
    print(f"  average: {np.mean(bertsimilarity[epoch_j]):.4f}")

mean_bertsimilarity = np.mean(bertsimilarity, axis=0)
print('bertsimilarity = [{}]'.format(','.join('{:.5f}'.format(x) for x in mean_bertsimilarity)))
save_results_to_file(path_folder, SNR_list, mean_bertsimilarity)