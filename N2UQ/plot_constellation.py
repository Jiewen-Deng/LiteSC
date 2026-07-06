"""
Plot constellation of uniform and nonuniform-to-uniform quantization.
@Author: JW Deng
@Time: 2026/1/27 19:20
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from tqdm import tqdm
from N2UQ_transformer import N2UQTransformer
from transformer_constellation import *
import json
from dataset import EurDataset, collate_data
from torch.utils.data import DataLoader
import argparse
from matplotlib import pyplot as plt
# from utils import*
from utils import *
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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

def forward_to_constellation(model, src, n_var, max_len, padding_idx):
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)  # [batch, 1, seq_len]

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)

    a = PowerNormalize(channel_enc_output)
    a = a.cpu().detach().numpy()
    # np.save("riginal.npy", a)

    channel_enc_output = model.quant_constellation(channel_enc_output)
    if hasattr(model.quant_constellation, 'quantize'):
        out_tensor = model.quant_constellation.quantize(channel_enc_output).cpu().detach().numpy()
        out_tensor = (out_tensor - 0) * (1 - (-1)) / (3 - 0) + (-1)
        np.save("out_tensor_quant.npy", out_tensor)
    else:
        out_tensor = (model.quant_constellation.scale1 * channel_enc_output).cpu().detach().numpy()
        np.save("out_tensor_N2UQ.npy", out_tensor)

    return out_tensor

def calculate_frequency_numpy(constellation_array):
    flattened = constellation_array.flatten()
    valid_values = flattened[flattened != -15]
    unique_values, counts = np.unique(valid_values, return_counts=True)
    frequencies = counts / np.sum(counts)

    return unique_values, counts, frequencies

parser = argparse.ArgumentParser(description='Model Hyperparameters')
# parser.add_argument('--quant-path', default='quant', type=str)
parser.add_argument('--bits', default=2, type=int, help='Please choose 2, 4, 6, 8, 10bits')
parser.add_argument('--quant', default='N2UQ_Symmetric', type=str,
                    help='Please choose  N2UQ_Symmetric, N2UQ_Asymmetric, quant')
parser.add_argument('--channel', default='AWGN', type=str, help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--network', type=str, default='N2UQTransformer', required=False,
                    help='Please choose Transformer, QuantTransformer, N2UQTransformer')
parser.add_argument('--ratio', type=str, default="0.5", required=False, help='0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99')
parser.add_argument('--hessian-mode', type=str, default='trace', required=False,
                    help='choose trace, random or magnitude')

# Data parameters
parser.add_argument('--vocab-file', type=str, default='../europarl/europarl_origin_cut/vocab.json',
                    help='Path to vocabulary file')
# Model architecture parameters
parser.add_argument('--num-layers', type=int, default=4, help='Number of layers')
parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
parser.add_argument('--dff', type=int, default=512, help='Feed forward dimension')
parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
parser.add_argument('--dropout-rate', type=float, default=0.1, help='Dropout rate')
parser.add_argument('--MAX-LENGTH', type=int, default=30, help='Maximum sequence length')
parser.add_argument('--MIN-LENGTH', type=int, default=4, help='Minimum sequence length')

# Batch sizes
parser.add_argument('--batch-size-test', type=int, default=64, help='Test batch size')
parser.add_argument('--batch-size-train', type=int, default=128, help='Training batch size')
args = parser.parse_args()
args.checkpoint_path = f'./pretrain/{args.channel}_{args.quant}/{args.bits}bits'
test_eur = EurDataset('test')
test_iterator = DataLoader(test_eur, batch_size=args.batch_size_test, num_workers=0,
                           pin_memory=True, collate_fn=collate_data)
SNR = [18]
bleu_score_1gram = BleuScore(1, 0, 0, 0)

vocab = json.load(open(args.vocab_file, 'rb'))
token_to_idx = vocab['token_to_idx']
num_vocab = len(token_to_idx)
pad_idx = token_to_idx["<PAD>"]
start_idx = token_to_idx["<START>"]
end_idx = token_to_idx["<END>"]
vocb_dictionary = token_to_idx.items()
StoT = SeqtoText(vocb_dictionary, end_idx)
src_pad_idx = token_to_idx["<PAD>"]
# load_model(transformer, PATH_FOLDER)
transformer = Transformer(args.num_layers, num_vocab, num_vocab,
                     num_vocab, num_vocab, args.d_model, args.num_heads,
                     args.dff, 0.1, args.quant, args.bits).to(device)
load_model(transformer, args.checkpoint_path)
# transformer, checkpoint, epoch_origin = load_network(args.checkpoint_path)

out_tensor_list = []
for snr in SNR:
    noise_std = SNR_to_noise(snr)

    for index_batch, src in enumerate(tqdm(test_iterator)):
        # for i, src in enumerate(test_iterator):
        transformer.eval()
        src = src.to(device)
        target = src

        out_tensor = forward_to_constellation(transformer, src, noise_std, args.MAX_LENGTH, src_pad_idx)
        out_tensor_list.append(out_tensor)
    break

# constellation_load = out_tensor
# constellation_load = np.load("out_tensor_N2UQ.npy")
# constellation_list = []
all_constellation_points = []
for index_batch, src_batch in enumerate(tqdm(test_iterator)):
    src_batch = src_batch.to(device)
    constellation_batch = out_tensor_list[index_batch]
    for i in range(len(src_batch)):
        for j in range(len(src_batch[i])):
            if src_batch[i][j] > 4:
                try:
                    vector = constellation_batch[i][j]
                    # get constellation points
                    constellation = [vector[m] + 1j * vector[m + 1] for m in range(0, len(vector), 2)]
                    all_constellation_points.extend(constellation)
                except IndexError:
                    continue

# Count constellation point frequencies
constellation_counter = {}
for point in all_constellation_points:
    # Quantize point positions for grouped statistics

    quantized_point = complex(round(point.real, 3), round(point.imag, 3))
    constellation_counter[quantized_point] = constellation_counter.get(quantized_point, 0) + 1

print(constellation_counter)
# plot constellation diagram
if constellation_counter:
    points = list(constellation_counter.keys())
    frequencies = list(constellation_counter.values())
    max_freq = 1.6e5
    # max_freq = max(frequencies)

    x_coords = [p.real for p in points]
    y_coords = [p.imag for p in points]

    # Radius is proportional to frequency
    sizes = [50 + (freq / max_freq) * 1000 for freq in frequencies]  # Base size 50, maximum magnification to 550

    plt.figure(figsize=(10, 8))
    plt.scatter(x_coords, y_coords, s=sizes, alpha=0.6, c=frequencies, cmap='plasma', vmin=0, vmax=max_freq)
    plt.colorbar(label='Frequency')
    plt.xlabel('In-phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.title('Constellation Diagram (Point size represents frequency)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.savefig(args.checkpoint_path +'/constellation_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
