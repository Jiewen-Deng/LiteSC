# -*- coding: utf-8 -*-
"""
Utils for main_N2UQ.py
@Author: JW Deng
@Time: 2026/1/27 19:20
"""
from pretrain_utils import *

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def build_quantized_model(transformer, new_dict):
    for m_name, m in transformer.named_modules():
        if isinstance(m, nn.Linear):
            weight_name = m_name + '.weight'
            new_weight = new_dict[weight_name]
            len_out, len_in= new_weight.data.shape
            m.weight.data = m.weight.data[:len_out, :len_in].clone()
            m.in_features = len_in
            m.out_features = len_out

            # Correct way: create new parameters instead of directly replacing .data
            with torch.no_grad():
                new_weight_tensor = m.weight.data[:len_out, :len_in].clone()
                m.weight = nn.Parameter(new_weight_tensor)

            if m.bias is not None:
                with torch.no_grad():
                    new_bias_tensor = m.bias.data[:len_out].clone()
                    m.bias = nn.Parameter(new_bias_tensor)

        elif isinstance(m, nn.LayerNorm):
            weight_name = m_name + '.weight'
            new_weight = new_dict[weight_name]
            len_in = new_weight.data.shape[0]
            m.normalized_shape = (len_in,)
            with torch.no_grad():
                new_weight_tensor = m.weight.data[:len_in].clone()
                m.weight = nn.Parameter(new_weight_tensor)

                new_bias_tensor = m.bias.data[:len_in].clone()
                m.bias = nn.Parameter(new_bias_tensor)

def save_model(args, model, avg_test_loss, folder_path, epoch, out_indices=range(22234), max_files=1):
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
    epoch_pattern = re.compile(r"checkpoint_(\d+)\.pth.tar")  # Assuming filename format is 'epoch_X.pt'
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
    model_filename = f"checkpoint_{epoch}.pth.tar"
    model_path = os.path.join(folder_path, model_filename)

    # Save the model's state_dict (weights)
    torch.save( {'args': args,
                    'net': model,
                    'avg_test_loss': avg_test_loss,
                    'out_indices': out_indices}, model_path)

    print(f"Model saved to {model_path}")

def update_quant(model, src, trg, pad):
    model.train()
    trg_inp = trg[:, :-1]
    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    model.quant_constellation.update_params(channel_enc_output)


def calibrate_all_quantizers(model, src, trg, pad):
    """
    Complete calibration process: traverse all quantizers in the model and collect statistical information.
    """
    model.train()
    with torch.no_grad():
        trg_inp = trg[:, :-1]
        src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)

        # Complete forward propagation, triggering the range_tracker of all QuantizedLinear layers.
        enc_output = model.encoder(src, src_mask)
        channel_enc_output = model.channel_encoder(enc_output)
        channel_enc_output_quant = model.quant_constellation(channel_enc_output)
        Tx_sig = PowerNormalize(channel_enc_output_quant)
        Rx_sig = Tx_sig  # add no noise
        channel_dec_output = model.channel_decoder(Rx_sig)
        dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
        pred = model.dense(dec_output)

    return model

'''
# analyse the time of each train step
def train_step(model, src, trg, n_var, pad, optimizer, criterion,
               channel, out_indices):
    import time
    t_start = time.time()

    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]
    channels = Channels()
    optimizer.zero_grad()

    t1 = time.time()
    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
    enc_output = model.encoder(src, src_mask)
    t2 = time.time()

    channel_enc_output = model.channel_encoder(enc_output)
    channel_enc_output = model.quant_constellation(channel_enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)
    t3 = time.time()

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")
    t4 = time.time()

    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred = model.dense(dec_output)
    t5 = time.time()

    ntokens = model.ntokens
    expanded_pred = torch.zeros(pred.shape[:-1] + (ntokens,), device=pred.device, dtype=pred.dtype)
    expanded_pred[:, :, out_indices] = pred
    pred = expanded_pred
    loss = loss_function(pred.contiguous().view(-1, ntokens),
                         trg_real.contiguous().view(-1),
                         pad, criterion)
    t6 = time.time()

    loss.backward()
    optimizer.step()
    t7 = time.time()

    # print the first batch
    if not hasattr(train_step, '_timed'):
        print("\n=== Training Step Timing ===")
        print(f"Preprocess:          {t1-t_start:.3f}s")
        print(f"Encoder:             {t2-t1:.3f}s")
        print(f"Channel+Quant:       {t3-t2:.3f}s  <-- Check this!")
        print(f"Channel Noise:       {t4-t3:.3f}s")
        print(f"Decoder+Dense:       {t5-t4:.3f}s")
        print(f"Loss Computation:    {t6-t5:.3f}s")
        print(f"Backward+Update:     {t7-t6:.3f}s")
        print(f"TOTAL:               {t7-t_start:.3f}s")
        print("=" * 40 + "\n")
        train_step._timed = True

    return loss.item()
'''

def train_step(model, src, trg, n_var, pad, optimizer, criterion,
               channel, out_indices):
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]
    channels = Channels()
    optimizer.zero_grad()

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
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
    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred = model.dense(dec_output)
    ntokens = model.ntokens
    # Create a new tensor with dimensions [batch_size, seq_len, ntokens], initialized with zeros
    expanded_pred = torch.zeros(pred.shape[:-1] + (ntokens,), device=pred.device, dtype=pred.dtype)
    # Distribute the pruned prediction values to specified positions
    expanded_pred[:, :, out_indices] = pred
    # Update pred to the expanded tensor
    pred = expanded_pred
    loss = loss_function(pred.contiguous().view(-1, ntokens),
                         trg_real.contiguous().view(-1),
                         pad, criterion)

    loss.backward()
    optimizer.step()

    return loss.item()

def test_step(model, src, trg, n_var, pad, criterion, channel, out_indices):
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]
    channels = Channels()
    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
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
    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred = model.dense(dec_output)
    ntokens = model.ntokens
    # Create a new tensor with dimensions [batch_size, seq_len, ntokens], initialized with zeros
    expanded_pred = torch.zeros(pred.shape[:-1] + (ntokens,), device=pred.device, dtype=pred.dtype)
    # Distribute the pruned prediction values to specified positions
    expanded_pred[:, :, out_indices] = pred
    # Update pred to the expanded tensor
    pred = expanded_pred
    # ntokens = pred.size(-1)
    loss = loss_function(pred.contiguous().view(-1, ntokens),
                         trg_real.contiguous().view(-1),
                         pad, criterion)

    return loss.item()


def greedy_decode(model, src, n_var, max_len, padding_idx, start_symbol,
                  channel, out_indices):
    channels = Channels()
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)  # [batch, 1, seq_len]
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)

    a = PowerNormalize(channel_enc_output)
    a = a.cpu().detach().numpy()
    # np.save("riginal.npy", a)
    channel_enc_output = model.quant_constellation(channel_enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)
    b = Tx_sig.cpu().detach().numpy()
    # np.save("dequantize.npy", b)

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
        trg_mask = (outputs == padding_idx).unsqueeze(-2).type(torch.FloatTensor)  # [batch, 1, seq_len]
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
        expanded_pred[:, :, out_indices] = pred
        # Update pred to the expanded tensor

        pred = expanded_pred
        # predict the word
        prob = pred[:, -1:, :]  # (batch_size, 1, vocab_size)
        # prob = prob.squeeze()

        # return the max-prob index
        _, next_word = torch.max(prob, dim=-1)
        # next_word = next_word.unsqueeze(1)

        # next_word = next_word.data[0]
        outputs = torch.cat([outputs, next_word], dim=1)

    return outputs

def load_model_state_dict(PATH):
    model_paths = []
    for fn in os.listdir(PATH):
        if not fn.endswith('.pth'): continue
        match = re.search(r'checkpoint_(\d+)\.pth', fn)
        if match:
            idx = int(match.group(1))
            model_paths.append((os.path.join(PATH, fn), idx))
    model_paths.sort(key=lambda x: x[1])  # sort the image by the idx
    model_path, _ = model_paths[-1]
    checkpoint = torch.load(model_path)

    return checkpoint

def load_network(PATH):
    model_paths = []
    for fn in os.listdir(PATH):
        if not fn.endswith('.pth.tar'): continue
        match = re.search(r'checkpoint_(\d+)\.pth\.tar', fn)
        if match:
            idx = int(match.group(1))
            model_paths.append((os.path.join(PATH, fn), idx))
    model_paths.sort(key=lambda x: x[1])  # sort the image by the idx
    model_path, epoch_origin = model_paths[-1]
    model_tar = torch.load(model_path, map_location=device)
    deepsc = model_tar['net'].to(device)  #
    print(f'model loaded from  {model_path}')

    return deepsc, model_tar, epoch_origin

def load_model(transformer, PATH):
    model_paths = []
    for fn in os.listdir(PATH):
        if not fn.endswith('.pth.tar'): continue
        # match = re.search(r'checkpoint_(\d+)\.pth', fn)
        match = re.search(r'checkpoint_(\d+)\.pth\.tar', fn)
        if match:
            idx = int(match.group(1))
            model_paths.append((os.path.join(PATH, fn), idx))
    model_paths.sort(key=lambda x: x[1])  # sort the image by the idx
    model_path, _ = model_paths[-1]
    checkpoint = torch.load(model_path)
    transformer.load_state_dict(checkpoint['net'].state_dict())
    # transformer.load_state_dict(checkpoint['state_dict'])
    print(f'model loaded from  {model_path}')

def save_results_to_file(args, SNR, bleu_score1):
    results_path = os.path.join(args.valid_path, 'performance_results.txt')

    with open(results_path, 'a') as f:
        f.write('=' * 50 + '\n')
        f.write('Performance Results\n')
        f.write('=' * 50 + '\n')
        f.write(f'Channel: {args.channel}\n')
        f.write(f'Quantization: {args.quant}\n')
        f.write(f'Bits: {args.bits}\n')
        f.write(f'Network: {args.network}\n')
        f.write(f'Ratio: {args.ratio}\n')
        f.write(f'Valid Path: {args.valid_path}\n')
        import time
        f.write(f'Test Time: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('=' * 50 + '\n\n')

        f.write('SNR values: [{}]\n'.format(','.join(str(x) for x in SNR)))
        f.write('BLEU-1 scores: [{}]\n'.format(','.join('{:.5f}'.format(x) for x in bleu_score1)))

        avg_bleu1 = np.mean(bleu_score1)
        f.write('-' * 30 + '\n')
        f.write('Average BLEU-1: {:.5f}\n'.format(avg_bleu1))

    print(f'Results saved to {results_path}')
    return results_path

def save_results_to_file_2metric(args, SNR, bleu_score1, bertsim_score):
    results_path = os.path.join(args.valid_path, 'performance_results.txt')

    with open(results_path, 'w') as f:
        f.write('=' * 50 + '\n')
        f.write('Performance Results\n')
        f.write('=' * 50 + '\n')
        f.write(f'Channel: {args.channel}\n')
        f.write(f'Quantization: {args.quant}\n')
        f.write(f'Bits: {args.bits}\n')
        f.write(f'Network: {args.network}\n')
        f.write(f'Ratio: {args.ratio}\n')
        f.write(f'Valid Path: {args.valid_path}\n')
        import time
        f.write(f'Test Time: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('=' * 50 + '\n\n')

        f.write('SNR values: [{}]\n'.format(','.join(str(x) for x in SNR)))
        f.write('BLEU-1 scores         : [{}]\n'.format(','.join('{:.5f}'.format(x) for x in bleu_score1)))
        f.write('BERT Similarity scores: [{}]\n'.format(','.join('{:.5f}'.format(x) for x in bertsim_score)))

        avg_bleu1 = np.mean(bleu_score1)
        avg_bleu2 = np.mean(bertsim_score)
        f.write('-' * 30 + '\n')
        f.write('Average BLEU-1               : {:.5f}\n'.format(avg_bleu1))
        f.write('Average BERT Similarity score: {:.5f}\n'.format(avg_bleu2))

    print(f'Results saved to {results_path}')
    return results_path