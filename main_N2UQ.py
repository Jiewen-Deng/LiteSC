"""
Perform nonuniform-to-uniform quantization on HAPed semantic system.
@Author: JW Deng
@Time: 2026/7/6 12:10
"""
import time
import torch
from tqdm import tqdm
from N2UQ.N2UQ_transformer import N2UQTransformer
from N2UQ.transformer_constellation import Transformer
import json
from dataset import EurDataset, collate_data
from torch.utils.data import DataLoader
import argparse
from sentence_transformers import SentenceTransformer, util
from pretrain_utils import *
from N2UQ.utils import *
from HAP.utils.common_utils import get_logger, makedirs
from tensorboardX import SummaryWriter

running_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Model Hyperparameters')
# parser.add_argument('--quant-path', default='quant', type=str)
parser.add_argument('--bits', default=4, type=int, help='Please choose 2, 4, 6, 8, 10bits')
parser.add_argument('--quant', default='N2UQ_Symmetric', type=str,
                    help='Please choose  N2UQ_Symmetric, N2UQ_Asymmetric, quant')
parser.add_argument('--channel', default='AWGN', type=str, help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--network', type=str, default='N2UQTransformer', required=False,
                    help='Please choose Transformer, QuantTransformer, N2UQTransformer')
parser.add_argument('--ratio', type=str, default="0.5", required=False, help='0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99')
parser.add_argument('--hessian-mode', type=str, default='trace', required=False,
                    help='choose trace, random or magnitude')

# Data parameters
parser.add_argument('--vocab-file', type=str, default='../Digital_SC/europarl/europarl_origin_cut/vocab.json',
                    help='Path to vocabulary file')
# Model architecture parameters
parser.add_argument('--num-layers', type=int, default=4, help='Number of layers')
parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
parser.add_argument('--dff', type=int, default=512, help='Feed forward dimension')
parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
parser.add_argument('--dropout-rate', type=float, default=0.1, help='Dropout rate')
parser.add_argument('--MAX-LENGTH', type=int, default=30, help='Maximum sequence length')
# Training modes
parser.add_argument('--stastic-quantization', action='store_true', default=True,
                    help='Enable static quantization')
parser.add_argument('--QAT', action='store_true', default=True,
                    help='Load QAT model')
parser.add_argument('--QAT-epochs', type=int, default=30, help='QAT training epochs')
parser.add_argument('--learning-rate', type=float, default=1e-3, help='QAT learning rate')
# Batch sizes
parser.add_argument('--batch-size-test', type=int, default=64, help='Test batch size')
parser.add_argument('--batch-size-train', type=int, default=128, help='Training batch size')
# Testing
parser.add_argument('--Test', action='store_true', default=True, help='Enable testing')
parser.add_argument('--Test-epochs',type=int, default=5, help='Test epochs')
parser.add_argument('--seed',type=int, default=10, help='Setup seed')
parser.add_argument('--model-name', default='C:/Users/dengjiewen/PycharmProjects/roberta-large-nli-stsb-mean-tokens', type=str)
# SNR values (as comma-separated string)
args = parser.parse_args()
setup_seed(args.seed)
""" preparing the dataset """
train_eur = EurDataset('train')
train_iterator = DataLoader(train_eur, batch_size=args.batch_size_train, num_workers=0,
                            pin_memory=True, collate_fn=collate_data)
test_eur = EurDataset('test')
test_iterator = DataLoader(test_eur, batch_size=args.batch_size_test, num_workers=0,
                           pin_memory=True, collate_fn=collate_data)
SNR = [-8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14]
# SNR = [14, 12, 10, 8, 6, 4, 2, 0, -2, -4, -6, -8]

if args.channel == 'AWGN':
    args.Test_epochs = 2

# Initialize logging and summary writer
# base_exp_name = f"Rician_{args.network}/pr{args.ratio}_{args.bits}bits_{args.hessian_mode}"
base_exp_name = f"{args.channel}_{args.network}/pr{args.ratio}_{args.bits}bits_{args.hessian_mode}"
# process_tags = ['search']
process_tags = []

if args.stastic_quantization:
    process_tags.append(f"SQ")

if args.QAT:
    process_tags.append(f"QAT{args.QAT_epochs}ep_lr{args.learning_rate}")

if args.Test:
    process_tags.append(f"Test{args.Test_epochs}ep")
process_tags.append(f"seed{args.seed}")

if process_tags:
    args.exp_name = base_exp_name + "_" + "_".join(process_tags)
else:
    args.exp_name = base_exp_name

# args.exp_name = 'Rician_N2UQTransformer/pr0.5_4bits_trace_SQ_QAT30ep_lr0.001_Test5ep'
runs_name = f'./runs/N2UQ/{args.exp_name}'
summary_dir = runs_name + f"/summary/test{args.channel}/"
PATH_FOLDER = runs_name + f"/checkpoint/"
# PATH_FOLDER = 'runs/N2UQ/AWGN_N2UQTransformer/pr0.5_4bits_trace_SQ_QAT30ep_lr0.001_Test2ep/checkpoint'
log_dir = runs_name + f"/logs/test{args.channel}/"
makedirs(summary_dir)
makedirs(log_dir)
if not os.path.exists(PATH_FOLDER):
    os.makedirs(PATH_FOLDER)

path_model = 'C:\\Users\\dengjiewen\\PycharmProjects\\LDSC_github\\N2UQ\\N2UQ_transformer.py'
path_main = os.path.abspath(__file__)
path_utils = 'C:\\Users\\dengjiewen\\PycharmProjects\\LDSC_github\\N2UQ\\utils.py'
logger = get_logger(f'log{running_time}.log_time', logpath=log_dir,
                    filepath=path_main, package_files=[path_model, path_utils], displaying=True, saving=True)
writer = SummaryWriter(summary_dir)
logger.info(f"Experiment Name: {args.exp_name}")
logger.info('='*120)
logger.info('Starting N2UQ Training with following configuration:')
logger.info(f'Channel: {args.channel}')
logger.info(f'Network: {args.network}')
logger.info(f'Quantization: {args.quant}')
logger.info(f'Bits: {args.bits}')
logger.info(f'Pruning Ratio: {args.ratio}')
logger.info(f'Hessian Mode: {args.hessian_mode}')
logger.info(f'Batch Size (Train): {args.batch_size_train}')
logger.info(f'Batch Size (Test): {args.batch_size_test}')
logger.info(f'QAT Epochs: {args.QAT_epochs}')
logger.info(f'Learning Rate: {args.learning_rate}')
logger.info(f'SNR Values: {SNR}')
logger.info('='*120)

bleu_score_1gram = BleuScore(1, 0, 0, 0)
model = SentenceTransformer(args.model_name, device=device)

vocab = json.load(open(args.vocab_file, 'rb'))
token_to_idx = vocab['token_to_idx']
num_vocab = len(token_to_idx)
pad_idx = token_to_idx["<PAD>"]
start_idx = token_to_idx["<START>"]
end_idx = token_to_idx["<END>"]
vocb_dictionary = token_to_idx.items()
StoT = SeqtoText(vocb_dictionary, end_idx)
src_pad_idx = token_to_idx["<PAD>"]
""" define optimizer and loss function """
PATH_LOAD = PATH_FOLDER
PATH_LOAD = f'runs/N2UQ/AWGN_N2UQTransformer/pr0.5_4bits_trace_SQ_QAT30ep_lr0.001_Test2ep/checkpoint'
transformer = N2UQTransformer(args.num_layers, num_vocab, num_vocab,
                           num_vocab, num_vocab, args.d_model, args.num_heads,
                           args.dff, args.dropout_rate, a_bits = args.bits, w_bits = args.bits).to(device)
total_params_origin = sum(p.numel() for p in transformer.parameters())
trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
logger.info(f"Total parameters (original model): {total_params_origin:,}")
logger.info(f"Trainable parameters (original model): {trainable_params:,}")

criterion = nn.CrossEntropyLoss(reduction='none')
"""################################# stastic quantization ###############################
# this module is used for quantizing the HAPed model"""
if args.stastic_quantization:
    logger.info('-'*120)
    logger.info('Starting Static Quantization Process')
    if args.ratio == '0':
        args.checkpoint_path = f"./runs/pretrain/{args.channel}_{args.quant}/{args.bits}bits_ep200_lr0.001/checkpoint/"
        pretrained_dict = load_model_state_dict(args.checkpoint_path)
        out_indices = range(22234)
        logger.info(f'Loading pretrained model from: {args.checkpoint_path}')
    else:
        import glob

        ckpt_base_folder = f"./runs/pruning/{args.channel}_{args.quant}"
        pattern = os.path.join(ckpt_base_folder, f"pr{args.ratio}_{args.bits}bits_{args.hessian_mode}*", "checkpoint")

        matching_folders = glob.glob(pattern)

        if not matching_folders:
            raise FileNotFoundError(f"No checkpoint folder found matching pattern: {pattern}")

        if len(matching_folders) > 1:
            logger.warning(f"Found multiple matching folders: {matching_folders}")
            logger.warning(f"Using the first one: {matching_folders[0]}")

        args.checkpoint_path = matching_folders[0]
        logger.info(f'Searching for HAP pruned model in: {ckpt_base_folder}')
        logger.info(f'Pattern: pr{args.ratio}_{args.bits}bits_{args.hessian_mode}*/checkpoint/')
        logger.info(f'Found checkpoint path: {args.checkpoint_path}')
        transformer_HAP, checkpoint, epoch_origin = load_network(args.checkpoint_path)
        pretrained_dict = checkpoint['net'].state_dict()
        out_indices = checkpoint['net'].dense.out_indices
        logger.info(f'Loading HAP pruned model from: {args.checkpoint_path}')
        logger.info(f'Original pruning epoch: {epoch_origin}')
    # pretrained_dict = checkpoint['state_dict']
    transformer_dict = transformer.state_dict()

    new_dict = {k: v for k, v in pretrained_dict.items() if k in transformer_dict.keys()}
    # create new model_state_dict()
    transformer_dict.update(new_dict)
    if args.ratio != '0':
        build_quantized_model(transformer, new_dict)
    transformer.load_state_dict(transformer_dict)
    logger.info(transformer)
    total_params = sum(p.numel() for p in transformer.parameters())
    prune_ratio = (1 - total_params / total_params_origin) * 100
    logger.info(f"Total parameters after quantization: {total_params:,}")
    logger.info(f"Pruning ratio: {prune_ratio:.2f}%")

    # update constellation quantizer parameters
    logger.info('Updating constellation quantizer parameters...')
    for sents in tqdm(test_iterator):
        sents = sents.to(device)
        update_quant(transformer, sents, sents, pad_idx)

    # calibration - collect statistics for ALL quantizers in the model
    logger.info('Starting full calibration process for all quantizers...')
    transformer.train()
    calib_count = 0
    with torch.no_grad():
        for i, src in enumerate(tqdm(test_iterator)):
            src = src.to(device)
            trg = src

            calibrate_all_quantizers(transformer, src, trg, pad_idx)

            calib_count += 1
            if calib_count >= 5:
                break

    logger.info(f'Calibration finished! Processed {calib_count} batches.')
    # Freeze all calibrations - mark that calibration is done
    logger.info('Freezing all quantizer statistics...')
    for name, module in transformer.named_modules():
        if hasattr(module, 'freeze_calibration'):
            module.freeze_calibration()
    logger.info('All quantizer statistics collected and frozen.')

    # Test calibration quality
    sum_test_loss = 0
    avg_test_loss = 0
    for i, src in enumerate(test_iterator):
        transformer.eval()
        src = src.to(device)
        trg = src
        test_loss = test_step(transformer, src, trg, 0.1,
                              pad_idx, criterion, args.channel, out_indices)
        writer.add_scalar('Calibration/test_loss', test_loss, i)
        if i > 5:
            avg_test_loss = sum_test_loss / i
            break
        else:
            sum_test_loss += test_loss
            logger.info(f'Calibration step {i}: test loss = {test_loss:.6f}')
    save_model(args, transformer, avg_test_loss, PATH_FOLDER, 0, out_indices)
    logger.info(f'Calibration finished! Final average test loss: {avg_test_loss:.6f}')
    logger.info(f'Model saved to: {PATH_FOLDER}')
    writer.add_scalar('Calibration/final_avg_loss', avg_test_loss, 0)

"""################################# quantization-aware training ###############################
# this module is used for recover the precision of system
"""
if args.QAT:
    logger.info('-'*120)
    logger.info('Starting Quantization-Aware Training (QAT)')
    # load pre-trained model
    transformer, checkpoint, epoch_origin = load_network(PATH_FOLDER)
    out_indices = checkpoint['out_indices']
    epoch_range = [epoch_i for epoch_i in range((epoch_origin + 1), epoch_origin + args.QAT_epochs + 1)]
    total_params = sum(p.numel() for p in transformer.parameters())
    logger.info(f"Total parameters (QAT model): {total_params:,}")
    logger.info(f"Pruning ratio: {prune_ratio:.2f}%")
    logger.info(f'QAT will run for epochs: {epoch_range[0]} to {epoch_range[-1]}')

    all_parameters = transformer.parameters()
    alpha_parameters = []
    for name, m in transformer.named_parameters():
        if 'quantizer.a' in name or 'start' in name:
            # print('alpha_param:', pname)
            alpha_parameters.append(m)
    alpha_parameters_id = list(map(id, alpha_parameters))
    other_parameters = list(filter(lambda p: id(p) not in alpha_parameters_id, all_parameters))
    lr = args.learning_rate
    optimizer = torch.optim.Adam(
            [{'params' : alpha_parameters, 'lr': lr/10, 'name': 'quantization_params'},
            {'params' : other_parameters, 'weight_decay':0.0005, 'lr': lr, 'name': 'other_params'}],
            betas=(0.9,0.999), eps=1e-9)
    adjustlr_epoch = [int(epoch_range[0] + args.QAT_epochs * 0.5), int(epoch_range[0] + args.QAT_epochs * 0.75)]
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params:,}")
    for i, param_group in enumerate(optimizer.param_groups):
        group_name = param_group.get('name', f'group_{i}')
        cur_lr = param_group['lr']
        if len(param_group) > 0:
            logger.info(f'Learning rate {i} ({group_name}): {cur_lr}')
    record_acc = 6
    train_loss_record = []
    validate_loss_record = []
    best_test_loss = float('inf')
    best_epoch = 0
    for epoch in epoch_range:# defalut = 10
        start = time.time()
        total = 0
        test_total = 0
        for pname, p in transformer.named_parameters():
            if 'quant_constellation.a' in pname:
                logger.info(f'Quantizer parameter {pname}:\n{p}')
                break
        ##########################################################################
        if args.channel == 'AWGN':
            noise_std = np.random.uniform(SNR_to_noise(0), SNR_to_noise(10), size=(1))[0]
            # noise_std = np.random.uniform(SNR_to_noise(-2), SNR_to_noise(18), size=(1))[0]
        else:
            noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))[0]
        transformer.train()
        logger.info(f'\nEpoch {epoch}/{epoch_range[-1]} - Training (noise_std: {noise_std:.4f})')
        pbar = tqdm(train_iterator)
        for i, src in enumerate(pbar):
            src = src.to(device)
            trg = src
            loss = train_step(transformer, src, trg,
                              noise_std, pad_idx, optimizer, criterion, args.channel,
                              out_indices)
            total += loss
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}'.format(
                    epoch, loss
                )
            )
            writer.add_scalar('QAT/train_batch_loss', loss, (epoch - epoch_range[0]) * len(train_iterator) + i)

        avg_train_loss = total / len(train_iterator)
        train_loss_record.append(avg_train_loss)
        writer.add_scalar('QAT/train_epoch_loss', avg_train_loss, epoch)
        logger.info(f'Epoch {epoch} - Average training loss: {avg_train_loss:.6f}')
        ##########################################################################
        transformer.eval()
        logger.info(f'Epoch {epoch} - Validation')
        pbar = tqdm(test_iterator)
        for i, src in enumerate(pbar):
            src = src.to(device)
            trg = src
            test_loss = test_step(transformer, src, trg,0.1,
                                  pad_idx, criterion, args.channel, out_indices)
            pbar.set_description(
                'Epoch: {}; Type: VAL; Loss: {:.5f}'.format(
                    epoch + 1, test_loss
                )
            )
            test_total += test_loss
        avg_test_loss = test_total/len(test_iterator)
        validate_loss_record.append(avg_test_loss)
        writer.add_scalar('QAT/test_epoch_loss', avg_test_loss, epoch)
        logger.info(f'Epoch {epoch} - Average test loss: {avg_test_loss:.6f}')

        if avg_test_loss < record_acc:
            save_model(args, transformer, avg_test_loss, PATH_FOLDER, epoch, out_indices)
            record_acc = avg_test_loss
            logger.info(f'Epoch {epoch} - New best model saved! Test loss: {avg_test_loss:.6f}')

            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                best_epoch = epoch

            if epoch in adjustlr_epoch:
                # if epoch % round(args.QAT_epochs/3) == 0:
                logger.info(f'Epoch {epoch} - Adjusting learning rate')
                load_model(transformer, PATH_FOLDER)
                # load_model(transformer, PATH_FOLDER)
                lr = adjust_learning_rate(optimizer, lr)
                for i, param_group in enumerate(optimizer.param_groups):
                    group_name = param_group.get('name', f'group_{i}')
                    cur_lr = param_group['lr']
                    if len(param_group) > 0:
                        logger.info(f'Adjusted learning rate {i} ({group_name}): {cur_lr}')
            ##########################################################################
            elapsed = time.time() - start
            logger.info(f'Epoch {epoch} completed in {elapsed:.2f} seconds')
            logger.info(f'Current best test loss: {best_test_loss:.6f} (epoch {best_epoch})')

        logger.info('-' * 120)
        logger.info(f'QAT Finished! Minimum Test Loss: {record_acc:.6f} at epoch {best_epoch}')
        np.save(PATH_FOLDER + '/train_loss_record.npy', train_loss_record)
        np.save(PATH_FOLDER + '/validate_loss_record.npy', validate_loss_record)
        writer.add_scalar('QAT/best_test_loss', best_test_loss, best_epoch)
        writer.add_hparams({
            'channel': args.channel,
            'network': args.network,
            'bits': args.bits,
            'ratio': args.ratio,
            'hessian_mode': args.hessian_mode,
            'QAT_epochs': args.QAT_epochs,
            'learning_rate': args.learning_rate,
            'batch_size_train': args.batch_size_train,
            'batch_size_test': args.batch_size_test
        }, {
            'best_test_loss': best_test_loss,
            'final_train_loss': train_loss_record[-1],
            'final_test_loss': validate_loss_record[-1]
        })

if args.Test:
    logger.info('-' * 120)
    logger.info('Starting Testing Phase')
    # load_model(transformer, PATH_FOLDER)
    transformer, checkpoint, epoch_origin = load_network(PATH_LOAD)
    out_indices = checkpoint['out_indices']

    total_params = sum(p.numel() for p in transformer.parameters())
    prune_ratio = (1 - total_params / total_params_origin) * 100
    logger.info(f"Total parameters (Test model): {total_params:,}")
    logger.info(f"Pruning ratio: {prune_ratio:.2f}%")
    args.valid_path = os.path.join(PATH_FOLDER, f'valid')
    if not os.path.exists(args.valid_path):
        os.makedirs(args.valid_path)
    StoT = SeqtoText(vocb_dictionary, end_idx)
    score = []
    score1 = []
    for epoch in range(args.Test_epochs):  # 10
        logger.info(f'\nTest Epoch {epoch + 1}/{args.Test_epochs}')
        final_word = []
        original_word = []

        for snr in SNR:
            logger.info(f'Testing at SNR: {snr} dB')
            word = []
            target_word = []
            noise_std = SNR_to_noise(snr)
            pbar = tqdm(test_iterator)
            for src in pbar:
                # for i, src in enumerate(test_iterator):
                transformer.eval()
                src = src.to(device)
                target = src

                out = greedy_decode(transformer, src, noise_std, args.MAX_LENGTH, src_pad_idx,
                                    start_idx, args.channel, out_indices)

                sentences = out.cpu().numpy().tolist()
                result_string = list(map(StoT.sequence_to_text, sentences))
                word = word + result_string

                target_sent = target.cpu().numpy().tolist()
                result_string = list(map(StoT.sequence_to_text, target_sent))
                target_word = target_word + result_string
                pbar.set_description_str(result_string[0])

            final_word.append(word)
            original_word.append(target_word)
            # valid_path = os.path.join(args.valid_path,'trans{}_{}.txt'.format(snr,epoch))
            with open(os.path.join(args.valid_path, 'trans{}_{}.txt'.format(snr, epoch)), 'w') as f:
                for line in word:
                    f.write('%s\n' % line)
            logger.info(f'SNR {snr} dB - Generated {len(word)} sequences')

        bleu_score = []
        bertsim_score = []
        for sent1, sent2 in zip(original_word, final_word):
            # 1-gram
            bleu_score.append(bleu_score_1gram.compute_blue_score(sent1, sent2))
            # bertsim
            sim_score = []
            for i in range(len(sent1)):
                sentences = [sent1[i], sent2[i]]
                embeddings = model.encode(sentences)
                predictions = util.cos_sim(embeddings[0], embeddings[1])
                sim_score.append(predictions[0][0])
            sim_score_mean = sum(sim_score) / len(sim_score)
            bertsim_score.append(sim_score_mean)

        bleu_score = np.array(bleu_score)
        bleu_score = np.mean(bleu_score, axis=1)
        score.append(bleu_score)
        score1.append(bertsim_score)
        # Log BLEU scores for this epoch
        for i, snr in enumerate(SNR):
            writer.add_scalar('Test/bleu_score_per_snr', bleu_score[i], epoch * len(SNR) + snr)
            logger.info(f'Test Epoch {epoch} - SNR {snr} dB - BLEU Score: {bleu_score[i]:.5f}')

    score = np.mean(np.array(score), axis=0)
    bleu_score_str = '[{}]'.format(','.join('{:.5f}'.format(x) for x in score))
    logger.info(f'Average BLEU Scores across {args.Test_epochs} epochs: {bleu_score_str}')
    score1 = np.mean(np.array(score1), axis = 0)
    bertsim_score_str = '[{}]'.format(','.join('{:.5f}'.format(x) for x in score1))
    logger.info(f'Average BLEU Scores across {args.Test_epochs} epochs: {bertsim_score_str}')
    # Log average BLEU scores
    for i, snr in enumerate(SNR):
        writer.add_scalar('Test/avg_bleu_score', score[i], snr)
        logger.info(f'Final Average - SNR {snr} dB - BLEU Score: {score[i]:.5f}')
        writer.add_scalar('Test/avg_bertsim_score', score[i], snr)
        logger.info(f'Final Average - SNR {snr} dB - BERT Similarity Score: {score[i]:.5f}')

    save_results_to_file_2metric(args, SNR, score, score1)
    logger.info(f'Results saved to: {args.valid_path}')
    writer.close()
    logger.info('TensorBoard writer closed.')
