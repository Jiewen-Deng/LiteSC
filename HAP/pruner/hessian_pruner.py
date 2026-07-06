"""
@Author: JW Deng
@Time: 2026/7/6 17:00
"""
import torch
import torch.nn as nn
import numpy as np
import re
import pickle
import os.path
from collections import OrderedDict
from tqdm import tqdm

from HAP.utils.common_utils import *
from HAP.utils.prune_utils import (filter_indices, filter_indices_qkv, filter_indices_num, filter_indices_ni,
                                   get_threshold, update_indices, get_layer_dependencies_out, prune_model_ni)
from HAP.utils.network_utils import stablize_bn
from pretrain_utils import SNR_to_noise
from N2UQ.transformer_constellation import Transformer
from .hessian_fact import get_trace_hut


def save_model_weight(save_item, folder_path, epoch, max_files=2):
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
    torch.save(save_item, model_path)

    print(f"Model saved to {model_path}")

class HessianPruner:

        def __init__(self,
                     model,
                     builder,
                     config,
                     writer,
                     logger,
                     prune_ratio_limit,
                     network,
                     batch_averaged=True,
                     use_patch=False,
                     fix_layers=0,
                     hessian_mode='Trace',
                     use_decompose=False):
            print('Using patch is %s' % use_patch)
            self.iter = 0
            self.logger = logger
            self.writer = writer
            self.config = config
            self.prune_ratio_limit = prune_ratio_limit
            self.network = network

            self.batch_averaged = batch_averaged
            self.use_decompose = use_decompose
            self.known_modules = {'Linear', 'Conv2d', 'Embedding', 'LayerNorm', 'QuantizedLinear', 'QuantizedLinear_cons'}#
            if self.use_decompose:
                self.known_modules = {'Conv2d'}
            self.modules = []
            self.module_names = []
            self.model = model
            self.builder = builder
            self.fix_layers = fix_layers
            self.steps = 0
            self.use_patch = False  # use_patch

            self.W_pruned = {}
            self.S_l = None

            self.hessian_mode = hessian_mode

            self.importances = {}
            self._inversed = False
            self._cfgs = {}
            self._indices = {}

        def make_pruned_model(self, dataloader,pad_idx, criterion, device, fisher_type, prune_ratio, is_loader=False, normalize=True, re_init=False, n_v=300):
            self.prune_ratio = prune_ratio # use for some special case, particularly slq_full, slq_layer
            self._prepare_model()
            self.init_step()
            if self.config.dataset == 'imagenet' or self.config.dataset == 'text':
                is_loader = True
            self._compute_hessian_importance(dataloader,pad_idx, criterion, device, is_loader, n_v=n_v)

            if self.use_decompose:
                self._do_prune_ni(prune_ratio, self.config.ni_ratio ,re_init)
                self._build_pruned_model_ni(re_init)
            else:
                self._do_prune(prune_ratio, re_init)
                self._build_pruned_model(re_init)

            self._rm_hooks()
            self._clear_buffer()
            return str(self.model)

        def _prepare_model(self):
            count = 0
            for name, module in self.model.named_modules():
                classname = module.__class__.__name__
                if classname in self.known_modules:
                    self.modules.append(module)
                    self.module_names.append(name)
                    count += 1
            self.modules = self.modules[self.fix_layers:]
            self.module_names = self.module_names[self.fix_layers:]


        def _compute_hessian_importance(self, dataloader,pad_idx, criterion, device, is_loader, n_v=300):
            ###############
            # Here, we use the fact that Conv does not have bias term
            ###############
            if self.hessian_mode == 'trace':
                for m in self.model.parameters():
                    # set requires_grad for convolution layers only
                    shape_list = [2, 4]
                    if self.use_decompose:
                        shape_list = [4]
                    if len(m.shape) in shape_list:
                        m.requires_grad = True
                    else:
                        m.requires_grad = False
                if self.config.network=='Transformer':
                    trace_dir = f"./runs/pruning/{self.config.channel}_{self.config.quant}/tract.pkl"
                else:
                    trace_dir = f"./runs/pruning/{self.config.channel}_{self.config.network}/tract.pkl"
                # trace_dir = f"./HAP/out/{self.config.dataset}_result/{self.config.network}{self.config.depth}/tract.pkl"
                print(trace_dir)
                if os.path.exists(trace_dir):
                    print(f"Loading trace from {trace_dir}")
                    with open(trace_dir, 'rb') as f:
                        results = pickle.load(f)
                    # results = np.load(trace_dir, allow_pickle=True)
                else:
                    results = get_trace_hut(self.model, dataloader, pad_idx, criterion, device, n_v=n_v, loader=is_loader, channel=self.config.channel, channelwise=True, layerwise=False)
                    with open(trace_dir, 'wb') as f:
                        pickle.dump(results, f)
                    # np.save(trace_dir, results)


                for name, m in self.model.named_parameters():
                    if 'two' in name or 'one' in name or 'zero' in name or 'minusone' in name or 'eps' in name:
                        m.requires_grad = False
                    m.requires_grad = True

                channel_trace, weighted_trace = [], []
                for k, layer in enumerate(results):
                    channel_trace.append(torch.zeros(len(layer)))
                    weighted_trace.append(torch.zeros(len(layer)))
                    for cnt, channel in enumerate(layer):
                        channel_trace[k][cnt] = sum(channel) / len(channel)

                k=0
                for m in self.modules:
                    tmp = []
                    if isinstance(m, nn.LayerNorm):
                        continue
                    else:
                        for cnt, channel in enumerate(m.weight.data):
                            tmp.append(abs((channel_trace[k][cnt] * channel.detach().norm()**2 / channel.numel())).cpu().item())# absolute value
                            # tmp.append((channel_trace[k][cnt] * channel.detach().norm()**2 / channel.numel()).cpu().item())
                        self.importances[m] = (tmp, len(tmp))
                        self.W_pruned[m] = fetch_mat_weights(m, False)
                        k += 1

            elif self.hessian_mode == 'random':
                # get uniform baseline
                for k, m in enumerate(self.modules):
                    tmp = []
                    if isinstance(m, nn.LayerNorm):
                        continue
                    else:
                        for cnt, channel in enumerate(m.weight.data):
                            tmp.append( np.random.randn() )
                        self.importances[m] = (tmp, len(tmp))
                        self.W_pruned[m] = fetch_mat_weights(m, False)
            elif self.hessian_mode == 'magnitude':
                # get magnitude-based importance scores
                for k, m in enumerate(self.modules):
                    tmp = []
                    if isinstance(m, nn.LayerNorm):
                        continue
                    else:
                        for cnt, channel in enumerate(m.weight.data):
                            # Calculate the average absolute value of each channel's weight as an importance score.
                            channel_magnitude = torch.mean(torch.abs(channel)).item()
                            tmp.append(channel_magnitude)
                        self.importances[m] = (tmp, len(tmp))
                        self.W_pruned[m] = fetch_mat_weights(m, False)

        def _do_prune(self, prune_ratio, re_init):
            # get threshold
            all_importances = []
            for idx, m in enumerate(self.modules):
                module_name = self.module_names[idx]
                if (('dense' in module_name and 'dense' != module_name) or 'w_2' in module_name or isinstance(m, nn.LayerNorm)
                        or isinstance(m, nn.Embedding) or 'channel_encoder.2' in module_name):
                    continue
                else:
                    imp_m = self.importances[m]
                    imps = imp_m[0]
                    all_importances += imps
            all_importances = sorted(all_importances)
            idx = int(prune_ratio * len(all_importances))
            threshold = all_importances[idx]

            threshold_recompute = get_threshold(all_importances, prune_ratio)
            idx_recomputed = len(filter_indices(all_importances, threshold))
            print('=> The threshold is: %.5f (%d), computed by function is: %.5f (%d).' %
                (threshold, idx, threshold_recompute, idx_recomputed))

            # do pruning
            print('=> Conducting network pruning. Max: %.5f, Min: %.5f, Threshold: %.5f' %
                (max(all_importances), min(all_importances), threshold))
            self.logger.info("[Weight Importances] Max: %.5f, Min: %.5f, Threshold: %.5f." %
                (max(all_importances), min(all_importances), threshold))

            for idx, m in enumerate(self.modules):
                module_name = self.module_names[idx]
                m.in_indices = None
                if isinstance(m, nn.LayerNorm):
                    continue

                # Special modules are not pruned.
                if (('dense' in module_name and 'dense' != module_name) or 'w_2' in module_name
                        or 'channel_encoder.2' in module_name or isinstance(m, nn.Embedding)):
                    row_indices = list(range(self.W_pruned[m].size(0)))
                else:
                    imp_m = self.importances[m]
                    n_r = imp_m[1]
                    row_imps = imp_m[0]
                    if 'wq' in module_name or 'wk' in module_name or 'wv' in module_name:
                        row_indices = filter_indices_qkv(row_imps, threshold)
                    else:
                        row_indices = filter_indices(row_imps, threshold)
                    r_ratio = 1 - len(row_indices) / n_r
                    # compute row indices (out neurons)
                    if r_ratio > self.prune_ratio_limit:
                        r_threshold = get_threshold(row_imps, self.prune_ratio_limit)
                        row_indices = filter_indices(row_imps, r_threshold)
                        print('* row indices empty!')
                m.out_indices = row_indices
            # Layers with corresponding relationships in the output dimension
            dependencies_out = get_layer_dependencies_out(self.model, self.network)
            for idx, m in enumerate(self.modules):
                module_name = self.module_names[idx]
                '''
                dependent_module = None
                if 'src_mha.wq' in module_name:
                    dependent_module = self.modules[idx + 1]
                # Dimensional Correspondence in Residual Connections
                elif 'mha.dense' in module_name and 'enc_layers' in module_name:
                    dependent_module = self.modules[idx + 2]
                elif 'self_mha.dense' in module_name:
                    dependent_module = self.modules[idx + 6]
                elif 'src_mha.dense' in module_name:
                    dependent_module = self.modules[idx + 2]
                elif 'channel_decoder.linear1' in module_name:
                    dependent_module = self.modules[idx + 2]
                if dependent_module is None:
                    continue
                else:
                    imp_m = self.importances[m]
                    row_imps = imp_m[0]
                    m.out_indices = filter_indices_num(row_imps, len(dependent_module.out_indices))
                '''
                if m in dependencies_out.keys():
                    deps = dependencies_out[m]
                    num_indices = len(deps.out_indices)
                    if len(m.out_indices) != num_indices:
                        imp_m = self.importances[m]
                        row_imps = imp_m[0]
                        m.out_indices = filter_indices_num(row_imps, num_indices)
            update_indices(self.model, self.network)

        def _build_pruned_model(self, re_init):
            for m_name, m in self.model.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    idxs = m.in_indices
                    m.num_features = len(idxs)
                    m.weight.data = m.weight.data[idxs]
                    m.bias.data = m.bias.data[idxs].clone()
                    m.running_mean = m.running_mean[idxs].clone()
                    m.running_var = m.running_var[idxs].clone()
                    m.weight.grad = None
                    m.bias.grad = None
                elif isinstance(m, nn.Conv2d):
                    in_indices = m.in_indices
                    if m.in_indices is None:
                        in_indices = list(range(m.weight.size(1)))
                    m.weight.data = m.weight.data[m.out_indices, :, :, :][:, in_indices, :, :].clone()
                    if m.bias is not None:
                        m.bias.data = m.bias.data[m.out_indices]
                        m.bias.grad = None
                    m.in_channels = len(in_indices)
                    m.out_channels = len(m.out_indices)
                    m.weight.grad = None

                elif isinstance(m, nn.Linear):
                    in_indices = m.in_indices
                    if m.in_indices is None:
                        in_indices = list(range(m.weight.size(1)))

                    m.weight.data = m.weight.data[m.out_indices, :][:, in_indices].clone()

                    if m.bias is not None:
                        m.bias.data = m.bias.data[m.out_indices].clone()
                        m.bias.grad = None

                    m.in_features = len(in_indices)
                    m.out_features = len(m.out_indices)
                    m.weight.grad = None
                elif isinstance(m, nn.Embedding):
                    # Embedding parameter shape is [num_embeddings, embedding_dim]
                    # out_indices equals num_embeddings, in_indices equals embedding_dim
                    in_indices = m.in_indices
                    if m.in_indices is None:
                        # If in_indices is not specified, retain all embedding dimensions
                        in_indices = list(range(m.weight.size(1)))

                    # For Embedding layer, we only prune based on out_indices (vocabulary dimension)
                    m.weight.data = m.weight.data[m.out_indices, :][:, in_indices].clone()

                    # Update parameters of the Embedding layer
                    m.num_embeddings = len(m.out_indices)
                    m.embedding_dim = len(in_indices)
                    m.weight.grad = None
                elif isinstance(m, nn.LayerNorm):
                    in_indices = m.in_indices
                    if m.in_indices is None:
                        in_indices = list(range(m.weight.size(0)))

                    # Reconstruct LayerNorm parameters based on in_indices
                    m.weight.data = m.weight.data[in_indices].clone()
                    m.bias.data = m.bias.data[in_indices].clone()

                    # Update feature dimension of LayerNorm
                    m.normalized_shape = (len(in_indices),)
                    m.weight.grad = None
                    m.bias.grad = None


        def _do_prune_ni(self, prune_ratio, ni_ratio, re_init):
            # get threshold
            all_importances = []
            for m in self.modules:
                imp_m = self.importances[m]
                imps = imp_m[0]
                all_importances += imps
            all_importances = sorted(all_importances)
            idx = int(prune_ratio * len(all_importances))
            ni_idx = int( (1-ni_ratio) *prune_ratio * len(all_importances))
            threshold = all_importances[idx]
            ni_threshold  = all_importances[ni_idx]

            # do pruning
            print('=> Conducting network pruning. Max: %.5f, Min: %.5f, Threshold: %.5f' %
                (max(all_importances),  min(all_importances), threshold))
            self.logger.info("[Weight Importances] Max: %.5f, Min: %.5f, Threshold: %.5f." %
                (max(all_importances), min(all_importances), threshold))

            for idx, m in enumerate(self.modules):
                imp_m = self.importances[m]
                n_r = imp_m[1]
                row_imps = imp_m[0]
                remained_indices, ni_indices, pruned_indices = filter_indices_ni(row_imps, threshold, ni_threshold)
                r_ratio = (len(remained_indices) + len(ni_indices)) / n_r

                # compute row indices (out neurons)
                if r_ratio > self.prune_ratio_limit:
                    row_imps = sorted(row_imps)
                    idx = int(self.prune_ratio_limit * len(row_imps))
                    ni_idx = int( (1-ni_ratio) *prune_ratio * len(row_imps))
                    tmp_threshold = row_imps[idx]
                    tmp_ni_threshold  = row_imps[ni_idx]
                    remained_indices, ni_indices, pruned_indices = filter_indices_ni(row_imps, tmp_threshold, tmp_ni_threshold)
                    print('* row indices empty!')
                # Special handling for the last Linear layer and Embedding layer
                if isinstance(m, nn.Linear) and idx == len(self.modules) - 1:
                    row_indices = list(range(self.W_pruned[m].size(0)))
                elif isinstance(m, nn.Embedding):
                    # For Embedding layer, limit pruning ratio
                    embedding_prune_limit = min(self.prune_ratio_limit, 0.2)  # Limit maximum pruning to 20% for Embedding layer
                    if r_ratio > embedding_prune_limit:
                        row_imps = sorted(row_imps)
                        idx = int(embedding_prune_limit * len(row_imps))
                        ni_idx = int( (1-ni_ratio) * embedding_prune_limit * len(row_imps))
                        tmp_threshold = row_imps[idx]
                        tmp_ni_threshold  = row_imps[ni_idx]
                        remained_indices, ni_indices, pruned_indices = filter_indices_ni(row_imps, tmp_threshold, tmp_ni_threshold)
                        print('* embedding row indices adjusted!')

                m.remained_indices = remained_indices
                m.ni_indices       = ni_indices
                m.pruned_indices   = pruned_indices

                m.out_indices = sorted(m.remained_indices + m.ni_indices)
                m.in_indices = None
            update_indices(self.model, self.network)

        def _build_pruned_model_ni(self, re_init):
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    idxs = m.in_indices
                    # print(len(idxs))
                    m.num_features = len(idxs)
                    m.weight.data = m.weight.data[idxs]
                    m.bias.data = m.bias.data[idxs].clone()
                    m.running_mean = m.running_mean[idxs].clone()
                    m.running_var = m.running_var[idxs].clone()
                    m.weight.grad = None
                    m.bias.grad = None

                elif isinstance(m, nn.Linear):
                    in_indices = m.in_indices
                    if m.in_indices is None:
                        in_indices = list(range(m.weight.size(1)))
                    m.weight.data = m.weight.data[:, in_indices].clone()

                    if m.bias is not None:
                        m.bias.data = m.bias.data.clone()
                        m.bias.grad = None

                    m.in_features = len(in_indices)
                    m.weight.grad = None
                elif isinstance(m, nn.Embedding):
                    in_indices = m.in_indices
                    if m.in_indices is None:
                        in_indices = list(range(m.weight.size(1)))
                    
                    m.weight.data = m.weight.data[m.remained_indices + m.ni_indices, :][:, in_indices].clone()
                    m.num_embeddings = len(m.remained_indices + m.ni_indices)
                    m.embedding_dim = len(in_indices)
                    m.weight.grad = None

            self.model = prune_model_ni(self.model.module)
            # if re_init:
            #     self.model.apply(_weights_init)

        def init_step(self):
            self.steps = 0

        def step(self):
            self.steps += 1

        def _rm_hooks(self):
            for m in self.model.modules():
                classname = m.__class__.__name__
                if classname in self.known_modules:
                    m._backward_hooks = OrderedDict()
                    m._forward_pre_hooks = OrderedDict()

        def _clear_buffer(self):
            self.m_aa = {}
            self.m_gg = {}
            self.d_a = {}
            self.d_g = {}
            self.Q_a = {}
            self.Q_g = {}
            self.modules = []
            if self.S_l is not None:
                self.S_l = {}

        def fine_tune_model(self, trainloader, testloader, criterion, learning_rate, weight_decay, channel='AWGN', nepochs=10,
                            device='cuda'):
            self.model = self.model.train()
            self.model = self.model.cpu()
            self.model = self.model.to(device)

            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
            # optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)# , weight_decay=weight_decay
            # optimizer = optim.Adam(self.model.parameters(), weight_decay=5e-4)
            if self.config.dataset == "text":
                lr_schedule = {0: learning_rate,
                            int(nepochs * 0.5): learning_rate * 0.1,
                            int(nepochs * 0.75): learning_rate * 0.01}

            else:
            # elif self.config.dataset == "imagenet":
                lr_schedule = {0 : learning_rate,
                   10: learning_rate * 0.1,
                   20: learning_rate * 0.01}


            lr_scheduler = PresetLRScheduler(lr_schedule)
            best_test_acc, best_test_loss = 0, 100
            iterations = 0

            for epoch in range(nepochs):
                self.model = self.model.train()
                correct = 0
                total = 0
                all_loss = 0
                if channel == 'AWGN':
                    noise_std = np.random.uniform(SNR_to_noise(0), SNR_to_noise(18), size=(1))[0]
                    # noise_std = np.random.uniform(SNR_to_noise(-2), SNR_to_noise(18), size=(1))[0]
                else:
                    noise_std = np.random.uniform(SNR_to_noise(-4), SNR_to_noise(10), size=(1))[0]
                    # noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))[0]
                lr_scheduler(optimizer, epoch)
                desc = ('[LR: %.5f] Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                lr_scheduler.get_lr(optimizer), 0, 0, correct, total))
                prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
                for batch_idx, inputs in prog_bar:
                    optimizer.zero_grad()
                    inputs = inputs.to(device)
                    outputs, loss = self.model(inputs, noise_std, 0, criterion, channel, device)
                    self.writer.add_scalar('train_%d/loss' % self.iter, loss.item(), iterations)
                    iterations += 1
                    all_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    _, predicted = outputs.max(2)
                    targets = inputs[:, 1:]
                    total += targets.size(0) * targets.size(1)
                    correct += predicted.eq(targets).sum().item()
                    desc = ('[%d][LR: %.5f] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                            (epoch, lr_scheduler.get_lr(optimizer), all_loss / (batch_idx + 1),
                             100. * correct / total, correct, total))# , WD: %.5f； , weight_decay
                    prog_bar.set_description(desc, refresh=True)

                test_loss, test_acc, top5_acc = self.test_model(testloader, criterion, channel, device)
                self.writer.add_scalar('valid_loss', test_loss.item(), epoch)
                self.writer.add_scalar('valid_acc', test_acc.item(), epoch)
                self.writer.add_scalar('valid_acc_top5', top5_acc.item(), epoch)
                self.logger.info(f'{epoch} Test Loss: %.3f, Test Top1 %.2f%%(test), Test Top5 %.2f%%(test).' % (test_loss, test_acc, top5_acc))

                if test_acc > best_test_acc:
                    best_test_loss = test_loss
                    best_test_acc  = test_acc
                    path_folder = self.config.checkpoint
                    # path = os.path.join(self.config.checkpoint, '%s_%s%s.pth.tar' % (dataset, network, depth))
                    save = {
                        'config': self.config,
                        'net': self.model,
                        'acc': test_acc,
                        'loss': test_loss,
                        'epoch': epoch
                    }
                    save_model_weight(save, path_folder,epoch)
                    # torch.save(save, path)
            print('** Finetuning finished. Stabilizing batch norm and test again!')
            stablize_bn(self.model, trainloader, criterion, channel, device)
            test_loss, test_acc, top5_acc = self.test_model(testloader, criterion, channel, device)
            best_test_loss = best_test_loss if best_test_acc > test_acc else test_loss
            best_test_acc = max(test_acc, best_test_acc)
            return best_test_loss, best_test_acc

        def fine_tune_N2UQ(self, trainloader, testloader, criterion, learning_rate, weight_decay, channel='AWGN', nepochs=10,
                            device='cuda'):
            for name, m in self.model.named_parameters():
                if 'two' in name or 'one' in name or 'zero' in name or 'minusone' in name or 'eps' in name:# or 'scale1' in name
                    m.requires_grad = False

            all_parameters = self.model.parameters()
            alpha_parameters = []
            for pname, p in self.model.named_parameters():
                # if 'quantizer.a' in pname or 'start' in pname:
                if 'quantizer.a' in pname or 'quant_constellation.a' in pname or 'start' in pname:
                    # print('alpha_param:', pname)
                    alpha_parameters.append(p)
            alpha_parameters_id = list(map(id, alpha_parameters))
            other_parameters = list(filter(lambda p: id(p) not in alpha_parameters_id, all_parameters))
            optimizer = torch.optim.Adam(
                [{'params': alpha_parameters, 'lr': learning_rate*0.1},
                 {'params': other_parameters, 'lr': learning_rate, 'weight_decay': weight_decay}],#'weight_decay': weight_decay,
                betas=(0.9, 0.999), eps=1e-9)

            # optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
            # optimizer = optim.Adam(self.model.parameters(), weight_decay=5e-4)
            if self.config.dataset == "text":
                lr_schedule = {0: learning_rate,
                            int(nepochs * 0.5): learning_rate * 0.1,
                            int(nepochs * 0.75): learning_rate * 0.01}

            else:
            # elif self.config.dataset == "imagenet":
                lr_schedule = {0 : learning_rate,
                   10: learning_rate * 0.1,
                   20: learning_rate * 0.01}
            lr_scheduler = PresetLRScheduler(lr_schedule)
            best_test_acc, best_test_loss = 0, 100
            iterations = 0

            self.model = self.model.train()
            self.model = self.model.cpu()
            self.model = self.model.to(device)
            for epoch in range(nepochs):
                self.model = self.model.train()
                correct = 0
                total = 0
                all_loss = 0
                for pname, p in self.model.named_parameters():
                    if 'quant_constellation.a' in pname:
                        print(f'{pname}:\n{p}')
                        break
                if channel == 'AWGN':
                    noise_std = np.random.uniform(SNR_to_noise(0), SNR_to_noise(10), size=(1))[0]
                    # noise_std = np.random.uniform(SNR_to_noise(-2), SNR_to_noise(18), size=(1))[0]
                else:
                    noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))[0]
                lr_scheduler.update_lr(optimizer, epoch)
                lr_alpha, lr_other = lr_scheduler.get_lr(optimizer, True)
                desc = ('[LR(alpha): %.5f LR: %.5f] Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                lr_alpha, lr_other, 0, 0, correct, total))
                prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
                for batch_idx, inputs in prog_bar:
                    optimizer.zero_grad()
                    inputs = inputs.to(device)
                    outputs, loss = self.model(inputs, noise_std, 0, criterion, channel, device)
                    self.writer.add_scalar('train_loss', loss.item(), iterations)
                    iterations += 1
                    all_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    _, predicted = outputs.max(2)
                    targets = inputs[:, 1:]
                    total += targets.size(0) * targets.size(1)
                    correct += predicted.eq(targets).sum().item()
                    desc = ('[%d][LR(alpha): %.5f LR: %.5f] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                            (epoch, lr_alpha, lr_other, all_loss / (batch_idx + 1),
                             100. * correct / total, correct, total))
                    prog_bar.set_description(desc, refresh=True)

                test_loss, test_acc, top5_acc = self.test_model(testloader, criterion, channel, device)
                self.writer.add_scalar('valid_loss', test_loss, epoch)
                self.writer.add_scalar('valid_acc', test_acc, epoch)
                self.writer.add_scalar('valid_acc_top5', top5_acc, epoch)
                self.logger.info(f'{epoch} Test Loss: %.3f, Test Top1 %.2f%%(test), Test Top5 %.2f%%(test).' % (test_loss, test_acc, top5_acc))

                if test_acc > best_test_acc:
                    best_test_loss = test_loss
                    best_test_acc  = test_acc
                    network = self.config.network
                    depth   = self.config.depth
                    dataset = self.config.dataset
                    path_folder = self.config.checkpoint
                    # path = os.path.join(self.config.checkpoint, '%s_%s%s.pth.tar' % (dataset, network, depth))
                    save = {
                        'args': self.config,
                        'net': self.model,
                        'acc': test_acc,
                        'loss': test_loss,
                        'epoch': epoch
                    }
                    save_model_weight(save, path_folder,epoch)
                    # torch.save(save, path)
            print('** Finetuning finished. Stabilizing batch norm and test again!')
            stablize_bn(self.model, trainloader, criterion, channel, device)
            test_loss, test_acc, top5_acc = self.test_model(testloader, criterion, channel, device)
            best_test_loss = best_test_loss if best_test_acc > test_acc else test_loss
            best_test_acc = max(test_acc, best_test_acc)
            return best_test_loss, best_test_acc

        def test_model(self, dataloader, criterion, channel='AWGN', device='cuda'):
            self.model = self.model.eval()
            self.model = self.model.cpu()
            self.model = self.model.to(device)
            correct = 0
            top_1_correct = 0
            top_5_correct = 0
            total = 0
            all_loss = 0
            desc = ('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (0, 0, correct, total))
            prog_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=desc, leave=True)
            for batch_idx, inputs in prog_bar:
                inputs = inputs.to(device)
                outputs, loss = self.model(inputs, 0.1, 0, criterion, channel, device)
                all_loss += loss.item()

                targets = inputs[:, 1:]
                total += targets.size(0)*targets.size(1)
                _, pred = outputs.topk(5, 2, True, True)
                # 最简洁的方式
                correct = pred.eq(targets.unsqueeze(2).repeat(1, 1, 5))
                # correct = pred.eq(targets.view(1, -1).expand_as(pred))
                top_1_correct += correct[:, :, :1].contiguous().view(-1).float().sum(0)
                top_5_correct += correct[:, :, :5].contiguous().view(-1).float().sum(0)
                desc = ('Loss: %.3f | Top1: %.3f%% | Top5: %.3f%% ' %
                        (all_loss / (batch_idx + 1), 100. * top_1_correct / total, 100. * top_5_correct / total))

                prog_bar.set_description(desc, refresh=True)
            return all_loss / (batch_idx + 1), 100. * float(top_1_correct / total), 100. * float(top_5_correct / total)

def init_pruner(net, config, writer, logger):
    if config.fisher_mode == 'hessian':
        pruner = HessianPruner(net,
                               Transformer,
                               config,
                               writer,
                               logger,
                               config.prune_ratio_limit,
                               '%s%d' % (config.network, config.depth),
                               batch_averaged=True,
                               use_patch=False,
                               fix_layers=0,
                               hessian_mode=config.hessian_mode,
                               use_decompose=config.use_decompose)
    else:
        raise NotImplementedError

    return pruner