"""
@Author: JW Deng
@Time: 2026/7/6 17:00
"""
import torch.nn as nn
from HAP.utils.prune_utils import ConvLayerRotation, LinearLayerRotation

def compute_transformer_flops(model, seq_len, cuda=False):
    """
    Calculate FLOPs for pruned Transformer model

    Args:
        model: Pruned Transformer model instance
        seq_len: Sequence length
        cuda: Whether to use CUDA

    Returns:
        total_flops: Total FLOPs
        rotation_flops: Rotation layer FLOPs (here is 0)
    """
    total_flops = 0

    # 1. Encoder Layers FLOPs (calculate layer by layer, considering pruned dimensions)
    for layer in model.encoder.enc_layers:
        # Multi-Head Attention dimension acquisition
        d_model_q_in = layer.mha.wq.in_features      # Q input dimension
        d_model_q_out = layer.mha.wq.out_features    # Q output dimension
        d_model_k_in = layer.mha.wk.in_features      # K input dimension
        d_model_k_out = layer.mha.wk.out_features    # K output dimension
        d_model_v_in = layer.mha.wv.in_features      # V input dimension
        d_model_v_out = layer.mha.wv.out_features    # V output dimension
        d_model_o_in = layer.mha.dense.in_features      # Output projection input dimension
        d_model_o_out = layer.mha.dense.out_features    # Output projection output dimension

        dff_in = layer.ffn.w_1.in_features            # FFN input dimension
        dff_out = layer.ffn.w_2.out_features          # FFN output dimension
        dff_hidden = layer.ffn.w_1.out_features       # FFN hidden layer dimension

        # Multi-Head Attention FLOPs (calculated according to each weight dimension)
        mha_linear_flops = (d_model_q_in * d_model_q_out +     # Q linear transformation
                           d_model_k_in * d_model_k_out +      # K linear transformation
                           d_model_v_in * d_model_v_out) * seq_len      # V linear transformation
        attention_score_flops = seq_len * seq_len * d_model_k_out  # Q*K^T
        attention_value_flops = seq_len * seq_len * d_model_v_out  # Score*V
        attention_flops = attention_score_flops + attention_value_flops
        output_linear_flops = d_model_o_in * d_model_o_out * seq_len    # Output transformation (after Concat projection)
        layernorm_flops = 2 * d_model_o_out * seq_len                   # LayerNorm

        # Position-wise Feed Forward FLOPs
        ffn_flops = (dff_in * dff_hidden + dff_hidden * dff_out) * seq_len # Two linear layers

        encoder_layer_flops = mha_linear_flops + attention_flops + output_linear_flops + \
                              2 * layernorm_flops + ffn_flops
        total_flops += encoder_layer_flops

    # 2. Decoder Layers FLOPs (similar processing)
    for layer in model.decoder.dec_layers:
        # Self-Attention dimension acquisition
        self_d_model_q_in = layer.self_mha.wq.in_features
        self_d_model_q_out = layer.self_mha.wq.out_features
        self_d_model_k_in = layer.self_mha.wk.in_features
        self_d_model_k_out = layer.self_mha.wk.out_features
        self_d_model_v_in = layer.self_mha.wv.in_features
        self_d_model_v_out = layer.self_mha.wv.out_features
        self_d_model_o_in = layer.self_mha.dense.in_features
        self_d_model_o_out = layer.self_mha.dense.out_features

        # Source Attention dimension acquisition
        src_d_model_q_in = layer.src_mha.wq.in_features
        src_d_model_q_out = layer.src_mha.wq.out_features
        src_d_model_k_in = layer.src_mha.wk.in_features
        src_d_model_k_out = layer.src_mha.wk.out_features
        src_d_model_v_in = layer.src_mha.wv.in_features
        src_d_model_v_out = layer.src_mha.wv.out_features
        src_d_model_o_in = layer.src_mha.dense.in_features
        src_d_model_o_out = layer.src_mha.dense.out_features

        dff_in = layer.ffn.w_1.in_features
        dff_out = layer.ffn.w_2.out_features
        dff_hidden = layer.ffn.w_1.out_features

        # Self-Attention FLOPs
        self_mha_linear_flops = (self_d_model_q_in * self_d_model_q_out +
                                self_d_model_k_in * self_d_model_k_out +
                                self_d_model_v_in * self_d_model_v_out) * seq_len
        self_attention_flops = seq_len * seq_len * self_d_model_k_out * 2
        self_output_linear_flops = self_d_model_o_in * self_d_model_o_out * seq_len
        self_layernorm_flops = 2 * self_d_model_o_out * seq_len

        # Source Attention FLOPs
        src_mha_linear_flops = (src_d_model_q_in * src_d_model_q_out +
                               src_d_model_k_in * src_d_model_k_out +
                               src_d_model_v_in * src_d_model_v_out) * seq_len
        src_attention_flops = seq_len * seq_len * src_d_model_k_out * 2
        src_output_linear_flops = src_d_model_o_in * src_d_model_o_out * seq_len
        src_layernorm_flops = 2 * src_d_model_o_out * seq_len

        # Feed Forward FLOPs
        decoder_ffn_flops = (dff_in * dff_hidden + dff_hidden * dff_out) * seq_len

        decoder_layer_flops = (
            self_mha_linear_flops + self_attention_flops + self_output_linear_flops + self_layernorm_flops +
            src_mha_linear_flops + src_attention_flops + src_output_linear_flops + src_layernorm_flops +
            decoder_ffn_flops)
        total_flops += decoder_layer_flops

    # 3. Channel related layer FLOPs (based on actual weight dimensions)
    channel_encoder_flops = 0
    channel_decoder_flops = 0

    # Channel Encoder FLOPs calculation
    if hasattr(model, 'channel_encoder'):
        # Directly get the input and output dimensions of the linear layer
        if hasattr(model.channel_encoder, '0'):  # First Linear layer
            layer1 = model.channel_encoder[0]
            in_features1 = layer1.in_features
            out_features1 = layer1.out_features
            channel_encoder_flops += (in_features1 * out_features1 + out_features1) * seq_len  # First layer FLOPs

        if hasattr(model.channel_encoder, '2'):  # Second Linear layer (index 2 because ReLU is in the middle)
            layer2 = model.channel_encoder[2]
            in_features2 = layer2.in_features
            out_features2 = layer2.out_features
            channel_encoder_flops += (in_features2 * out_features2 + out_features2) * seq_len  # Second layer FLOPs
    total_flops += channel_encoder_flops
    # Channel Decoder FLOPs calculation
    if hasattr(model, 'channel_decoder'):
        # Get the input and output dimensions of linear layers layer by layer
        linear_layers = []
        for layer in model.channel_decoder.children():
            if isinstance(layer, nn.Linear):
                linear_layers.append(layer)

        # Calculate FLOPs for each linear layer
        for linear_layer in linear_layers:
            in_features = linear_layer.in_features
            out_features = linear_layer.out_features
            channel_decoder_flops += (in_features * out_features + out_features) * seq_len

        # Add LayerNorm FLOPs (if exists)
        for layer in model.channel_decoder.children():
            if isinstance(layer, nn.LayerNorm):
                norm_features = layer.normalized_shape[0] if isinstance(layer.normalized_shape,
                                                                        tuple) else layer.normalized_shape
                channel_decoder_flops += 2 * norm_features * seq_len  # LayerNorm approximate FLOPs
    total_flops += channel_decoder_flops
    # 4. Final Dense Layer FLOPs
    final_layer = model.dense
    d_model_final = final_layer.in_features
    vocab_size = final_layer.out_features
    final_dense_flops = (d_model_final * vocab_size + vocab_size) * seq_len

    total_flops += final_dense_flops

    return total_flops, 0

def count_parameters_embedding(model):
    """The number of trainable parameters.
    It will exclude the rotation matrix in bottleneck layer.
    For embedding layers, replace vocabulary size with dense layer's output dimension.
    """
    total_params = 0
    dense_out_features = model.dense.out_features  # Get the output dimension of dense layer

    for name, p in model.named_parameters():
        # if p.requires_grad:  # Only count trainable parameters
        if 'embedding' in name:  # Handle embedding layer specially
            # Replace vocabulary size with dense layer's output dimension
            param_count = p.numel() // p.size(0) * dense_out_features
            total_params += param_count
        else:
            total_params += p.numel()

    return total_params

def count_rotation_numels(model):
    """Count how many parameters in the rotation matrix.
    Call this only when they are not trainable for complementing
    the number of parameters.
    """
    total = 0
    for m in model.modules():
        if isinstance(m, (ConvLayerRotation, LinearLayerRotation)):
            total += m.rotation_matrix.numel()
    return total

def compute_ratio(model, total, fix_rotation, logger):
    indicator = 1 if fix_rotation else 0
    rotation_numel = count_rotation_numels(model)
    pruned_numel = count_parameters_embedding(model) + rotation_numel*indicator
    # pruned_numel = count_parameters(model) + rotation_numel*indicator
    ratio = 100. * pruned_numel / total
    logger.info('Compression ratio: %.2f%%(%d/%d), Total: %d, Rotation: %d.' % (ratio,
                                                                                pruned_numel,
                                                                                total,
                                                                                pruned_numel,
                                                                                rotation_numel))
    unfair_ratio = 100 - 100. * (pruned_numel - rotation_numel*indicator)
    return ratio, unfair_ratio, pruned_numel, rotation_numel