"""
@Author: JW Deng
@Time: 2026/1/27 19:20
"""
import bitsandbytes as bnb
import torch
import os
import tqdm
from utils import load_network

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
def save_model_4bit_int(model, path_folder, epoch):
# Replace the slow loop with vectorized operations
    """
    Optimized version using vectorized operations
    """
    quantized_state_dict = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            # 4-bit integer quantization
            param_flat = param.data.flatten()
            min_val = param_flat.min()
            max_val = param_flat.max()

            # Scale to [0, 15] range for 4-bit representation
            scaled_param = (param_flat - min_val) / (max_val - min_val) * 15
            quantized_param = torch.round(scaled_param).to(torch.uint8)

            # Vectorized packing - much faster
            # Pad with zero if odd number of elements
            if len(quantized_param) % 2 == 1:
                quantized_param = torch.cat([quantized_param, torch.tensor([0], dtype=torch.uint8, device=quantized_param.device)])

            # Reshape to pairs and pack
            paired_params = quantized_param.view(-1, 2)
            packed_param = (paired_params[:, 0] << 4) | (paired_params[:, 1] & 0x0F)

            quantized_state_dict[name] = {
                'data': packed_param,
                'min': min_val,
                'max': max_val,
                'original_shape': param.data.shape
            }

    # Save the quantized model
    save_path = os.path.join(path_folder, f'checkpoint_{epoch}_4bit_int.pth')
    with open(save_path, 'wb') as f:
        torch.save(quantized_state_dict, f)

    print(f"4-bit integer quantized model saved to {save_path}")

# Modify the model saving section (around line 142 or where you save the model)
def save_model_4bit(model, path_folder, epoch):
    """
    Save model with 4-bit quantization
    """
    # Save the quantized model
    save_path = os.path.join(path_folder, f'checkpoint_{epoch}_32bit.pth')
    with open(save_path, 'wb') as f:
        torch.save(model.state_dict(), f)
    # Quantize the model parameters to 4-bit before saving
    quantized_state_dict = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            # Use 4-bit quantization for trainable parameters
            quantized_param = bnb.nn.Params4bit(param.data, quant_type='fp4')  # or 'nf4'
            quantized_state_dict[name] = quantized_param

    # Save the quantized model
    save_path = os.path.join(path_folder, f'checkpoint_{epoch}_4bit.pth')
    with open(save_path, 'wb') as f:
        torch.save(quantized_state_dict, f)

    print(f"4-bit quantized model saved to {save_path}")

# Then replace the existing save_model call in your static quantization block:
# save_model(args, transformer, avg_test_loss, PATH_FOLDER, 0, out_indices)
# With:
# save_model_4bit(args, transformer, avg_test_loss, PATH_FOLDER, 0, out_indices)

PATH_FOLDER = './N2UQ/out/Rician_LTQ_Symmetric/pr_0.5_4bits_trace'
transformer, checkpoint, epoch_origin = load_network(PATH_FOLDER)
save_model_4bit_int(transformer, PATH_FOLDER, epoch_origin)
