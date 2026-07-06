"""
@Author: JW Deng
@Time: 2026/7/6 17:00
"""
from N2UQ.transformer_constellation import Transformer
# from N2UQ.quant_transformer import QuantTransformer
from N2UQ.N2UQ_transformer import N2UQTransformer
from pretrain_utils import load_model
from N2UQ.transformer_constellation import Transformer

def get_network(network, quant):
    if network == 'DeepSC':
        return DeepSC(
            num_layers=4,
            src_vocab_size=22234,
            trg_vocab_size=22234,
            src_max_len=22234,
            trg_max_len=22234,
            d_model=128,
            num_heads=8,
            dff=512,
            dropout=0.1,
            )
    elif network == 'Transformer':
        return Transformer(
            num_layers=4,
            src_vocab_size=22234,
            trg_vocab_size=22234,
            src_max_len=22234,
            trg_max_len=22234,
            d_model=128,
            num_heads=8,
            dff=512,
            dropout=0.1,
            quant=quant, quant_bits = 4
        )
    elif network == 'QuantTransformer':
        return QuantTransformer(
            num_layers=4,
            src_vocab_size=22234,
            trg_vocab_size=22234,
            src_max_len=22234,
            trg_max_len=22234,
            d_model=128,
            num_heads=8,
            dff=512,
            dropout=0.1)
    elif network == 'N2UQTransformer':
        return N2UQTransformer(
            num_layers=4,
            src_vocab_size=22234,
            trg_vocab_size=22234,
            src_max_len=22234,
            trg_max_len=22234,
            d_model=128,
            num_heads=8,
            dff=512,
            dropout=0.1,
            a_bits=4, w_bits = 4
        )
    else:
        raise NotImplementedError

def stablize_bn(net, trainloader, criterion, channel='AWGN', device='cuda'):
    """Iterate over the dataset for stabilizing the
    BatchNorm statistics.
    """
    net = net.train()
    for batch, inputs in enumerate(trainloader):
        inputs = inputs.to(device)
        net(inputs, 0.1, 0, criterion, channel)

def init_network(config, logger, device):
    net = get_network(network=config.network, quant = config.quant)
    print('==> Loading checkpoint from %s.' % config.load_checkpoint)
    logger.info('==> Loading checkpoint from %s.' % config.load_checkpoint)
    load_model(net, config.load_checkpoint)
    bottleneck_net = None

    return net.to(device), bottleneck_net