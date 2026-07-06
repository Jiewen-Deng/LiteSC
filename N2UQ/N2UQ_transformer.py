"""
Transformer-based Structure of nonuniform-to-uniform quantization-based semantic communication system.
@Author: JW Deng
@Time: 2026/7/6 12:00
"""
import torch
import torch.nn.functional as F
from N2UQ.utils_N2UQ import QuantizedLinear, QuantizedLinear_cons, AveragedRangeTracker, N2UQ_Symmetric_constellation
from N2UQ.utils import *

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model)) #math.log(math.exp(1)) = 1
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #[1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x
  
class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1, a_bits = 2, w_bits = 2):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.wq = QuantizedLinear(d_model, d_model, 
                                  a_bits = a_bits, w_bits = w_bits)
        self.wk = QuantizedLinear(d_model, d_model, 
                                  a_bits = a_bits, w_bits = w_bits)
        self.wv = QuantizedLinear(d_model, d_model, 
                                  a_bits = a_bits, w_bits = w_bits)

        self.dense = QuantizedLinear(d_model, d_model, 
                                  a_bits = a_bits, w_bits = w_bits)
        
        #self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    
    def attention(self, query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        #print(mask.shape)
        if mask is not None:
            scores += (mask * -1e9)
            # attention weights
        p_attn = F.softmax(scores, dim = -1)
        return torch.matmul(p_attn, value), p_attn
    
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query = self.wq(query)
        d_k = query.size(2) // self.num_heads
        query = query.view(nbatches, -1, self.num_heads, d_k)
        # query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k)
        query = query.transpose(1, 2)

        key = self.wk(key)
        d_k = key.size(2) // self.num_heads
        key = key.view(nbatches, -1, self.num_heads, d_k)
        # key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k)
        key = key.transpose(1, 2)

        value = self.wv(value)
        d_k = value.size(2) // self.num_heads
        value = value.view(nbatches, -1, self.num_heads, d_k)
        # value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k)
        value = value.transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.num_heads * d_k)

        x = self.dense(x)
        x = self.dropout(x)
        
        return x

    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1, a_bits = 2, w_bits = 2):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = QuantizedLinear(d_model, d_ff, 
                                   a_bits = a_bits, w_bits = w_bits) 
        self.w_2 = QuantizedLinear(d_ff, d_model, 
                                   a_bits = a_bits, w_bits = w_bits) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.w_2(x)
        x = self.dropout(x) 
        return x
    
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, d_model, num_heads, dff, dropout = 0.1, a_bits = 2, w_bits = 2):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadedAttention(num_heads, d_model, dropout, a_bits, w_bits)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout, a_bits, w_bits)
        
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        attn_output = self.mha(x, x, x, mask)
        x = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + ffn_output)
        
        return x
    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, d_model, num_heads, dff, dropout, a_bits, w_bits):
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadedAttention(num_heads, d_model, dropout, a_bits, w_bits)
        self.src_mha = MultiHeadedAttention(num_heads, d_model, dropout, a_bits, w_bits)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout, a_bits, w_bits)
        
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
        
        #self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        "Follow Figure 1 (right) for connections."
        #m = memory
        
        attn_output = self.self_mha(x, x, x, look_ahead_mask)
        x = self.layernorm1(x + attn_output)
        
        src_output = self.src_mha(x, memory, memory, trg_padding_mask) # q, k, v
        x = self.layernorm2(x + src_output)
        
        fnn_output = self.ffn(x)
        x = self.layernorm3(x + fnn_output)
        return x

    
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, num_layers, src_vocab_size, max_len, d_model, 
                 num_heads, dff, dropout = 0.1, a_bits = 2, w_bits = 2):
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, dropout, a_bits, w_bits) 
                                            for _ in range(num_layers)])
        
    def forward(self, x, src_mask):
        "Pass the input (and mask) through each layer in turn."
        # the input size of x is [batch_size, seq_len]
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for enc_layer in self.enc_layers:
            x = enc_layer(x, src_mask)
        
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers, trg_vocab_size, max_len, d_model, 
                 num_heads, dff, dropout = 0.1, a_bits = 2, w_bits = 2):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, dropout, a_bits, w_bits) 
                                            for _ in range(num_layers)])
    
    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for dec_layer in self.dec_layers:
            x = dec_layer(x, memory, look_ahead_mask, trg_padding_mask)
            
        return x

class ChannelDecoder(nn.Module):
    def __init__(self, in_features, size1, size2, a_bits, w_bits):
        super(ChannelDecoder, self).__init__()
        
        self.linear1 = QuantizedLinear_cons(in_features, size1,
                                       a_bits = a_bits, w_bits = w_bits)
        self.linear2 = QuantizedLinear(size1, size2,
                                       a_bits = a_bits, w_bits = w_bits)
        self.linear3 = QuantizedLinear(size2, size1,
                                       a_bits = a_bits, w_bits = w_bits)
        # self.linear4 = nn.Linear(size1, d_model)
        
        self.layernorm = nn.LayerNorm(size1, eps=1e-6)
        
    def forward(self, x):
        x1 = self.linear1(x)
        x2 = F.relu(x1)
        x3 = self.linear2(x2)
        x4 = F.relu(x3)
        x5 = self.linear3(x4)
        
        output = self.layernorm(x1 + x5)
        
        # output = self.linear4(output)
        
        return output

        
class N2UQTransformer(nn.Module):
# class QuantTransformer(nn.Module):
    def __init__(self, num_layers, src_vocab_size, trg_vocab_size, src_max_len,
                 trg_max_len, d_model, num_heads, dff, dropout = 0.1, a_bits = 2, w_bits = 2):
        super(N2UQTransformer, self).__init__()
        
        self.encoder = Encoder(num_layers, src_vocab_size, src_max_len, 
                               d_model, num_heads, dff, dropout, a_bits, w_bits)
        
        self.channel_encoder = nn.Sequential(QuantizedLinear(d_model, 256, 
                                                             a_bits = a_bits, w_bits = w_bits),
                                             nn.ReLU(inplace = True),
                                             QuantizedLinear(256, 16, 
                                                             a_bits = a_bits, w_bits = w_bits))
        
        self.quant_constellation = N2UQ_Symmetric_constellation(bits = a_bits, range_tracker = AveragedRangeTracker(q_level='L'))
        # self.quant_constellation = AsymmetricQuantizer(bits = 4, range_tracker = AveragedRangeTracker(q_level='L'))

        self.channel_decoder = ChannelDecoder(16, d_model, 512, a_bits = a_bits, w_bits = w_bits)
        
        self.decoder = Decoder(num_layers, trg_vocab_size, trg_max_len, 
                               d_model, num_heads, dff, dropout, a_bits, w_bits)
        
        self.dense = QuantizedLinear(d_model, trg_vocab_size, a_bits = a_bits, w_bits = w_bits)

        self.ntokens = trg_vocab_size

    def forward(self, src, n_var, pad, criterion, channel='AWGN', device = 'cpu'):
        trg = src
        trg_inp = trg[:, :-1]
        trg_real = trg[:, 1:]
        channels = Channels()

        # opt.zero_grad()
        src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)

        enc_output = self.encoder(src, src_mask)
        channel_enc_output = self.channel_encoder(enc_output)
        channel_enc_output_quant = self.quant_constellation(channel_enc_output)  # 新增量化
        Tx_sig = PowerNormalize(channel_enc_output_quant)

        if channel == 'AWGN':
            Rx_sig = channels.AWGN(Tx_sig, n_var)
        elif channel == 'Rayleigh':
            Rx_sig = channels.Rayleigh(Tx_sig, n_var)
        elif channel == 'Rician':
            Rx_sig = channels.Rician(Tx_sig, n_var)
        else:
            raise ValueError("Please choose from AWGN, Rayleigh, and Rician")
        channel_dec_output = self.channel_decoder(Rx_sig)
        dec_output = self.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
        pred = self.dense(dec_output)
        ntokens = self.ntokens
        if hasattr(self.dense, 'out_indices'):
            expanded_pred = torch.zeros(pred.shape[:-1] + (ntokens,), device=pred.device, dtype=pred.dtype)
            expanded_pred[:, :, self.dense.out_indices] = pred
            pred = expanded_pred

        loss = loss_function(pred.contiguous().view(-1, ntokens),
                             trg_real.contiguous().view(-1),
                             pad, criterion)

        return pred, loss
    
    
    
    


    


