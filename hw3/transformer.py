import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def pad_tensors_to_window_size(q, k , v, half_window_size, padding_mask, paddin_value=0):
    seq_size = q.shape[-2]
    window = 2 * half_window_size
    padding_len = (window - seq_size % window) % window
    padding_l, padding_r = (padding_len//2, padding_len//2) if window > 2 else (0, 1)
    padding_dim = (0, 0, padding_l, padding_r)
    q = F.pad(q, padding_dim, value=paddin_value)
    k = F.pad(k, padding_dim, value=paddin_value)
    v = F.pad(v, padding_dim, value=paddin_value)
    if padding_mask is not None:
        padding_mask = F.pad(padding_mask, padding_dim[2:], value=0)
    return q, k, v, padding_mask

def _get_wrong_locations_mask(window, device):
    diags = []
    for j in range(-window, 1):
        diag_mask = torch.zeros(window, device='cpu', dtype=torch.uint8)
        diag_mask[:-j] = 1
        diags.append(diag_mask)
    mask = torch.stack(diags, dim=-1)
    mask = mask[None, :, None, :]
    mask_end = mask.flip(dims=(1, 3)).bool().to(device)
    return mask.bool().to(device), mask_end

def mask_wrong_elements(x, half_window_size):
    # get masks
    mask_begin, mask_end = _get_wrong_locations_mask(half_window_size, x.device)
    #apply begin mask
    seq_size = x.size(1)
    input_begin = x[:, :half_window_size, :, :half_window_size+1]
    mask_begin = mask_begin[:, :seq_size].expand_as(input_begin)
    input_begin.masked_fill_(mask_begin, -9e15)
    # apply end mask
    input_end = x[:, -half_window_size:, :, -(half_window_size+1):]
    mask_end = mask_end[:, -seq_size:].expand_as(input_end)
    input_end.masked_fill_(mask_end, -9e15)


def get_main_diagonals_indices(batches, seq_size, window):
    diag = torch.arange(-window, window + 1)
    row = torch.arange(0, seq_size * seq_size, seq_size + 1)
    col = row.reshape(1, -1, 1) + diag
    col = col.repeat(batches, 1, 1)
    return col.flatten(1)[:, window:-window]

def populate_diags(x):
    batches, seq_size, window = x.size()
    window = (window - 1)//2
    x= x.flatten(1)[:, window:-window].float()
    res = torch.zeros(batches, seq_size, seq_size, device=x.device).flatten(1)
    idx = get_main_diagonals_indices(batches, seq_size, window).to(x.device)
    res= res.scatter_(1, idx, x).view(batches, seq_size, seq_size)
    return res


def sliding_chunks_matmul_qk(q, k, half_window_size, padding_value):
    batches, number_of_heads, seq_size, head_dim = q.size()
    q = q.reshape(batches * number_of_heads, seq_size, head_dim)
    k = k.reshape(batches * number_of_heads, seq_size, head_dim)

    k_chunk = k.unfold(-2, 2*half_window_size, half_window_size).transpose(-1, -2)
    q_chunk = q.unfold(-2, 2*half_window_size, half_window_size).transpose(-1, -2)
    chunk_attention = torch.matmul(q_chunk, k_chunk.transpose(-2, -1))
    # convert diags into cols
    chunk_attention = nn.functional.pad(chunk_attention, (0, 0, 0, 1), value=padding_value)
    diag_chunk_attention = chunk_attention.view(*chunk_attention.size()[:-2], chunk_attention.size(-1), chunk_attention.size(-2))

    number_chunks = seq_size // half_window_size - 1
    # this will be the overall attention
    diag_attention = torch.full((batches * number_of_heads, number_chunks + 1, half_window_size, half_window_size * 2 + 1), fill_value=1, device=chunk_attention.device)*(-9e15)

    # copy upper tri and main diag
    diag_attention[:, :-1, :, half_window_size:] = diag_chunk_attention[:, :, :half_window_size, :(half_window_size + 1)]
    diag_attention[:, -1, :, half_window_size:] = diag_chunk_attention[:, -1, half_window_size:, :(half_window_size + 1)]
   # copy lower tri
    diag_attention[:, 1:, :, :half_window_size] = diag_chunk_attention[:, :, - (half_window_size + 1):-1, (half_window_size + 1):]
    diag_attention[:, 0, 1:half_window_size, 1:half_window_size] = diag_chunk_attention[:, 0, :(half_window_size - 1), ((half_window_size > 1) - half_window_size):]
    # restore to original dims
    diag_attention = diag_attention.reshape(batches, number_of_heads, seq_size, 2 * half_window_size + 1).transpose(2, 1)
    mask_wrong_elements(diag_attention, half_window_size)
    diag_attention = diag_attention.transpose(1,2).reshape(batches * number_of_heads, seq_size, 2 * half_window_size + 1)
    return diag_attention



def sliding_chunks_matmul_pv(probabilities, values, half_window_size):

    batches, seq_size, number_of_heads, head_dim = values.size()
    chunks_count = seq_size // half_window_size - 1
    chunk_prob = probabilities.transpose(1,2).reshape(batches * number_of_heads, seq_size // half_window_size, half_window_size, 2 * half_window_size + 1)

    values = values.transpose(1, 2).reshape(batches * number_of_heads, seq_size, head_dim)
    values = F.pad(values, (0, 0, half_window_size, half_window_size), value=-1)

    chunk_size = (batches * number_of_heads, chunks_count + 1, 3 * half_window_size, head_dim)
    chunk_stride = values.stride()
    chunk_stride = chunk_stride[0], half_window_size * chunk_stride[1], chunk_stride[1], chunk_stride[2]
    chunk_v = values.as_strided(size=chunk_size, stride=chunk_stride)
    B, C, M, L = chunk_prob.size()
    chunk_prob = F.pad(chunk_prob, (0, M + 1), value=0)
    chunk_prob = chunk_prob.reshape(B, C, -1)
    chunk_prob = chunk_prob[:, :, :-M]
    chunk_prob = chunk_prob.reshape(B, C, M, M + L)
    skewed_prob = chunk_prob[:, :, :, :-1]
    context = torch.matmul(skewed_prob, chunk_v)
    return context.reshape(batches, number_of_heads, seq_size, head_dim)
    

def sliding_window_attention(q, k, v, window_size, padding_mask=None):
    '''
    Computes the simple sliding window attention from 'Longformer: The Long-Document Transformer'.
    This implementation is meant for multihead attention on batched tensors. It should work for both single and multi-head attention.
    :param q - the query vectors. #[Batch, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param k - the key vectors.  #[Batch, *, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param v - the value vectors.  #[Batch, *, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param window_size - size of sliding window. Must be an even number.
    :param padding_mask - a mask that indicates padding with 0.  #[Batch, SeqLen]
    :return values - the output values. #[Batch, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :return attention - the attention weights. #[Batch, SeqLen, SeqLen] or [Batch, num_heads, SeqLen, SeqLen]
    '''
    assert window_size%2 == 0, "window size must be an even number"
    seq_size = q.shape[-2]
    batch_size = q.shape[0] 
    embed_dim = q.shape[-1]
    values, attention = None, None


    # TODO:
    #  Compute the sliding window attention.
    # NOTE: We will not test your implementation for efficiency, but you are required to follow these two rules:
    # 1) Implement the function without using for loops.
    # 2) DON'T compute all dot products and then remove the uneccessary comptutations 
    #    (both for tokens that aren't in the window, and for tokens that correspond to padding according to the 'padding mask').
    # Aside from these two rules, you are free to implement the function as you wish. 
    # ====== YOUR CODE: ======
    # Compute the sliding window attention
     # Compute left and right padding based on the window size

    half_window_size = window_size //2 

    single_head = False
    if len(q.shape) == 3:
        single_head = True
        q = q.unsqueeze(1)
    if len(k.shape) == 3:
        k = k.unsqueeze(1)
    if len(v.shape) == 3:
        v = v.unsqueeze(1)
    number_of_heads = q.shape[1]
    q, k, v, padding_mask = pad_tensors_to_window_size(q, k, v, half_window_size, padding_mask)
    new_seq_len = q.shape[-2]
    attention_weights = sliding_chunks_matmul_qk(q, k, half_window_size, padding_value=-9e15).view(batch_size, number_of_heads, new_seq_len, 2 * half_window_size + 1)

    if padding_mask is not None:
        padding_mask = torch.logical_not(padding_mask.unsqueeze(dim=1).unsqueeze(dim=-1))
        padding_mask = padding_mask.type_as(q).masked_fill(padding_mask, -9e15)
        ones = torch.ones_like(padding_mask)
        d_mask = sliding_chunks_matmul_qk(ones, padding_mask, half_window_size, padding_value=-9e15).view(batch_size, 1, new_seq_len, 2 * half_window_size + 1)  
        attention_weights += d_mask 
    attention =  torch.nn.functional.softmax(attention_weights / math.sqrt(embed_dim), dim=-1).transpose(1,2)

    values = sliding_chunks_matmul_pv(attention, v.transpose(1,2), half_window_size) #[batch_size,num_heads,seq_len, embed_dim]
    attention = attention.transpose(2,1)
    attention = attention.reshape(batch_size * number_of_heads, new_seq_len, 2 * half_window_size + 1)
    attention = populate_diags(attention)
    attention = attention.reshape(batch_size, number_of_heads, new_seq_len, new_seq_len)
    if new_seq_len != seq_size:
        padding_len = new_seq_len - seq_size
        padding_left, padding_right = (padding_len//2, padding_len//2) if padding_len > 1 else (0, 1)
        attention = attention[:, :, padding_left:-padding_right, padding_left:-padding_right]
        values = values[:, :, padding_left:-padding_right, :]
    if single_head:
        values = values.squeeze(1)
        attention = attention.squeeze(1)        
    return values, attention

class MultiHeadAttention(nn.Module):
    
    def __init__(self, input_dim, embed_dim, num_heads, window_size):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        
        # Stack all weight matrices 1...h together for efficiency
        # "bias=False" is optional, but for the projection we learned, there is no teoretical justification to use bias
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation of the paper if you would like....
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, padding_mask, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)
        
        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, 3*Dims]
        
        q, k, v = qkv.chunk(3, dim=-1) #[Batch, Head, SeqLen, Dims]
        
        # Determine value outputs
        # TODO:
        # call the sliding window attention function you implemented
        # ====== YOUR CODE: ======
        values, attention = sliding_window_attention(q, k, v, self.window_size, padding_mask)
        # ========================

        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim) #concatination of all heads
        o = self.o_proj(values)
        
        if return_attention:
            return o, attention
        else:
            return o
        
        
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000): 
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model) 
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
    
    

class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

    
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, window_size, dropout=0.1):
        '''
        :param embed_dim: the dimensionality of the input and output
        :param hidden_dim: the dimensionality of the hidden layer in the feed-forward network
        :param num_heads: the number of heads in the multi-head attention
        :param window_size: the size of the sliding window
        :param dropout: the dropout probability
        '''
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, embed_dim, num_heads, window_size)
        self.feed_forward = PositionWiseFeedForward(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, padding_mask):
        '''
        :param x: the input to the layer of shape [Batch, SeqLen, Dims]
        :param padding_mask: the padding mask of shape [Batch, SeqLen]
        :return: the output of the layer of shape [Batch, SeqLen, Dims]
        '''
        # TODO:
        #   To implement the encoder layer, do the following:
        #   1) Apply attention to the input x, and then apply dropout.
        #   2) Add a residual connection from the original input and normalize.
        #   3) Apply a feed-forward layer to the output of step 2, and then apply dropout again.
        #   4) Add a second residual connection and normalize again.
        # ====== YOUR CODE: ======
        weighted_values = self.self_attn(x, padding_mask)
        dropout = self.dropout(weighted_values)
        normalized_weighted_vals = self.norm1(x + dropout)

        dropout = self.dropout(self.feed_forward(normalized_weighted_vals))
        x = self.norm2(dropout + normalized_weighted_vals)
        # ========================
        
        return x
    
    
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_seq_length, window_size, dropout=0.1):
        '''
        :param vocab_size: the size of the vocabulary
        :param embed_dim: the dimensionality of the embeddings and the model
        :param num_heads: the number of heads in the multi-head attention
        :param num_layers: the number of layers in the encoder
        :param hidden_dim: the dimensionality of the hidden layer in the feed-forward network
        :param max_seq_length: the maximum length of a sequence
        :param window_size: the size of the sliding window
        :param dropout: the dropout probability

        '''
        super(Encoder, self).__init__()
        self.encoder_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(embed_dim, hidden_dim, num_heads, window_size, dropout) for _ in range(num_layers)])

        self.classification_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentence, padding_mask):
        '''
        :param sententence #[Batch, max_seq_len]
        :param padding mask #[Batch, max_seq_len]
        :return: the logits  [Batch]
        '''
        output = None

        # TODO:
        #  Implement the forward pass of the encoder.
        #  1) Apply the embedding layer to the input.
        #  2) Apply positional encoding to the output of step 1.
        #  3) Apply a dropout layer to the output of the positional encoding.
        #  4) Apply the specified number of encoder layers.
        #  5) Apply the classification MLP to the output vector corresponding to the special token [CLS] 
        #     (always the first token) to receive the logits.
        # ====== YOUR CODE: ======
        embedding = self.encoder_embedding.forward(sentence)
        positional_encoding = self.positional_encoding.forward(embedding)
        encoded_out = self.dropout(positional_encoding)

        for layer in self.encoder_layers:
            encoded_out = layer(encoded_out, padding_mask)
        
        output = self.classification_mlp(encoded_out[:, 0, :]).squeeze(-1)
        
        # ========================
        
        
        return output  
    
    def predict(self, sentence, padding_mask):
        '''
        :param sententence #[Batch, max_seq_len]
        :param padding mask #[Batch, max_seq_len]
        :return: the binary predictions  [Batch]
        '''
        logits = self.forward(sentence, padding_mask)
        preds = torch.round(torch.sigmoid(logits))
        return preds

    

