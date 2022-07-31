import tensorflow as tf
from utils import positional_encoding


def scaled_dot_product_attention(q, k, v, mask):
    """
    Calculate the attention weights.
      q, k, v must have matching leading dimensions.
      k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
      The mask has different shapes depending on its type(padding or look ahead) 
      but it must be broadcastable for addition.

    Arguments:
        q -- query shape == (..., seq_len_q, depth)
        k -- key shape == (..., seq_len_k, depth)
        v -- value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable 
              to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output -- attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None: # Don't replace this None
        scaled_attention_logits += (1 - mask) * -1e9 # set zero masks to -infinity so that after we apply softmax, padding tokens attention weights would be going to zero

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits , axis=-1)  # (..., seq_len_q, seq_len_k)


    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)


    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights


def FullyConnected(embedding_dim, fully_connected_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(fully_connected_dim, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(embedding_dim)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ffn_dims, dropout=0.1, epsilon=1e-6):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.normlayer1 = tf.keras.layers.LayerNormalization(epsilon=epsilon)
        self.normlayer2 = tf.keras.layers.LayerNormalization(epsilon=epsilon)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.ffn = FullyConnected(d_model, ffn_dims)

    def call(self, x, training, mask):
        """
            Q = K = V = x <- when Query Key and Value matricies are the same this is called self-attention
            We do self-attention here 
        """ 
        
        scaled_attention, _  = self.mha(x, x, x, mask)      # (batch_size, input_seq_len, d_model)
        out1 = self.normlayer1(x + scaled_attention)
        ffn_out = self.ffn(out1)                            # (batch_size, input_seq_len, d_model)
        ffn_out = self.dropout(ffn_out, training=training)  # (batch_size, input_seq_len, d_model)
        out2 = self.normlayer2(out1 + ffn_out)              # (batch_size, input_seq_len, d_model)

        return out2



class DecoderLayer(tf.keras.layer.Layer):
    def __init__(self, d_model, num_heads, ffn_dims, dropout=0.1m epsilon=1e-6);
        super(DecoderLayer, self).__init__():
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.normlayer1 = tf.keras.layer.LayerNormalization(epsilon=epsilon)
        self.normlayer2 = tf.keras.layer.LayerNormalization(epsilon=epsilon)
        self.normlayer3 = tf.keras.layer.LayerNormalization(epsilon=epsilon)
        self.ffn = FullyConnected(d_model, ffn_dims)
        self.dropout = tf.keras.layer.Droput(dropout)
    
    def call(self, x, training, mask, encoder_k, encoder_v):
        
        








