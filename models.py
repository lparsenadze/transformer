import tensorflow as tf
from utils import positional_encoding
from layers import EncoderLayer


class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 num_layers, 
                 d_model, 
                 num_heads, 
                 ffn_dims, 
                 input_vocab_size, 
                 max_pos_tokens, 
                 dropout=0.1, 
                 epsilon=1e-6):
        
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.position_encoding = positional_encoding(max_pos_tokens, d_model)
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, ffn_dims, dropout=dropout, epsilon=epsilon)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training, mask):
        # x shape (batch_size, seq_len)
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        # Scale embedding by multiplying it by the square root of the embedding dimension
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.position_encoding[:, :seq_len, :]  # (..., seq_len, d_model)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x # (batch_size, seq_len, d_modelexit

