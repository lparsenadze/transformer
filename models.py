import tensorflow as tf
from utils import positional_encoding, create_padding_mask, create_look_ahead_mask
from layers import EncoderLayer, DecoderLayer



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

class Decoder(tf.keras.layers.Layer):
    def __init__(self, 
                 num_layers,
                 d_model,
                 num_heads,
                 ffn_dims,
                 target_vocab_size,
                 max_pos_tokens,
                 dropout=0.1,
                 epsilon=1e-6):

        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
            
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, self.d_model)
        self.pos_encoding = positional_encoding(max_pos_tokens, self.d_model)
        self.dec_layers = [
            DecoderLayer(d_model, num_heads, ffn_dims, dropout=dropout, epsilon=epsilon)
            for _ in range(self.num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, encoder_out, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, encoder_out, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        return x, attention_weights



class Transformer(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, ffn_dims, input_vocab_size, target_vocab_size, max_pos_tokens,  dropout=0.1, epsilon=1e-6):
        
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, ffn_dims=ffn_dims,
                           input_vocab_size=input_vocab_size, max_pos_tokens=max_pos_tokens, dropout=dropout, epsilon=epsilon)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, ffn_dims=ffn_dims,
                           target_vocab_size=target_vocab_size, max_pos_tokens=max_pos_tokens, dropout=dropout, epsilon=epsilon)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, training):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs

        padding_mask, look_ahead_mask = self.create_masks(inp, tar)

        enc_output = self.encoder(inp, training, padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


    def create_masks(self, inp, tar):
        # Encoder padding mask (Used in the 2nd attention block in the decoder too.)
        padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return padding_mask, look_ahead_mask









