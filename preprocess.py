import tensorflow_datasets as tfds
import tensorflow as tf
from args import get_preprocessing_args
from transformers import DistilBertTokenizerFast


def load_dataset(tfds_name):
    examples, _  = tfds.load(tfds_name, with_info=True,
                              as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']
    train = {'source': [], 'target': []}
    test = {'source': [], 'target': []}
    for pt, en in train_examples:
        train['source'].append(en.numpy().decode('utf-8'))
        train['target'].append(pt.numpy().decode('utf-8'))
        
    for pt, en in test_examples:
        test['source'].append(en.numpy().decode('utf-8'))
        test['target'].append(pt.numpy().decode('utf-8'))
    
    return train, test
         

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len) 


def convert_to_features(sentences, max_length, tokenizer, return_padding_masks=True):
    sequences = []
    padding_masks = []
    for sent in sentences:
        toks = tokenizer(sent, max_length=max_length, padding='max_length')
        seq = toks['input_id']
        mask = toks['attention_mask']
        if seq <= max_length:
            sequences.append(list(seq))
            padding_masks.append(list(mask))
    if return_padding_masks:
        return np.array(sequences), np.array(padding_masks)
    return np.array(sequences)

def get_angles(pos, k, d):
    """
    Get the angles for the positional encoding
    
    Arguments:
        pos -- Column vector containing the positions [[0], [1], ...,[N-1]]
        k --   Row vector containing the dimension span [[0, 1, 2, ..., d-1]]
        d(integer) -- Encoding size
    
    Returns:
        angles -- (pos, d) numpy array 
    """
    
    # Get i from dimension span k
    i = k // 2
    # Calculate the angles using pos, i and d
    angles = pos / 10000**(2*i / d) #(n x 1) / (1, d) -> (n x d)
    
    return angles    

def positional_encoding(positions, d):
    """
    Precomputes a matrix with all the positional encodings 
    
    Arguments:
        positions (int) -- Maximum number of positions to be encoded 
        d (int) -- Encoding size 
    
    Returns:
        pos_encoding -- (1, position, d_model) A matrix with the positional encodings
    """
    # START CODE HERE
    # initialize a matrix angle_rads of all the angles 
    angle_rads = get_angles(np.arrange(positions)[:, np.newaxis], # pos is a COLUMN vector of shape (N x 1), N - dims of the input 
                            np.arrange(d)[np.newaxis, :],         # k is a ROW vector of shape (1, d), d  - dims of the encoding 
                            d)
  
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    # END CODE HERE
    
    pos_encoding = angle_rads[np.newaxis, ...]  # shapes: (pos, d) -> (1, pos, d)
    
    return tf.cast(pos_encoding, dtype=tf.float32)

def plot_possitional_encoding(n, d):
    """
    Plots the positional encodings as a colormesh

    Arguments:
        n -- dims of the inputs
        d -- dims of the model
    
    Returns:
        colormesh  plot
    """

    pos_encoding = positional_encoding(n, d)
    pos_encoding = pos_encoding[0]

    # Juggle the dimensions for the plot
    pos_encoding = tf.reshape(pos_encoding, (n, d//2, 2))
    pos_encoding = tf.transpose(pos_encoding, (2, 1, 0))
    pos_encoding = tf.reshape(pos_encoding, (d, n))

    plt.pcolormesh(pos_encoding, cmap='RdBu')
    plt.ylabel('Depth')
    plt.xlabel('Position')
    plt.colorbar()
    plt.show()


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
    
    matmul_qk = tf.matmul(q, k.T)  # (..., seq_len_q, seq_len_k)
    
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
