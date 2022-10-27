import tensorflow_datasets as tfds
import tensorflow as tf
from args import get_preprocessing_args
from transformers import DistilBertTokenizerFast
import numpy as np
import tensorflow_datasets as tfds


def load_dataset(tfds_name):
    examples, _  = tfds.load(tfds_name, with_info=True,
                              as_supervised=True)
    train_examples, test_examples = examples['train'], examples['validation']
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


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def get_seq_pairs(source_sents, target_sents, source_tokenizer, target_tokenizer, max_tokens=128):
    zipped_data = filter(lambda x: len(source_tokenizer(x[0])['input_ids'])<=max_tokens or len(target_tokenizer(x[1])['input_ids'])<=max_tokens, zip(source_sents, target_sents))
    zipped_data = list(zipped_data)
    source_sents_filtered, target_sents_filtered = list(zip(*zipped_data))
    source_seqs = source_tokenizer(list(source_sents_filtered), padding='max_length', max_length=max_tokens, return_tensors='pt')['input_ids']
    target_seqs = target_tokenizer(list(target_sents_filtered), padding='max_length', max_length=max_tokens, return_tensors='pt')['input_ids']

    return source_seqs, target_seqs

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
    # initialize a matrix angle_rads of all the angles 
    angle_rads = get_angles(np.arange(positions)[:, np.newaxis], # pos is a COLUMN vector of shape (N x 1), N - dims of the input 
                            np.arange(d)[np.newaxis, :],         # k is a ROW vector of shape (1, d), d  - dims of the encoding 
                            d)
  
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
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


