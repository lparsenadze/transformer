from models import Transformer
from utils import load_dataset, get_seq_pairs
from transformers import BertTokenizerFast
import tensorflow as tf

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred) 

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.cast(tf.argmax(pred, axis=2), dtype=real.dtype))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


def main():
    train, test = train, test = load_dataset('ted_hrlr_translate/ru_to_en')
    source_tokenizer = BertTokenizerFast.from_pretrained('DeepPavlov/rubert-base-cased')
    target_tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')     
    max_tokens = 256
    ## source is russian, target is english 
    train_source_seq, train_target_seq = get_seq_pairs(
        train['source'], train['target'], target_tokenizer, source_tokenizer, max_tokens)
    

    num_layers = 4
    d_model = 128
    dff = 512
    num_attention_heads = 8
    dropout_rate = 0.1        
    learning_rate = 0.001

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_attention_heads,
        ffn_dims=dff,
        input_vocab_size=source_tokenizer.vocab_size,
        target_vocab_size=target_tokenizer.vocab_size,
        max_pos_tokens=max_tokens,
        dropout=dropout_rate)


    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    for epoch in range(20):
        train_loss.reset_states()
        train_accuracy.reset_states()
        for (batch, (inp, tar)) in enumerate(ds.batch(16)):
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]
            with tf.GradientTape() as tape:
                predictions, _ = transformer([inp, tar_inp],
                                     training = True)
                loss = loss_function(tar_real, predictions)
    
            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    
            train_loss(loss)
            train_accuracy(accuracy_function(tar_real, predictions))
            if batch % 50 == 0:
                print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
            print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
        

if __name__ == "__main__":
    main()


 
