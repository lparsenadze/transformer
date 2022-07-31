import argparse

def get_preprocessing_args():
    """Get arguments needed in process.py."""
    parser = argparse.ArgumentParser('Pre-process for NMT with Transformer')

    parser.add_argument('--max_tokens',
                        type=int,
                        default=128)
    parser.add_argument('--train_features_file',
                        type=str,
                        default='data/train.npz')
    parser.add_argument('--test_features_file',
                        type=str,
                        default='data/npz')
    parser.add_argumetn('--tfds_name',
                        type=str,
                        default='ted_hrlr_translate/pt_to_en',
                        help='TensorFlow dataset for NMT')
    
    args = parser.parse_args()

    return args

