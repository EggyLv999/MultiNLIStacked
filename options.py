from __future__ import print_function, division

import numpy as np
import argparse
import yaml

import sys

parser = argparse.ArgumentParser()

# Dynet parameters
# ================
parser.add_argument('--dynet-seed', default=0, type=int)
parser.add_argument('--dynet-mem', default=512, type=int)
parser.add_argument('--dynet-gpu', help='Use dynet with GPU', action='store_true')
parser.add_argument('--dynet-gpus', help='Number of GPUs', type=int, default=1)
parser.add_argument('--dynet-autobatch', default=0, type=int, help='Use dynet autobatching')

# Configuration
# =============
parser.add_argument('--config_file', '-c', default=None, type=str)
parser.add_argument('--env', '-e', help='Environment in the config file', default='train', type=str)
parser.add_argument('--exp_name', '-en', type=str, default='experiment', help='Name of the experiment')

# File/folder paths
# =================
parser.add_argument('--output_dir',help='output directory', type=str, default='.')
parser.add_argument('--temp_dir', help='Temp directory', type=str, default='temp')
# Trees
parser.add_argument('--train_file', help='Training data', type=str)
parser.add_argument('--dev_file', help='Validation data', type=str)
parser.add_argument('--test_file', help='Test data', type=str)
# Dataset options
parser.add_argument('--subtrees', help='Use subtrees for training', action='store_true')
parser.add_argument('--include_words', help='Include words for training', action='store_true')
parser.add_argument('--binary', help='Test trees', action='store_true')
# Other files
parser.add_argument('--vocab_file', help='File to save/load the vocabulary', type=str)
parser.add_argument('--pretrained_wembs',  help='File containing pretrained word embeddings', type=str)
parser.add_argument('--model_file', type=str, help='Model file ([exp_name]_model if not specified)')
parser.add_argument('--predict_file', help='Predictions of the model (for submission to Kaggle)', type=str)
parser.add_argument('--sentence_reps_file', help='Output file format string for sentence representations', type=str)

# Hyper-parameters
# ================
# Trainer
parser.add_argument('--trainer', type=str, help='Optimizer.For now only Adam', default='adam')
parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping')
parser.add_argument('--learning_rate', type=float, help='Learning rate', default=1.0)
parser.add_argument('--learning_rate_decay', type=float, help='Learning rate decay', default=0.0)
# Training and early stopping
parser.add_argument('--num_iters', type=int, default=1, help='Number of iterations of training')
parser.add_argument('--batch_size', type=int, help='minibatch size', default=20)
parser.add_argument('--patience', type=int, default=0, help='Patience before early stopping')
parser.add_argument('--self_loss_pct', type=float, default=1.0, help='Percentage of loss based on self attention')
parser.add_argument('--multi_reg', type=float, default=0.5, help='Percentage of loss based on correcting self att to look like cross att')
# Word embeddings
parser.add_argument('--vocab_size', type=int, help='Maximum vocab size', default=10000)
parser.add_argument('--min_freq', type=int, help='Minimum frequency under which words are unked', default=1)
parser.add_argument('--embed_dim', type=int, help='Word embedding dimension', default=100)
# Model type
parser.add_argument('--model_type', type=str, help='Model type to use', default='BILSTM',
                    choices = ['CBOW', 'BILSTM', 'SIF'])
# BILSTM hyper-parameters
parser.add_argument('--hidden_dim', type=list, help='Hidden state dimension', default=[100])
parser.add_argument('--compact_lstm', help='Use CompactLSTM for better memory efficiency', action='store_true')
parser.add_argument('--optimized_compare', help='Optimize the compare method by putting premise + hyp in a dingle batch (takes more memory)', action='store_true')
parser.add_argument('--att_dim', type=list, help='Hidden dimension for the attention network', default=[100])
# SIF hyper-parameters
parser.add_argument('--sif_a', type=float, help='Smoothing coefficient for SIF', default=0.001)
# Comparator
parser.add_argument('--comparator', type=str, help='Comparator type', default='mou')
# Classifier
parser.add_argument('--classifier', type=str, help='Classifier type', default='linear')
parser.add_argument('--classifier_num_layers', type=int, help='Number of layers (classifier)', default=1)
parser.add_argument('--classifier_input_dim', type=int, default=200,
                    help='Input dimension for the classifier (depends on the model and the task)')
parser.add_argument('--classifier_hidden_dim', type=int, help='Hidden dimension (classifier)', default=200)
parser.add_argument('--classifier_residual', help='Use residual connections in the classifier', action='store_true')
# Genre classifier (for adversarial training)
parser.add_argument('--genre_classifier', type=str, help='Genre classifier type', default='linear')
parser.add_argument('--genre_classifier_num_layers', type=int, help='Number of layers (genre classifier)', default=1)
parser.add_argument('--genre_classifier_hidden_dim', type=int, help='Hidden dimension (genre classifier)', default=200)
parser.add_argument('--adv_weight', type=float, help='Weight for the adversarial loss', default=0.0)
parser.add_argument('--adv_start_epoch', type=int, help='Epoch at which adversarial training should start', default=1)
# Regularization
parser.add_argument('--multi_labels', help='Whether to use all labels as a target', action='store_true')
parser.add_argument('--l2_reg', type=float, help='L2 regularization coefficient', default=0.0)
parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing coefficient')
parser.add_argument('--dropout_encoder', type=float, help='Dropout rate for the encoder', default=0.0)
parser.add_argument('--dropout', type=float, help='Dropout rate for the classifier', default=0.0)
parser.add_argument('--word_dropout', type=float, help='Word dropout rate', default=0.0)
parser.add_argument('--classifier_layer_norm', type=float, help='Layer norm in classifier', default=0.0)
# Language model
parser.add_argument('--laplace_smoothing', type=float, help='Smoothing coeff for the unigram LM', default=0.0)
# Misc
parser.add_argument('--max_len', type=int, help='Maximum allowed sentence length (in case of memory issues)', default=500)

# Flags
# =====
parser.add_argument('--verbose', '-v', help='increase output verbosity', action='store_true')
parser.add_argument('--train', help='Train model', action='store_true')
parser.add_argument('--test', help='Evaluate on test set', action='store_true')
parser.add_argument('--predict', help='Predict labels for unlabeled data', action='store_true')
parser.add_argument('--sentence_reps', help='Get sentence representation .npy files', action='store_true')
parser.add_argument('--saliency', help='What is our model attending to?', action='store_true')
parser.add_argument('--pretrained', help='Whether to use a pretrained model', action='store_true')


def parse_options():
    """Parse options from command line arguments and optionally config file

    Returns:
        Options
        argparse.Namespace
    """
    opt = parser.parse_args()
    if opt.config_file:
        with open(opt.config_file, 'r') as f:
            data = yaml.load(f)
            delattr(opt, 'config_file')
            arg_dict = opt.__dict__
            for key, value in data.items():
                if isinstance(value, dict):
                    if key == opt.env:
                        for k, v in value.items():
                            arg_dict[k] = v
                    else:
                        continue
                else:
                    arg_dict[key] = value
    # Little trick : add dynet general options to sys.argv if they're not here
    # already. Linked to this issue : https://github.com/clab/dynet/issues/475
    # sys.argv.append('--dynet-devices')
    # sys.argv.append('CPU,GPU:0')
    if opt.dynet_gpu and '--dynet-gpus' not in sys.argv:
        sys.argv.append('--dynet-gpus')
        sys.argv.append(str(opt.dynet_gpus))
    if '--dynet-autobatch' not in sys.argv:
        sys.argv.append('--dynet-autobatch')
        sys.argv.append(str(opt.__dict__['dynet_autobatch']))
    if '--dynet-mem' not in sys.argv:
        sys.argv.append('--dynet-mem')
        sys.argv.append(str(opt.__dict__['dynet_mem']))
    if '--dynet-seed' not in sys.argv:
        sys.argv.append('--dynet-seed')
        sys.argv.append(str(opt.__dict__['dynet_seed']))
        if opt.__dict__['dynet_seed'] > 0:
            np.random.seed(opt.__dict__['dynet_seed'])
    return opt


def print_config(opt, **kwargs):
    """Print the current configuration

    Prints command line arguments plus any kwargs

    Arguments:
        opt (argparse.Namespace): Command line arguments
        **kwargs: Any other key=value pair
    """
    print('======= CONFIG =======')
    for k, v in vars(opt).items():
        print(k, ':', v)
    for k, v in kwargs.items():
        print(k, ':', v)
    print('======================')


# Do this so sys.argv is changed upon import
options = parse_options()


def get_options():
    """Clean way to get options

    Returns:
        Options
        argparse.Namespace
    """
    return options
