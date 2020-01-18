"""
Main entry for the language classification task. This script provides a means
to train, save, load, and run models for this task.

Some of this taken from https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html.
"""
import argparse
import configs.configuration as configuration
import math
import os
import srcdatasets.dataset as srcdata
import srcdatasets.commonvoice as commonvoice
import srcmodels.lstm as srclstm
import time
import torch
import torch.utils.tensorboard as tensorboard


MY_DIR = os.path.abspath(os.path.dirname(__file__))


def _get_feature_vector_length(cnfparser: configuration.Configuration) -> int:
    """
    Determines how long each feature vector is based on the configuration and returns it.
    """
    feature_type = cnfparser.getstr("Features", "type").lower()

    if feature_type == "raw":
        # We are feeding raw audio samples into the network. Each sample is a scalar.
        return 1
    else:
        raise NotImplementedError(f"Have not yet implemented feature type {feature_type} (or it is misspelled).")

def _get_path_to_config_file(name_or_path: str) -> str:
    """
    Check the given `name_or_path` to determine which it is: a path or a name
    and then find the path to it.

    If it is a name, we find the right file from ./configs.

    If we can't find the file, we throw a FileNotFoundError.
    """
    if os.path.exists(name_or_path):
        # This is a file path; nothing to be done
        return name_or_path
    else:
        # Check the configs directory for something that could be it
        configdir = os.path.join(MY_DIR, "configs")
        for fname in os.listdir(configdir):
            if fname.lower().startswith(name_or_path.lower()):
                return os.path.join(configdir, fname)

        raise FileNotFoundError(f"Cannot find any config file matching {name_or_path}")

def _load_dataset(cnfparser: configuration.Configuration) -> srcdata.DataSet:
    """
    Figure out from the configuration which dataset and then load it.
    """
    dataset_name = cnfparser.getstr("Dataset", "dataset").lower()

    if dataset_name == "commonvoice":
        return commonvoice.CommonVoice(cnfparser.getstr("Dataset", "path-to-root"))
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not yet implemented or is misspelled.")

def _load_model(cnfparser: configuration.Configuration) -> torch.nn.Module:
    """
    Figure out from the configuration which type of model and whether to load a
    trained one or to train a new one.
    """
    modeltype = cnfparser.getstr("Hyperparameters", "model-type").lower()

    if modeltype == "lstm":
        return srclstm.LSTMModel(input_size=_get_feature_vector_length(cnfparser),
                                 hidden_size=cnfparser.getint("Hyperparameters", "hidden-layer-size"),
                                 num_layers=cnfparser.getint("Hyperparameters", "hidden-layers"),
                                 dropout=cnfparser.getfloat("Hyperparameters", "dropout-probability"),
                                 bidirectional=cnfparser.getbool("Hyperparameters", "bidirectional"))
    else:
        raise NotImplementedError(f"Have not yet implemented model type {modeltype} (or it was misspelled)")


def train(model: torch.nn.Module, dataset: srcdata.DataSet, cnfparser: configuration.Configuration, print_every: int, writer=None):
    """
    Train the model and save it per configuration specifications.
    """
    batchsize = cnfparser.getint("Experiment Parameters", "batch-size")
    n_epochs = cnfparser.getint("Experiment Parameters", "number-of-epochs")
    n_batches = math.ceil(dataset.n_data_points * n_epochs / batchsize)

    start = time.time()
    for i in range(1, n_batches + 1):
        batch_labels, batch_data = dataset.get_random_batch(batchsize)

        output, loss = _train_one_batch()

        if writer is not None:
            writer.add_scalar('loss', loss, i)

        if (print_every > 0) and (i % print_every == 0):
            category, line, category_tensor, line_tensor = get_random_training_example(all_categories, category_lines, all_letters)
            output = test(line_tensor, rnn)
            guess, guess_i = category_from_output(output, all_categories)
            correct = '✓' if guess == category else f"✗ ({category})"
            print(f"{i} {(i / (n_batches * 100)):.4f}% ({time_since(start)}) {loss:.4f}, {line} / {guess} {correct}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=str, help="Name of experiment file or path to it.")
    parser.add_argument('--print-every', type=int, default=150, help="Print a summary every this many batches. Give 0 or negative for never.")
    parser.add_argument('--tensorboard', action='store_true', help="Should we track this with Tensorboard?")
    args = parser.parse_args()

    # Handle args
    configfpath = _get_path_to_config_file(args.config)
    cnfparser = configuration.load(configfpath)
    print_every = args.print_every
    writer = tensorboard.SummaryWriter() if args.tensorboard else None

    # Load up the model
    model = _load_model(cnfparser)

    # Load up the dataset
    dataset = _load_dataset(cnfparser)

    # Train the model
    train(model, dataset, cnfparser, print_every, writer=writer)
