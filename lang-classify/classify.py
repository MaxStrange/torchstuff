"""
Main entry for the language classification task. This script provides a means
to train, save, load, and run models for this task.
"""
import argparse
import configs.configuration as configuration
import os
import srcmodels.lstm as srclstm
import torch


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=str, help="Name of experiment file or path to it.")
    args = parser.parse_args()

    # Handle args
    configfpath = _get_path_to_config_file(args.config)
    cnfparser = configuration.load(configfpath)

    model = _load_model(cnfparser)

    ## Train it if we are training
    #model.train()

    ## Evaluate it if we are evaluating it
    #model.evaluate()
