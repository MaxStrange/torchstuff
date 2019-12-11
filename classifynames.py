"""
RNN to classify names by country of origin.

Taken from https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
"""
# pylint: disable=no-member
# pylint: disable=not-callable
import argparse
import math
import models.classifyrnn as classifyrnn
import glob
import os
import random
import string
import time
import torch
import torch.utils.tensorboard as tensorboard
import unicodedata


def unicode_to_ascii(s: str, all_letters: str) -> str:
    """
    Converts the given `s` into an ASCII-encoded string
    from a Unicode-encoded one.
    """
    chars = []
    for c in unicodedata.normalize('NFD', s):
        if unicodedata.category(c) != 'Mn' and c in all_letters:
            chars.append(c)
    return "".join(chars)

def get_data() -> (dict, [str], int, int):
    """
    Returns
    -------

    * category_lines: a mapping of language to names
    * all_categories: list of all the languages
    * n_categories:   the number of languages
    * all_letters:    the letters in the alphabet we are using
    * n_data_samples: the total number of data points in the dataset

    """
    all_letters = string.ascii_letters + " .,;'"

    category_lines = {}
    all_categories = []
    n_data_samples = 0

    for fname in glob.glob("data/names/*.txt"):
        category, _ext = os.path.splitext(os.path.basename(fname))
        all_categories.append(category)

        with open(fname, encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
            lines = [unicode_to_ascii(line, all_letters) for line in lines]
            category_lines[category] = lines
            n_data_samples += len(lines)

    n_categories = len(all_categories)

    return category_lines, all_categories, n_categories, all_letters, n_data_samples

def time_since(t: float) -> str:
    """
    Returns the minutes and seconds since the given timestamp.
    """
    now = time.time()
    delta = now - t
    minutes = math.floor(delta / 60)
    seconds = delta - (minutes * 60)

    return f"{minutes}:{seconds:.2f}"

def name_to_tensor(name: str, all_letters: str) -> torch.tensor:
    """
    Converts `name` to a tensor of the form (ncharacters_in_name, 1, n_letters_in_alphabet),
    i.e., a stacked one-hot, with each vector encoding a character.

    Args
    ----

    * name: The name to convert
    * all_letters: The alphabet we are using

    """
    n_letters = len(all_letters)

    tensor = torch.zeros(len(name), 1, n_letters)
    for i, letter in enumerate(name):
        tensor[i][0][all_letters.find(letter)] = 1
    return tensor

def get_random_training_example(all_categories: [str], category_lines: dict, all_letters: str) -> (str, str, torch.tensor, torch.tensor):
    """
    Get a random instance of the data set, along with a Tensor-encoded version of it.

    Args
    ----

    * all_categories: A list of category names
    * category_lines: A dict of the form {category: [name, name, name]}
    * all_letters: The alphabet we are using.

    Returns
    -------

    * category (str)
    * name (str)
    * category as tensor
    * name as tensor

    """
    category = random.choice(all_categories)
    name = random.choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    name_tensor = name_to_tensor(name, all_letters)

    return category, name, category_tensor, name_tensor

def train(category_batch: [torch.Tensor], name_batch: [torch.Tensor], rnn: classifyrnn.RNN, lossfn: callable, learning_rate: float) -> (torch.tensor, float):
    """
    Train the network on the given batch of data.

    Args
    ----

    * category_batch: List of One-hot Tensors of classes
    * name_batch: List of Tensors of shape (n_chars in name, 1, n_letters_in_alphabet)
    * rnn: Model to train
    * lossfn: The loss function
    * learning_rate: The learning rate of the optimizer

    Returns

    """
    for category_tensor, name_tensor in zip(category_batch, name_batch):
        # Initialize the first pass of the hidden state
        hidden = rnn.init_hidden()

        # Initialize the gradient
        rnn.zero_grad()

        # Encode over the single given name instance
        for i in range(name_tensor.size()[0]):
            output, hidden = rnn(name_tensor[i], hidden)

        # Check decoded value against label
        loss = lossfn(output, category_tensor)
        loss.backward()

        # Add parameters' gradients to their values, multiplied by learning rate
        for p in rnn.parameters():
            p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()

def test(name_tensor: torch.Tensor, rnn: classifyrnn.RNN) -> torch.Tensor:
    """
    Test the network on the given input Tensor.
    """
    rnn.eval()

    hidden = rnn.init_hidden()
    for i in range(name_tensor.size()[0]):
        output, hidden = rnn(name_tensor[i], hidden)

    rnn.train()

    return output

def category_from_output(softmax: torch.tensor, all_categories: dict) -> (str, int):
    """
    Get a category string and index for the given softmax-encoded vector.
    """
    _top_n, top_i = softmax.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def get_random_batch(batchsize: int, all_categories: dict, category_lines: dict, all_letters: str, n_categories: int) -> ([torch.Tensor], [torch.Tensor]):
    """
    Gets a random batch of data and labels and returns them.
    """
    labels = []
    data = []
    for _ in range(batchsize):
        _, _, category_tensor, line_tensor = get_random_training_example(all_categories, category_lines, all_letters)
        labels.append(category_tensor)
        data.append(line_tensor)

    return labels, data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--learning-rate', default=0.005, type=float, help="Learning rate")
    parser.add_argument('--lossfn', default="NLLLoss", type=str, help="Loss function")
    parser.add_argument('--nepochs', default=10, type=int, help="Number of times through the data to train")
    parser.add_argument('--batchsize', default=32, type=int, help="Number of items to train on at a time")
    parser.add_argument('--print-every', default=150, type=int, help="Print summary every this many batches")
    args = parser.parse_args()

    # Hyper parameters
    learning_rate = args.learning_rate
    n_hidden = 128
    lossfn = getattr(torch.nn, args.lossfn)()
    batchsize = args.batchsize

    # Experiment parameters
    n_epochs = args.nepochs
    print_every = args.print_every

    # Data
    category_lines, all_categories, n_categories, all_letters, n_data_samples = get_data()
    n_letters = len(all_letters)

    # Network
    rnn = classifyrnn.RNN(n_letters, n_hidden, n_categories)

    # Tensorboard
    writer = tensorboard.SummaryWriter()

    # Train
    start = time.time()
    n_iters = math.ceil(n_data_samples * n_epochs / batchsize)

    for i in range(1, n_iters + 1):
        batch_labels, batch_data = get_random_batch(batchsize, all_categories, category_lines, all_letters, n_categories)

        output, loss = train(batch_labels, batch_data, rnn, lossfn, learning_rate)
        writer.add_scalar('loss', loss, i)

        if i % print_every == 0:
            category, line, category_tensor, line_tensor = get_random_training_example(all_categories, category_lines, all_letters)
            output = test(line_tensor, rnn)
            guess, guess_i = category_from_output(output, all_categories)
            correct = '✓' if guess == category else f"✗ ({category})"
            print(f"{i} {(i / (n_iters * 100)):.4f}% ({time_since(start)}) {loss:.4f}, {line} / {guess} {correct}")

    writer.close()
