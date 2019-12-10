"""
RNN to classify names by country of origin.

Taken from https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
"""
import models.classifyrnn as classifyrnn
import glob
import os
import string
import torch
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

def get_data() -> (dict, [str], int):
    """
    Returns
    -------

    * category_lines: a mapping of language to names
    * all_categories: list of all the languages
    * n_categories:   the number of languages
    * n_letters:      the number of letters in our name alphabet

    """
    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)

    category_lines = {}
    all_categories = []

    for fname in glob.glob("data/names/*.txt"):
        category, _ext = os.path.splitext(os.path.basename(fname))
        all_categories.append(category)

        with open(fname, encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
            lines = [unicode_to_ascii(line, all_letters) for line in lines]
            category_lines[category] = lines

    n_categories = len(all_categories)

    return category_lines, all_categories, n_categories, n_letters


if __name__ == "__main__":
    # Hyper parameters
    n_hidden = 128

    # Data
    category_lines, all_categories, n_categories, n_letters = get_data()

    # Network
    rnn = classifyrnn.RNN(n_letters, n_hidden, n_categories)
