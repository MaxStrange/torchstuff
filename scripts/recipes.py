"""
Fetches and preprocesses the recipe data.
"""
import argparse
import glob
import json
import os
import string
import subprocess
import unicodedata

# Where this script is located
MYLOC = os.path.split(os.path.abspath(__file__))[0]

# Where the root of the directory is (one up from our location)
ROOT = os.path.abspath(os.path.join(MYLOC, ".."))

# Where we will put the recipes directory
RECIPES_DIRPATH = os.path.join(ROOT, "recipes")

# Where we will put the reconstituted .txt file
TEXT_FPATH = os.path.join(RECIPES_DIRPATH, "recipes.txt")

# Allowed chars
ALLOWED_CHARS = string.ascii_letters + " .,;'"


def unicode_to_ascii(s: str) -> str:
    """
    Converts the given `s` into an ASCII-encoded string
    from a Unicode-encoded one.
    """
    chars = []
    for c in unicodedata.normalize('NFD', s):
        if unicodedata.category(c) != 'Mn' and c in ALLOWED_CHARS:
            chars.append(c)
    return "".join(chars)

def json_fpaths() -> [str]:
    """
    Returns the JSON file locations as a list.
    """
    return glob.glob(f"{RECIPES_DIRPATH}/*.json")

def fetch_the_data():
    """
    Fetches the data and places it in ROOT/recipes.
    """
    subprocess.run(["wget", "https://storage.googleapis.com/recipe-box/recipes_raw.zip"])
    subprocess.run(["unzip", "recipes_raw.zip", "-d", RECIPES_DIRPATH])
    subprocess.run(["rm", "recipes_raw.zip"])

def _reconstitute_recipe(txt, recipe: dict):
    """
    Recipes should each have the following items:

    - title (str)
    - ingredients (list)
    - instructions (str)
    - picture_link (str)

    We currently ignore the picture link in reconstituting the text.
    """
    title = unicode_to_ascii(recipe['title'].strip())

    reconsd = f"{title}\n{separator}\n\n{ingredients}\n{instructions}\n"
    txt.write(reconsd)

def reconstitute():
    """
    Attempt to reconstitute the recipe texts (.txt files) from the .json files in the data dump.

    # Format for typical recipe in recipes_raw_nosource_ar.json

    "clyYQv.CplpwJtjNaFGhx0VilNYqRxu": {
        "title": "Brown Sugar Meatloaf",
        "ingredients": [
        "1/2 cup packed brown sugar ADVERTISEMENT",
        "1/2 cup ketchup ADVERTISEMENT",
        "1 1/2 pounds lean ground beef ADVERTISEMENT",
        "3/4 cup milk ADVERTISEMENT",
        "2 eggs ADVERTISEMENT",
        "1 1/2 teaspoons salt ADVERTISEMENT",
        "1/4 teaspoon ground black pepper ADVERTISEMENT",
        "1 small onion, chopped ADVERTISEMENT",
        "1/4 teaspoon ground ginger ADVERTISEMENT",
        "3/4 cup finely crushed saltine cracker crumbs ADVERTISEMENT",
        "ADVERTISEMENT"
        ],
        "instructions": "Preheat oven to 350 degrees F (175 degrees C). Lightly grease a 5x9 inch loaf pan.\nPress the brown sugar in the bottom of the prepared loaf pan and spread the ketchup over the sugar.\nIn a mixing bowl, mix thoroughly all remaining ingredients and shape into a loaf. Place on top of the ketchup.\nBake in preheated oven for 1 hour or until juices are clear.\n",
        "picture_link": "LVW1DI0vtlCrpAhNSEQysE9i/7rJG56"
    },

    Other file's JSON formats are similar, but usually don't have 'ADVERTISEMENT' at the end of every ingredient.

    """
    with open(TEXT_FPATH, 'w') as txt:
        for jfpath in json_fpaths():
            with open(jfpath) as f:
                jstruct = json.load(f)

            for recipe in jstruct.keys():
                _reconstitute_recipe(txt, jstruct[recipe])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true', help="Get the recipe data from the web.")
    parser.add_argument('--reconstitute', action='store_true', help="Reconstitute the original(ish) texts from the JSON files in the data dump.")
    args = parser.parse_args()

    if args.fetch:
        fetch_the_data()

    if args.reconstitute:
        reconstitute()
