import random as rnd
import os
import datetime


path = os.path.dirname(__file__)
output_dir = path + "/../songs/locally-generated"
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)


def random_name():
    """
    Return a randomly generated midi file title
    """
    nouns_file = path + "/../words/music-nouns.txt"  # "english-nouns.txt"
    ajdectives_file = path + "/../words/english-adjectives.txt"

    with open(nouns_file) as f:
        nouns = f.readlines()

    with open(ajdectives_file) as f:
        adjectives = f.readlines()

    return f"{output_dir}/{rnd.choice(adjectives).strip()}-{rnd.choice(nouns).strip()}.mid"


def timestamp_name():
    """
    Return a midi file title with timestamp: 'song-YYMMDD-HHMMSS'
    """
    date_format = "%y%m%d-%H%M%S"
    return f"{output_dir}/song-{datetime.datetime.now().strftime(date_format)}.mid"
