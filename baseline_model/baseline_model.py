# File: baseline_model.py
# Description: Implements a baseline sentiment analysis model using spaCy.
# Author: Jacob Tinkelman & Michelle Zuckerberg
# Date: December 17, 2024

import spacy
from spacy.training import Example
from typing import Iterable, List, Tuple
import numpy as np
from spacy.scorer import Scorer
from spacy.util import minibatch
import re
import random
import pickle

# This sets the seed for the random number generator.
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)

# USEFUL LINKS
# https://spacy.io/api/doc
# https://spacy.io/api/example
# https://spacy.io/api/textcategorizer


# This function processes the kaggle data and returns a list of tuples. Each tuple contains a processed sentence and a sentiment dictionary.
def process_kaggle(file_path: str) -> List[Tuple[str, dict[str, dict[str, float]]]]:
    """
    Process a text file with sentences and binary sentiment labels.
    
    Args:
        file_path (str): Path to the input text file
    
    Returns:
        List of tuples, each containing:
        - Lowercase processed sentence
        - Sentiment dictionary with POSITIVE and NEGATIVE probabilities
    """
    processed_data = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Split the line into text and label
            parts = line.strip().split('\t')
            
            # Handle cases with unexpected line format
            if len(parts) != 2:
                continue
            
            # Extract text and label
            text, label = parts[0], parts[1]
            
            # Preprocess text: lowercase, remove extra whitespace
            cleaned_text = re.sub(r'\s+', ' ', text.lower()).strip()
            
            # Create sentiment dictionary based on label
            sentiment = {
                "cats": {
                    "POSITIVE": float(label),
                    "NEGATIVE": 1.0 - float(label)
                }
            }
            
            processed_data.append((cleaned_text, sentiment))
    return processed_data


# This is the path to the training, dev, and test sets.
TRAINING_PATH = 'training_set.txt'
DEV_PATH = "dev_set.txt"
TEST_PATH = "test_set.txt"

TRAINING_DATA = process_kaggle(TRAINING_PATH)
DEV_DATA = process_kaggle(DEV_PATH)
TEST_DATA = process_kaggle(TEST_PATH)

# This is the model that we are using.
nlp = spacy.load("en_core_web_md") 

# This is the configuration for the model.
config = {
    "threshold": 0.5,
    "model": {
        "@architectures": "spacy.TextCatEnsemble.v2",
        "tok2vec": {
            "@architectures": "spacy.Tok2Vec.v2",
            "embed": {
                "@architectures": "spacy.MultiHashEmbed.v2",
                "width": 64,
                "rows": [2000, 2000, 500, 1000, 500],
                "attrs": ["NORM", "LOWER", "PREFIX", "SUFFIX", "SHAPE"],
                "include_static_vectors": False,
            },
            "encode": {
                "@architectures": "spacy.MaxoutWindowEncoder.v2",
                "width": 64,
                "window_size": 1,
                "maxout_pieces": 3,
                "depth": 2,
            },
        },
        "linear_model": {
            "@architectures": "spacy.TextCatBOW.v3",
            "length": 262144,
            "no_output_layer": False,
        },
    },
}


# A list of example( the datatype ) which textcat trains on.
training_examples = [] 

# This function makes the model data. It is a list of examples for the model.
def make_model_data(data: tuple[str, dict[str, dict[str, float]]]) -> Iterable[Example]:
    example_list = [] # This is the list of examples for the model.
    for text, annotations in data:

        doc_gold = nlp(text)
        doc_guess = nlp(text)

        doc_gold.cats = annotations["cats"]

        an_example = Example(doc_guess, doc_gold)
        example_list.append(an_example)

    return example_list

training_examples = make_model_data(TRAINING_DATA)

textcat = nlp.add_pipe("textcat") 
textcat.add_label("POSITIVE")
textcat.add_label("NEGATIVE")

# Confirm the pipeline components
print(f"nlp: pipe names: {nlp.pipe_names}")

# This function trains the model.
def train(train_example, a_batch_size, an_epoch, a_dropnum, a_learnrate):
    """this actually tains the model based on the data that is examples """
    optimizer = nlp.begin_training()  # Initialize optimizer
    optimizer.learn_rate = a_learnrate

    # Set all components to be updated during training (this is important)
    for epoch in range(an_epoch):  # Number of epochs
        losses = {}
        # Create mini-batches
        batches = minibatch(train_example, size=a_batch_size)

        for batch in batches:
            # Update the model with the optimizer for all components
            # We pass 'drop' as a regularization parameter and use the optimizer
            nlp.update(batch, drop=a_dropnum, losses=losses, sgd=optimizer)
        
        print(f"Epoch {epoch + 1}, Losses: {losses}")

# This function sweeps the model.   
def sweep():
    EPOCH_NUM = 7
    BATCH_SIZE = int( len(TRAINING_DATA) /EPOCH_NUM )
    DROP_NUMS = [0.01, 0.02, 0.03, 0.04]
    LEARN_RATES = [0.001, 0.002, 0.003, 0.004]

    data = []
    for drop in DROP_NUMS:
        for rate in LEARN_RATES:
            
            # Train the model
            train(training_examples, BATCH_SIZE, EPOCH_NUM, drop, rate)
            
            # Scores and stores the result
            dev_examples = make_model_data(DEV_DATA)
            scorer = Scorer()
            scores = scorer.score_cats(dev_examples, "cats", labels=["POSITIVE", "NEGATIVE"], multi_label=False)
            accuracy = scores['cats_score']
            datum = {"accuracy": accuracy, "lr": rate, "drop": drop}
            data.append(datum)

    file_name = "best_scores.pkl" 
    with open(file_name, "wb") as file:  # Open the file in write-binary mode
        pickle.dump(data, file)

sweep()
