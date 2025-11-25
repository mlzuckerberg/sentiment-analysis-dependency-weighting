# File: read_score.py
# Description: Reads the scores from the best_scores.pkl file and calculates the average accuracy.
# Author: Jacob Tinkelman & Michelle Zuckerberg
# Date: December 17, 2024

import pickle
from collections import defaultdict, Counter

# Load the data from the pickle file
with open('best_scores.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)

# This function calculates the average accuracy
def average__accuracy(dataset):
    the_sums = 0
    occurences = len(dataset)

    for datum in dataset:
        the_sums += datum["accuracy"]

    return the_sums/occurences


the_average_accuracy = average__accuracy(data)

print(the_average_accuracy)