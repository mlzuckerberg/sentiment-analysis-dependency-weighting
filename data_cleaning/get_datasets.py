# File: get_datasets.py
# Description: Creates the training, dev, and test sets.
# Author: Jacob Tinkelman & Michelle Zuckerberg
# Date: December 17, 2024

from collections import defaultdict
import re
import random
import os

# Lists to hold positive and negative reviews
positive = []
negative = []

# This function splits the data into positive and negative lists based on their sentiment label.
def split_og(file_path: str) -> tuple[list[str], list[str]]:
    negative_lines = []
    positive_lines = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line:
                # Split the line into text and label
                line = line.lower()
                line = line.strip()
                parts = line.split('\t')

                # Extract text and label
                text, label = parts[0], parts[1]
                
                # Categorize based on label
                if label == '0':
                    negative_lines.append(line)
                elif label == '1':
                    positive_lines.append(line)
    
    return (negative_lines, positive_lines)


# This function splits the data into positive and negative lists based on their sentiment label.
data = split_og("kaggle_og.txt")
positive = data[0]
negative = data[1]
print(positive[0])


# This function checks for duplicates in a list.
def check_duplicates(a_list):
    my_dict = defaultdict(list)
    duplicates = []
    
    for idx, line in enumerate(a_list):
        my_dict[line].append(idx)
    
    for key, value in my_dict.items():
        if len(value) > 1:
            duplicates.append((key, value))
    
    return duplicates



#   This function selects random indices from a list.
def random_split(a_list, percentage):
    if not 0 < percentage < 1:
        raise ValueError("Percentage must be between 0 and 1.")
    
    sublist_size = int(len(a_list) * percentage)
    chosen_indices = set(random.sample(range(len(a_list)), sublist_size))
    
    sub_list = [a_list[i] for i in chosen_indices]
    remaining_list = [a_list[i] for i in range(len(a_list)) if i not in chosen_indices]
    
    return sub_list, remaining_list

# This function creates the training, dev, and test sets.
def make_sets(dev_percent, test_percent):
    if not 0 < dev_percent + test_percent < 1:
        raise ValueError("Dev and test percentages must sum to less than 1.")
    
    # Split data into development and remaining sets
    pos_dev, pos_remaining = random_split(positive, dev_percent)
    neg_dev, neg_remaining = random_split(negative, dev_percent)
    
    # Create development set
    dev_set = pos_dev + neg_dev
    
    # Split remaining data into test and training sets
    pos_test, pos_train = random_split(pos_remaining, test_percent / (1 - dev_percent))
    neg_test, neg_train = random_split(neg_remaining, test_percent / (1 - dev_percent))
    
    # Create test and training sets
    test_set = pos_test + neg_test
    training_set = pos_train + neg_train
    
    # Shuffle to ensure balanced representation
    random.shuffle(dev_set)
    random.shuffle(test_set)
    random.shuffle(training_set)
    
    return training_set, dev_set, test_set

# Generate the sets
training_set, dev_set, test_set = make_sets(0.15, 0.15)

print(f"Training set size: {len(training_set)}")
print(f"Development set size: {len(dev_set)}")
print(f"Test set size: {len(test_set)}")
 
set_names = ("dev_set", "training_set", "test_set")

for a_name in set_names:
    file_path = a_name +".txt"
    if not os.path.exists(file_path ):

        with open(file_path, 'w') as file:
            match a_name:
                case "test_set":
                    data = test_set
                case "training_set":
                    data = training_set
                case "dev_set":
                    data = dev_set
            for datum in data:
                file.write(datum + "\n")
        print(f"File '{file_path}' created and list written successfully!")
    else:
        print(f"File '{file_path}' already exists. Consider appending or deleting it first.")