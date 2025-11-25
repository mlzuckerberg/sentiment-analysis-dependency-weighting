import spacy
from spacy.training import Example
from typing import Iterable, List, Tuple
import numpy as np
from spacy.scorer import Scorer
from spacy.util import minibatch
import re
import random
import pickle

seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)

# USEFUL LINKS
# https://spacy.io/api/doc
# https://spacy.io/api/example
# https://spacy.io/api/textcategorizer

# Loads the spacy model
nlp = spacy.load("en_core_web_md") 


# This function processes the kaggle data and returns a list of tuples. Each tuple contains a processed sentence and a sentiment dictionary.
def process_kaggle(file_path: str) -> List[Tuple[str, dict[str, dict[str, float]]]]:
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


TRAINING_PATH = 'training_set.txt'
DEV_PATH = "dev_set.txt"
TEST_PATH = "test_set.txt"

TRAINING_DATA = process_kaggle(TRAINING_PATH)
DEV_DATA = process_kaggle(DEV_PATH)
TEST_DATA = process_kaggle(TEST_PATH)

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


# Extract the dependency layers from the sentences
training_examples = [] 

# This function extracts the dependency layers from the sentences
def extract_dependency_layers(sentences: str) -> List[List[List[str]]]:
    # Process the input string to split it into multiple sentences
    doc = nlp(sentences)
    
    # Initialize the list to store the dependency layers for all sentences
    all_layers = []
    
    # Process each sentence separately
    for sent in doc.sents:
        # Find the main verb (root) of the sentence
        main_verb = None
        for token in sent:
            if token.dep_ == "ROOT":
                main_verb = token
                break
        
        if main_verb is None:
            all_layers.append([])  # If no main verb, append empty list
            continue
        
        # Initialize layers dictionary
        layers: dict[int, set[str]] = {0: {main_verb.text}}
        
        # Track visited tokens to prevent infinite loops
        visited = set()
        
        def trace_dependency_layers(current_token, current_depth=0):
            # Mark current token as visited
            visited.add(current_token)
            
            # Explore children
            for child in current_token.children:
                # Skip if already visited
                if child in visited:
                    continue
                
                # Add to appropriate layer
                depth = current_depth + 1
                if depth not in layers:
                    layers[depth] = set()
                layers[depth].add(child.text)
                
                # Recursively explore this child's dependencies
                trace_dependency_layers(child, depth)
            
            # Explore parent if not already at root
            if current_token.head != current_token and current_token.head not in visited:
                depth = current_depth + 1
                if depth not in layers:
                    layers[depth] = set()
                layers[depth].add(current_token.head.text)
                trace_dependency_layers(current_token.head, depth)
        
        # Start tracing from main verb
        trace_dependency_layers(main_verb)
        
        # Convert to sorted list of layers, converting sets to lists
        max_layer = max(layers.keys()) if layers else 0
        result = [list(layers.get(i, set())) for i in range(max_layer + 1)]
        
        # Append the result for the current sentence
        all_layers.append(result)
    
    return all_layers


# This function weights the sentences.
def weight_sentences(datalayer, k):
    # Loop through each layer in the datalayer
    for data in datalayer:
        currk = k
        for layer in data:
            #print(layer)  # Print the current layer
            iter =1
            for word in layer:
                #print(word)  # Print the current word
                try:
                    # Get the lexeme for the word by indexing the vocab
                    lexeme = nlp.vocab[word]
                    
                    # Check if lexeme exists and has a vector
                    if lexeme is not None and lexeme.has_vector:
                        #print(f"Processing word: {word}")  # Log the word being processed
                        lexeme.vector *= k  # Use 'k' to scale the vector
                    else:
                        print(f"Skipping word '{word}' - not in vocabulary or has no vector.")
                
                except KeyError:
                    # Handle case where the word is not in the vocabulary
                    print(f"Word '{word}' not found in vocabulary.")
                
                except Exception as e:
                    print(f"Error processing word '{word}': {e}")
        iter+=1
        currk = currk/iter


# This function makes the model data.
def make_model_data(data: tuple[str, dict[str, dict[str, float]]]) -> Iterable[Example]:
    example_list = [] #the list of examples for the model
    for text, annotations in data:

        doc_gold = nlp(text)
        doc_guess = nlp(text)

        doc_gold.cats = annotations["cats"]

        an_example = Example(doc_guess, doc_gold)
        example_list.append(an_example)

    return example_list


# This function trains the model.
def train(train_example, a_batch_size, an_epoch, a_dropnum, a_learnrate):
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
    
    scalars = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3,1.5,2]
    all_score = []

    for c in scalars:
        # Goes through and scales the wordvectors
        print("processing words")
        for text, label in TRAINING_DATA:
            datalayer = extract_dependency_layers(text)
            weight_sentences(datalayer, c)
        print("words processed")
        
        # Loads our data into the model
        training_examples = make_model_data(TRAINING_DATA) # Stores sentiment in doc.cat attribute

        
        if not nlp.has_pipe("textcat"):
            textcat = nlp.add_pipe("textcat")  # Sets up the classifier component
            textcat.add_label("POSITIVE")
            textcat.add_label("NEGATIVE")
    

        EPOCH_NUM = 7
        BATCH_SIZE = int( len(TRAINING_DATA) /EPOCH_NUM )
        DROP_NUMS = [0.01, 0.02, 0.03, 0.04]
        LEARN_RATES = [0.001, 0.002, 0.003, 0.004]

        for rate in LEARN_RATES:
            for drop in DROP_NUMS:
                train(training_examples, BATCH_SIZE, EPOCH_NUM, drop , rate)

                # Scores the model
                dev_examples = make_model_data(DEV_DATA)
                scorer = Scorer()
                scores = scorer.score_cats(dev_examples, "cats", labels=["POSITIVE", "NEGATIVE"], multi_label=False)
                accuracy = scores['cats_score']
                
                all_score.append( {"scalar": c,"lr": rate, "drop": drop, "accuracy": accuracy} )
      
    file_name = "best_scores.pkl" 
    with open(file_name, "wb") as file:  # Open the file in write-binary mode
        data = all_score
        pickle.dump(data, file)

sweep()