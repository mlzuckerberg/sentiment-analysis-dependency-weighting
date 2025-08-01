# Sentiment Analysis with Dependency-Weighted Word Vectors

## Authors
- Jacob Tinkelman
- Michelle Zuckerberg

## Overview
A novel approach to sentiment analysis that weights word vectors based on their dependency relationships within sentences. The hypothesis is that verbs, as the core of sentences, might carry more sentiment weight (e.g., "love" vs "hate" being more significant than variations in their objects).

## Scope 
The scope of this project was to see whether weighting word vectors by a scalar based upon their depdency relationship within the sentences of the training data might be useful for text categorization of positive and negative sentiment. The linguistic theory behind this is that it is often said that verbs are the core of the sentence and the rest of the sentence is built around them. As such, it might be reasonable to think that weighting the verb as more important will increase the accuracy of the model as compared to a Bag of Words approach. EG "I love you"  and "I hate you" are almost entirely different but "I hate this" and "I hate that" are rather similar.

### Key Points
- **What's the deliverable?**: A model in which you can initially weight word vectors on the traing-data to artificially raise or lower the magnitude of certain word vectors and hence their importance in the model. Along with thise a nearly identical model that doesn't weight the vectors to use as a neutral baseline.
- **Boundaries**: We do not deal much beyond one sentneces reviews and if this weighting model might be more useful for non theme based classification such as identifying a literary genre given a text. We also did not take a look beyond binary labels such positive, negative, neutral classification. Nor did we implement multi-label where a text can be labeled as more than one category. This is because something cannot be both positive and negative.
- **Pipeline** Imagine you are a peice of training data. The first thing that happens to you is that you are read into a tuple that is in a list and your annotation of 1 or 0 is turned into a dictionary format. Then you are processed into a doc a datatype. Here you split into a goldlabel version and non-gold label version. These two versions are then put back together into an example datatype. You are then fed to the train function as part a batch. 

## Requirements
### Python Dependencies
- Python 3.8+
- spaCy 3.0+
- NumPy
- Pandas
- Matplotlib
- Pickle

### Data Files
- kaggle_og.txt: Original dataset file from Kaggle
- training_set.txt: Training data
- dev_set.txt: Development/validation data
- test_set.txt: Test data

## Technical Approach
 We decided to add a textcat pipeline to spacy which classifies an input based on the number of categories/labels you give to pipeline and annotations you add to your docs(a doc is Spacey datatype). We then need to prepare our annotated docs to be fed into the model. We do this by creating an Example datatype which holds one doc with the gold label and one with the initial training label. The model then trains on the Examples using the update function within the train function of our code. 

## Execution and How to Run

### Running the Models
1. To evaluate model performance:
   - Run `weighted_model.py` for the dependency-weighted model
   - Run `baseline_model.py` for the baseline model

### Analyzing Results
1. For weighted model results:
   - Check the generated CSV file for performance across hyperparameters
   - Use `weighted_model/read_score.py` for detailed analysis and visualizations

2. For baseline model results:
   - Use `baseline_model/read_score.py` for analysis

### Notes
- Some hyperparameters are fixed to manage computational overhead
- Model states are saved as `.pkl` files for efficiency
  - To disable pickle file usage, comment out the relevant lines in the code
  - Look for lines containing `pickle.dump()` or `pickle.load()`


## File Structure and Functions

### Data Cleaning Directory (`/data_cleaning/`)
- **get_datasets.py**
  - Creates training, development, and test datasets from raw Kaggle data
  - Implements random splitting with configurable percentages (default 70/15/15)
  - Handles duplicate removal and data validation
  - Key functions:
    - `split_og()`: Splits data into positive/negative sentiment lists
    - `check_duplicates()`: Validates data for duplicates
    - `random_split()`: Performs random dataset splitting
    - `make_sets()`: Creates final training/dev/test sets

- **format_data.py**
  - Processes raw Kaggle data into model-ready format
  - Converts text and labels into required dictionary structure
  - Key functions:
    - `process_kaggle()`: Converts raw data into (text, sentiment_dict) tuples

- **kaggle_og.txt**
  - Raw dataset containing restaurant reviews
  - Each line contains:
    - Review text
    - Binary sentiment label (1 for positive, 0 for negative)
  - Used as source data for training/dev/test set creation

### Model Directories
#### Baseline Model (`/baseline_model/`)
- **baseline_model.py**
  - Implements standard spaCy text categorization without dependency weighting
  - Provides baseline performance metrics for comparison
  - Key functions:
    - `process_kaggle()`: Formats data for model input
    - `train()`: Trains the baseline model
    - `evaluate()`: Tests model performance
- **read_score.py**
  - Analyzes baseline model performance
  - Creates visualizations and statistics for baseline results

#### Weighted Model (`/weighted_model/`)
- **weighted_model.py**
  - Implements text categorization with dependency-weighted word vectors
  - Applies different weights based on dependency relationships
  - Key functions:
    - `process_kaggle()`: Formats and weights input data
    - `train()`: Trains the weighted model
    - `evaluate()`: Tests model performance with different weight configurations
- **read_score.py**
  - Analyzes and visualizes weighted model performance
  - Loads evaluation data from pickle files
  - Creates comparative performance visualizations
  - Key functions:
    - `process_data()`: Converts results to pandas DataFrame
    - `average_scalar_accuracy()`: Calculates average accuracy by weight scalar
    - `plot()`: Visualizes performance metrics across different weights

Both directories use the same split datasets:
- **training_set.txt**: Main training data
- **dev_set.txt**: Development/validation data
- **test_set.txt**: Test data for final evaluation
