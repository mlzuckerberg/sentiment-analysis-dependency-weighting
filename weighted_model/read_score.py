# File: read_score.py
# Description: Reads the scores from the model and plots them.
# Author: Jacob Tinkelman & Michelle Zuckerberg
# Date: December 17, 2024

import pickle
import pandas as pd
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# Loads the data from the pickle file.
with open("best_scores.pkl", "rb") as file:  # Open the file in read-binary mode
    loaded_data = pickle.load(file)
print(loaded_data)

# This function converts a list of dictionaries into a pandas DataFrame, sorts it by accuracy, and saves the DataFrame to a CSV file.
def process_data(data, file_name="sorted_data.csv"):
    # Convert list of dictionaries to a DataFrame
    df = pd.DataFrame(data)
    
    # Sort the DataFrame by 'accuracy' in descending order
    df_sorted = df.sort_values(by="scalar", ascending=False)
    
    # Save the sorted DataFrame to a CSV file
    df_sorted.to_csv(file_name, index=False)
    
    return df_sorted

sorted_df = process_data(loaded_data)


# This function averages the accuracy for each scalar value.
def average_scalar_accuracy():
    the_sums = defaultdict(float) 
    occurences = Counter()

    for datum in loaded_data:
        the_scalar = datum["scalar"]
        the_sums[the_scalar] += datum["accuracy"]
        occurences[the_scalar] +=1

    averages = defaultdict(float)
    
    for key in the_sums.keys():
        averages[key] = the_sums[key] / occurences[key]

    return averages

averages = average_scalar_accuracy()


# This function plots the accuracy for each scalar value.
def plot():
    xvalues, yvalues = [], []
    for scalar, accuracy in averages.items():
        xvalues.append(scalar)
        yvalues.append(accuracy)
    plt.plot(xvalues, yvalues, label="average accuracy by scalar")
    plt.xlabel("scalar")  # Add X-axis label
    plt.ylabel("accuracy")  # Add Y-axis label
    plt.title("average accuracy by scalar")  # Add a title
    plt.legend()  # Add a legend
    plt.grid(True)  # Add gridlines
    plt.show()  # Display the plot

plot()