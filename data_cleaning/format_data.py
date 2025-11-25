# File: format_data.py
# Description: Processes the Kaggle data and formats it for the model.
# Author: Jacob Tinkelman & Michelle Zuckerberg
# Date: December 17, 2024   

import re
from typing import List, Tuple

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
    
process_kaggle('kaggle_og.txt')