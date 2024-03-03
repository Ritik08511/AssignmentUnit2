import pandas as pd
from math import log2

# Define the training dataset
data = {
    'Day': [f'D{i}' for i in range(1, 15)],
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Play Tennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Calculate entropy
def calculate_entropy(df, target_column):
    entropy = 0
    total_rows = len(df)

    for label in df[target_column].unique():
        label_count = len(df[df[target_column] == label])
        probability = label_count / total_rows
        entropy -= probability * log2(probability)

    return entropy

# Calculate information gain
def calculate_information_gain(df, attribute, target_column):
    entropy_before_split = calculate_entropy(df, target_column)
    total_rows = len(df)

    unique_values = df[attribute].unique()
    weighted_entropy_after_split = 0

    for value in unique_values:
        subset = df[df[attribute] == value]
        subset_rows = len(subset)
        weight = subset_rows / total_rows
        weighted_entropy_after_split += weight * calculate_entropy(subset, target_column)

    information_gain = entropy_before_split - weighted_entropy_after_split
    return information_gain

# Calculate entropy and gain for each attribute
target_column = 'Play Tennis'
attributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']

for attribute in attributes:
    entropy = calculate_entropy(df, target_column)
    gain = calculate_information_gain(df, attribute, target_column)

    print(f'Attribute: {attribute}')
    print(f'Entropy: {entropy:.4f}')
    print(f'Information Gain: {gain:.4f}\n')
