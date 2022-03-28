import numpy as np
import pandas as pd
import random as rn


# structure of decision tree
class Node:

    def __init__(self, question=None, yes_answer=None, no_answer=None):
        self.question = question
        self.yes_answer = yes_answer
        self.no_answer = no_answer

# Function to split data into specified test size / training size
# using .values converts data from a pandas df to a numpy array which speeds up runtime
def train_test_split(df, test_size):
    indices = df.index.tolist()
    test_indices = rn.sample(indices, round(test_size * len(df)))

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)

    return train_df, test_df


# check if labels are all same for classification
def isPure(data):
    identical = np.all(data == data[0, :], axis=0)
    if identical[0] and identical[1]:
        return True
    else:
        labels = data[:, -1]
        unique_classes = np.unique(labels)

        if len(unique_classes) == 1:
            return True
        else:
            return False


# chooses the most frequent label among the data for classification
def classify(data):
    labels = data[:, -1]
    unique_classes, counts = np.unique(labels, return_counts=True)

    most_frequent = counts.argmax()
    classification = unique_classes[most_frequent]

    return classification


# identifies possible splits in data. Will be used to determine resulting entropy of each split
def get_splits(data):
    # dictionary to keep track of each columns(features) potential splits
    potential_splits = {}

    # data.shape returns tuple (# of rows, # of col), used to iterate through columns
    _, num_columns = data.shape
    for column in range(num_columns - 1):
        potential_splits[column] = []
        values = data[:, column]
        unique_values = np.unique(values)

        # for each pair of unique values, find its split and append it to dictionary
        for i in range(len(unique_values)):
            if i != 0:
                cur = unique_values[i]
                prev = unique_values[i - 1]
                split = (cur + prev) / 2

                potential_splits[column].append(split)

    return potential_splits


# splits data during decision tree traversal
# parameters: data to be split,
#             column (feature) were splitting on,
#             value at which that feature is being split
# returns data that is above the split and data that is below the split
def perform_split(data, column, value):
    # selects values from the column to be split
    split_values = data[:, column]

    # splits values
    data_above = data[split_values > value]
    data_below = data[split_values <= value]

    return data_below, data_above


# given dataset, count labels and determine entropy of that set of labels
def entropy(data):
    labels = data[:, -1]
    _, freq = np.unique(labels, return_counts=True)

    probabilities = freq / freq.sum()
    ent = sum(probabilities * -np.log2(probabilities))

    return ent


# determine entropy of split values which will help determine which splits
# cause the least amount of entropy. Used for finding best splits.
def total_entropy(data_below, data_above):
    total_num_values = len(data_below) + len(data_above)
    probability_below = len(data_below) / total_num_values
    probability_above = len(data_above) / total_num_values

    total_ent = (probability_below * entropy(data_below)
                 + probability_above * entropy(data_above))

    return total_ent


# Use total entropy to determine which splits result in lowest entropy
def find_best_split(data, potential_splits):
    # initialize entropy to max
    overall_entropy = np.inf

    # iterate through the potential splits dictionary, each key is a feature column
    # then for each value in column, determine if splitting there will result in lower entropy
    for column in potential_splits:
        for value in potential_splits[column]:
            below, above = perform_split(data, column, value)
            cur_entropy = total_entropy(below, above)

            if cur_entropy <= overall_entropy:
                overall_entropy = cur_entropy
                best_split = column
                best_value = value

    return best_split, best_value


def build_tree(data, df):
    # base case, if data already only consists of one label
    if isPure(data):
        classification = classify(data)
        return classification

    else:

        # find splits, find best split, peform split
        possible_splits = get_splits(data)
        split, split_value = find_best_split(data, possible_splits)
        below, above = perform_split(data, split, split_value)

        # declare root node that holds the question and children nodes (yes or no answers)
        question = "{} <= {}".format(df.columns[split], split_value)
        root = Node(question=question, yes_answer=None, no_answer=None)

        # recursively build tree by deciding new splits and appending nodes to root
        yes = build_tree(below, df)
        no = build_tree(above, df)

        # if splitting the tree results in the same classification, then there is no need to split
        # just pick one. Otherwise continue with assigning split
        if yes == no:
            root = yes
        else:
            root.yes_answer = yes
            root.no_answer = no

        return root


def classify_single(sample, tree):
    # get the question the tree's current node is asking
    question = tree.question

    # question is a string, so get that strings individual
    # parts to actually perform asking the question
    attribute, operator, value = question.split()

    # record if the sample were testing is a yes or a no answer
    if sample[attribute] <= float(value):
        answer = tree.yes_answer
    else:
        answer = tree.no_answer

    # if answer is a string, this must be a label and beer has been classified return that classification
    # else answer is another node, so we recursively call the function on the new node of the tree
    if isinstance(answer, str):
        return answer
    else:
        return classify_single(sample, answer)


def determine_accuracy(data_df, tree):
    data_df["classifications"] = data_df.apply(classify_single, axis=1, args=(tree,))
    data_df["correct"] = data_df.classifications == data_df.label

    accuracy = round(data_df.correct.mean() * 100)

    return accuracy