import numpy as np
import pandas as pd
import random as rn

from buildTree import *

# Read in Beer Dataset
df = pd.read_csv("beers.csv")
notBadData = pd.read_csv("iris.csv")
notBadData = notBadData.rename(columns={'variety': 'label'})

# Dataset of only top 5 most frequent labels
occurences = df['label'].value_counts()[:5]
occurences = occurences.index.tolist()
df2 = df[df.label.isin(occurences)]

# ------------------------------------------------------
# Perform splitting

# Original data 20 split and 30 split
train_df, test_df = train_test_split(df, 0.20)
train_df2, test_df2 = train_test_split(df, 0.30)

# Top 10 Labels 20 and 30 split
train_df3, test_df3 = train_test_split(df2, 0.20)
train_df4, test_df4 = train_test_split(df2, 0.30)

# -------------------------------------------------------------
# turn data into type npArray for easier use in functions

# Original 20 and 30 splits

# 20
train_data = train_df.values
test_data = test_df.values
# 30
train_data2 = train_df2.values
test_data2 = test_df2.values

# -----------------------------------------------------------
# Top 10 Labels 20 and 30 splits

# 20
train_data3 = train_df3.values
test_data3 = test_df3.values

# 30
train_data4 = train_df4.values
test_data4 = test_df4.values
# ------------------------------------------------------------

# build tree on training data
tree = build_tree(train_data, train_df)
tree2 = build_tree(train_data2, train_df2)
tree3 = build_tree(train_data3, train_df3)
tree4 = build_tree(train_data4, train_df4)

# ------------------------------------------------------------

# Run classification on all testing data and return accuracy

print("\n-------ORIGINAL DATA SET-------\n")

print("TOTAL SIZE: 1377")
print("TRAIN/TEST SIZE = 80/20")
print("Size of Training Set: " + str(len(train_df.index)))
print("Number of Beers to Classify: " + str(len(test_df.index)))
acc1 = determine_accuracy(test_df, tree)
print("Accuracy of Classification: " + str(acc1))

print("---------------------------------------------")

print("TOTAL SIZE: 1377")
print("TRAIN/TEST SIZE = 70/30")
print("Size of Training Set: " + str(len(train_df2.index)))
print("Number of Beers to Classify: " + str(len(test_df2.index)))
acc2 = determine_accuracy(test_df2, tree2)
print("Accuracy of Classification: " + str(acc2))

print("---------------------------------------------")

print("\n -------TOP 5 MOST FREQUENT LABELS-------\n")

print("TOTAL SIZE: 669")
print("TRAIN/TEST SIZE = 80/20")
print("Size of Training Set: " + str(len(train_df3.index)))
print("Number of Beers to Classify: " + str(len(test_df3.index)))
acc3 = determine_accuracy(test_df3, tree3)
print("Accuracy of Classification: " + str(acc3))

print("---------------------------------------------")

print("TOTAL SIZE: 669")
print("TRAIN/TEST SIZE = 70/30")
print("Size of Training Set: " + str(len(train_df4.index)))
print("Number of Beers to Classify: " + str(len(test_df4.index)))
acc4 = determine_accuracy(test_df4, tree4)
print("Accuracy of Classification: " + str(acc4))

print("---------------------------------------------")

print("\n -------IRIS DATA THAT ISN'T TERRIBLE-------\n")

train_df5, test_df5 = train_test_split(notBadData, 0.25)
train_data5 = train_df5.values
test_data5 = train_df5.values

tree5 = build_tree(train_data5, train_df5)

print("TOTAL SiZE: " + str(len(notBadData.index)))
print("TRAIN/TEST SIZE = 75/25")
print("Size of Training Set: " + str(len(train_df5.index)))
print("Number of Beers to Classify: " + str(len(test_df5.index)))
acc5 = determine_accuracy(test_df5, tree5)
print("Accuracy of Classification: " + str(acc5))