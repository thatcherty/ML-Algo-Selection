# Overview
Classify datasets by the recommended machine learning algorithm for model training. Provide results that discuss several important metrics and the expected performance of the selected model.

# Datasets
Data sets will be selected from the [UCI Machine Learning Repo](https://archive.ics.uci.edu/datasets)

# Approach
2 possible paths:
 - Select 3 different machine learning algorithms to classify datasets into
 - Dive into ANN and change the classification to the recommended layers and nodes to use for training

## Selecting Data
 - Choose several datasets, the more the better
 - Extract the relevant features:
   - Size
   - Distribution (left-skewed, normal, right-skewed)
   - Feature count
   - Etc.

## Creating Training/Testing Data
- Train models using each of the different algorithms on each dataset
- Report relevant metrics:
  - Precision
  - F1
  - Accuracy
  - Etc.
- Report ideal algorithm
  - We will need to be specific in the criteria
  - Consider recommending models based on the goal (high performance, high accuracy, etc.)
 
## Training the final model
 - The data from above will be used to train the final model to recommend an algorithm based on data characteristics
 - It should output the recommended model, along with the expected metrics that can be achieved with that model
