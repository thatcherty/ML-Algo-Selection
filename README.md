# Overview
Classify datasets by the recommended machine learning algorithm for model training. Provide results that discuss several important metrics and the expected performance of the selected model.

# Intro
 - Short introductions
 - What interested you about this project?
 - Is there anything specific you hope to get out of it?
 - What kind of experience do you have with Machine Learning?
 - General availability, spring break availability?


# Datasets
Data sets will be selected from the [UCI Machine Learning Repo](https://archive.ics.uci.edu/datasets)

# Approach
2 possible paths:
 - Select 3 different machine learning algorithms to classify datasets into
   - Linear Regression
   - Logistic Regression
   - A third option (possibly ANN, this may be too complex in setup)
 - Dive into ANN and change the classification to the recommended layers and nodes to use for training

Other recommendations are welcome.

## Selecting Data
 - Choose several datasets, the more the better
   - Ideally, selecting benchmark datasets
   - at least 10 to start
 - Extract the relevant features:
   - Size
   - Distribution (left-skewed, normal, right-skewed)
   - Feature count
   - Type of data (categorical, continuous, etc.)
   - Etc.

The data extractions will be as automated as possible with existing or custom functions.

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

# Work split
## Data Collection
 - We will all use the same datasets
 - We will use the same functions to do the following:
   - Extract the features (Size of dataset, distribution, feature count)
   - Insert the feature data into the new dataset
   - Report the metrics
   - Insert the metrics into the new dataset
   - Report the final recommendation
   - Insert the final recommendation into the new dataset
 - We will be responsible for collecting the relevant features, metrics, and the final ML recommendation for each dataset within our specific algorithm
 - We will each be responsible for developing the program that trains the model on the datasets for our specific algorithm
   - The approach to train the model should follow our standards from class

## Final Model Development
 - I am under the assumption that we should use ANN
 - We can approach this collaboratively
   - I would like to have a user interface to enter data characteristics
   - If possible, we could automate the feature extraction from the dataset a user provides, rather than request them to identify the features
   - We will need a model to be trained, of course, and may require more data
- These lead to a few roles:
  - Design the UI
    - Validate data available
  - Train the model
    - Validate UI
  - Collect additional data
    - Validate model training  

## Presentation
 - We can likely create slides about each of our model training experiences
 - We can collaborate on presenting the final results

# Collaboration
 - I appreciate code review. If you have any input on the programs I write, let me know. I will plan to do the same.
 - I would like to work within GitHub mainly. How comfortable are you with this?
 - Tasks and milestones will be tracked in the project associated with this repository. We will be assigned to our responsibilities, and each should be given an expected end date and associated with a milestone.
 - There is a discussion tab (after I add collaborators) where we can put ideas and thoughts related to the project.
 - If you find something is broken, raise an issue.
 - We will use Python mainly, but the UI can be in any language that can integrate with the model we develop

# Where to train models
- What are your set-ups like? Where do you plan to train models?
  - I am considering Google Colab, as students, we get a free subscription and can get access to better resources
  - GitHub also allows Codespaces for free for Pro members, which is provided to you as a student. You can install a Colab extension and work on all your training in VS Code with Jupyter notebooks, running your training on the Google Colab GPU
 
# Timeline
 - More granualar tasks are still in the works
 - High-level deadline on 4/29/2026 (project presentation)
   - 2 months to complete
   - Data collection completed by the end of March
   - Final model training by the 3rd week of April
   - Presentation and final reviews by 4/28/2026
 





