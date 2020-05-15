#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random
from math import log
from collections import Counter
from sklearn.tree import DecisionTreeClassifier


# In[2]:


from sklearn.datasets import load_breast_cancer

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

cancer_data = load_breast_cancer()
data = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
data['diagnosis'] = cancer_data.target

features = data.columns[:-1]
label = 'diagnosis'


# In[3]:


#Counter(data[label])


# In[4]:


def gini(rows):
    group_labels = [row[-1] for row in rows]
    probs = {label: count/len(group_labels) for label, count in Counter(group_labels).items()}
    gini = 1 - sum([prob*prob for label, prob in probs.items()])
    return(gini)

def information_gain(i_parent, child1, child2):
    n1 = len(child1)
    n2 = len(child2)
    p = n1 / (n1 + n2)
    ig = i_parent - p*gini(child1) - (1-p)*gini(child2)
    return(ig)


# In[5]:


class Question:
    def __init__(self, split_column, split_position):
        self.split_column = split_column
        self.split_position = split_position
    
    def match(self, example):
        #print("example:", example)
        val = example[self.split_column]
        #print("value:", val)
        if isinstance(val, int) or isinstance(val, float):
            return val >= self.split_position
        else:
            return val == self.split_position

    def __repr__(self):
         # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if  isinstance(self.split_position, int) or isinstance(self.split_position, float):
            condition = ">="
        return "Is %s %s %s?" % (features[self.split_column], condition, str(self.split_position))


# In[6]:


def split(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return(true_rows, false_rows)


# In[7]:


def find_best_split_position(rows, split_column):
    
    i_parent = gini(rows)
    best_ig = 0
    
    values = sorted([row[split_column] for row in rows])
    best_split_position = None
    
    for i in range(len(values)-1):
        #print(values[i], values[i+1])
        split_position = (values[i] + values[i+1]) / 2
        question = Question(split_column, split_position)
        true_rows, false_rows = split(rows, question)
        
        if len(true_rows) == 0 or len(false_rows) == 0:
                continue
                
        ig = information_gain(i_parent, true_rows, false_rows)
        if ig > best_ig:
            best_ig = ig
            best_split_position = split_position
    
    best_question = Question(split_column, best_split_position)
    return(best_ig, best_question)


# In[8]:

#
#random.seed(42)
#training_data = data.loc[random.sample(list(range((len(data)-1))), 100), :]
#training_data = training_data.values.tolist()
#best_ig, best_question = find_best_split_position(training_data, 0)
#print(best_ig, best_question)


# In[9]:


def find_best_split(rows):
    best_ig = 0
    best_question = None
    for col in range(len(features)):
        ig, question = find_best_split_position(rows, col)
        if ig > best_ig:
            best_ig = ig
            best_question = question
    return(best_ig, best_question)


# In[10]:


#best_ig, best_question = find_best_split(training_data)
#print(best_ig, best_question)


# In[11]:


class Decision_Node:
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

class Leaf:
    def __init__(self,
                 rows):
        counts = Counter([row[-1] for row in rows])
        self.label = max(zip(counts.keys(), counts.values()))[0]


# In[12]:


def build_tree(rows, max_depth, min_samples_split, min_impurity_decrease, depth):
    #print(depth)
    samples_split = len(rows)
    ig, question = find_best_split(rows)

    if depth >= max_depth or samples_split <= min_samples_split or ig <= min_impurity_decrease:
        return(Leaf(rows))

    true_rows, false_rows = split(rows, question)
    true_branch = build_tree(true_rows, max_depth, min_samples_split, min_impurity_decrease, depth+1)
    false_branch = build_tree(false_rows, max_depth, min_samples_split, min_impurity_decrease, depth+1)
    
    return Decision_Node(question, true_branch, false_branch)


# In[13]:


def print_tree(node, spacing=" "):
    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.label)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


# In[14]:


def train(training_data, max_depth=999, min_samples_split=2, min_impurity_decrease=0):
    my_tree = build_tree(training_data, max_depth, min_samples_split, min_impurity_decrease, depth=1)
    return(my_tree)

def predict(row, node):
    
     # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return(node.label)
    else:
        # Decide whether to follow the true-branch or the false-branch.
        # Compare the feature / value stored in the node,
        # to the example we're considering.
        if node.question.match(row):
            return predict(row, node.true_branch)
        else:
            return predict(row, node.false_branch)


# In[15]:


#my_tree = train(training_data, max_depth=5)
#print_tree(my_tree)


# In[16]:

#for row in data[0:5].values.tolist():
#    predict(row, my_tree)


# In[17]:


def splitDataset(data, k=5, random_state=42):
    random.seed(random_state)
    
    num_fold = round(len(data)/k)
    indices = list(data.index)
    fold_indices = [0]*k
    
    for i in range(k-1):
        fold_indices[i] = random.sample(list(range(len(indices))), num_fold)
        indices = list(set(indices).difference(set(fold_indices[i])))
    fold_indices[k-1] = indices
    
    return(fold_indices)


# In[18]:


def crossValidation(data, k=5, random_state=42, max_depth=999, min_samples_split=2, min_impurity_decrease=0):
    indices = list(data.index)
    fold_indices = splitDataset(data, k, random_state)
    accuracies = [0]*k
    for i in range(k):
        print("Fold ", i+1, ":")
        test_indices = fold_indices[i]
        train_indices = list(set(indices).difference(set(fold_indices[i])))
        train_data = data.loc[train_indices, :].values.tolist()
        test_data = data.loc[test_indices, :]
        
        my_tree = train(train_data, max_depth, min_samples_split, min_impurity_decrease)
        print_tree(my_tree)
        
        predictions = [predict(row, my_tree) for row in test_data.drop(label, axis=1).values.tolist()]
        
        accuracies[i] = sum(list(map(lambda x, y: x == y, test_data[label], predictions))) / len(test_data)
        print("Accuracy:", accuracies[i])
    
    return (np.mean(accuracies))


# In[19]:


ave_acc = crossValidation(data, k=5, random_state=42, max_depth=5, min_samples_split=2, min_impurity_decrease=0)
print("Mean accuracy of 5-fold cross validation: %f" %ave_acc)

