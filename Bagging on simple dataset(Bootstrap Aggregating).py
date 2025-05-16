#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


# Step 1: Load data
iris = load_iris()
X, y = iris.data, iris.target


# In[3]:


# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[4]:


# Step 3: Create a single base model (e.g., decision tree)
base_model = DecisionTreeClassifier()


# In[7]:


# Step 4: Create a Bagging model (use `estimator` instead of `base_estimator`)
bagging_model = BaggingClassifier(
    estimator=base_model,       # model to use
    n_estimators=10,            # number of models to train
    max_samples=0.8,            # 80% data per model
    bootstrap=True,             # sample with replacement
    random_state=42
)


# In[8]:


# Step 5: Train the Bagging model
bagging_model.fit(X_train, y_train)


# In[9]:


# Step 6: Predict and evaluate
y_pred = bagging_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# In[10]:


print(f"Accuracy of Bagging Model: {accuracy:.2f}")


#  What's Happening Behind the Scenes?
# BaggingClassifier trains 10 different decision trees.
# 
# Each tree is trained on a random 80% subset of the training data.
# 
# For each test point, it takes the majority vote from all 10 trees.
# 
# This reduces overfitting compared to just one tree.

# In[11]:


single_tree = DecisionTreeClassifier()
single_tree.fit(X_train, y_train)
y_pred_single = single_tree.predict(X_test)
single_accuracy = accuracy_score(y_test, y_pred_single)

print(f"Accuracy of Single Tree: {single_accuracy:.2f}")


#  Why Did Bagging and Single Tree Give the Same Accuracy?
# You're using the Iris dataset, which is:
# 
# Very clean, small, and well-separated.
# 
# A simple classification task — even a single decision tree can already perform near-perfectly.
# 
# So in this case, bagging doesn’t really improve performance because there's not much variance or noise to reduce.
# 
# 

# In[ ]:




