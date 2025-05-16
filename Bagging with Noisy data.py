#!/usr/bin/env python
# coding: utf-8

# In[12]:


from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[13]:


# Create a noisy, complex dataset
X, y = make_classification(n_samples=1000, n_features=20, 
                           n_informative=5, n_redundant=5, 
                           n_clusters_per_class=2, flip_y=0.1,
                           random_state=42)


# Parameter	What it does
# 
# n_samples=1000	Generates 1000 data points (rows)
# 
# n_features=20	Each data point has 20 input features (columns)
# 
# n_informative=5	5 features carry actual information useful for classification
# 
# n_redundant=5	5 features are random linear combinations of the informative ones
# 
# n_clusters_per_class=2	Each class is made up of 2 subclusters of data
# 
# flip_y=0.1	Adds 10% noise by randomly flipping the class label
# 
# random_state=42	Sets the seed for reproducibility — same data each run
# 
# 
# Real-world analogy:
# Imagine you're creating fake customer data:
# 
# You make 1000 customers (n_samples)
# 
# Each one has 20 attributes like age, salary, habits (n_features)
# 
# But only 5 of those are actually helpful in deciding if they’ll buy your product (n_informative)
# 
# 5 more are just fluff or mixtures of those (n_redundant)
# 
# And some people are labeled incorrectly on purpose (flip_y=0.1) to simulate real-world messiness.

# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[15]:


# Train single decision tree
single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)
single_pred = single_tree.predict(X_test)
single_acc = accuracy_score(y_test, single_pred)


# In[16]:


# Train a bagging model
bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,
    max_samples=0.8,
    bootstrap=True,
    random_state=42
)


# In[17]:


bagging_model.fit(X_train, y_train)
bagging_pred = bagging_model.predict(X_test)
bagging_acc = accuracy_score(y_test, bagging_pred)


# In[18]:


# Print results
print(f"Single Tree Accuracy: {single_acc:.2f}")
print(f"Bagging Accuracy: {bagging_acc:.2f}")


# In[ ]:




