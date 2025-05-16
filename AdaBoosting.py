#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


# Step 1: Create a noisy, complex dataset
X, y = make_classification(n_samples=1000, n_features=20,
                           n_informative=5, n_redundant=5,
                           flip_y=0.1, random_state=42)


# In[3]:


# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[4]:


# Step 3: Train a single decision tree (shallow)
single_tree = DecisionTreeClassifier(max_depth=1, random_state=42)
single_tree.fit(X_train, y_train)
y_pred_single = single_tree.predict(X_test)
acc_single = accuracy_score(y_test, y_pred_single)


# In[5]:


# Step 4: Train AdaBoost using the same weak learner
boosting_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
boosting_model.fit(X_train, y_train)
y_pred_boost = boosting_model.predict(X_test)
acc_boost = accuracy_score(y_test, y_pred_boost)


# In[6]:


# Step 5: Print the results
print("🔍 Accuracy Comparison:")
print(f"Single Decision Tree Accuracy: {acc_single:.2f}")
print(f"AdaBoost Accuracy (with 50 trees): {acc_boost:.2f}")


# In[ ]:




