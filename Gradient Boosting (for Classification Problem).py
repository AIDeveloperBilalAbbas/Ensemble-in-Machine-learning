#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


# Step 1: Create a dummy binary classification dataset
X, y = make_classification(n_samples=1000, n_features=10,
                           n_informative=5, n_redundant=2,
                           n_classes=2, flip_y=0.1, random_state=42)


# In[3]:


# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)


# In[4]:


# Step 3: Train a single decision tree
single_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
single_tree.fit(X_train, y_train)
pred_tree = single_tree.predict(X_test)
acc_tree = accuracy_score(y_test, pred_tree)


# In[6]:


# Step 4: Train a Gradient Boosting Classifier
gboost = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gboost.fit(X_train, y_train)
pred_boost = gboost.predict(X_test)
acc_boost = accuracy_score(y_test, pred_boost)


# In[7]:


# Step 5: Compare Accuracy
print("📊 Accuracy Comparison:")
print(f"Single Decision Tree Accuracy:  {acc_tree:.2f}")
print(f"Gradient Boosting Accuracy:     {acc_boost:.2f}")


# In[ ]:




