#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1. Install CatBoost if not installed
# Run this in terminal or Jupyter if needed: !pip install catboost

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor


# In[2]:


# 2. Generate synthetic dataset (simulating house features)
X, y = make_regression(n_samples=1000, n_features=10, noise=20, random_state=42)


# In[3]:


# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[4]:


# 4. Train Single Decision Tree
tree = DecisionTreeRegressor(max_depth=4, random_state=42)
tree.fit(X_train, y_train)
tree_preds = tree.predict(X_test)
tree_mse = mean_squared_error(y_test, tree_preds)


# In[5]:


# 5. Train CatBoost Regressor
cat_model = CatBoostRegressor(
    iterations=100,          # number of boosting rounds
    learning_rate=0.1,       # how fast to learn
    depth=4,                 # max tree depth
    random_seed=42,
    verbose=0                # suppress training output
)
cat_model.fit(X_train, y_train)
cat_preds = cat_model.predict(X_test)
cat_mse = mean_squared_error(y_test, cat_preds)


# In[6]:


# 6. Compare Results
print("📊 Mean Squared Error Comparison:")
print(f"Single Decision Tree MSE: {tree_mse:.2f}")
print(f"CatBoost Regressor MSE:   {cat_mse:.2f}")


# In[ ]:




