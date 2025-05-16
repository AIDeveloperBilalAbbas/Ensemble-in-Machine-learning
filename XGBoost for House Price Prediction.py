#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1. Import necessary libraries
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


# In[2]:


# 2. Create synthetic regression data (simulating house price)
X, y = make_regression(n_samples=1000, n_features=10, noise=20, random_state=42)


# In[3]:


# 2. Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[4]:


# 3. Train a single decision tree (for comparison)
single_tree = DecisionTreeRegressor(max_depth=3, random_state=42)
single_tree.fit(X_train, y_train)
pred_tree = single_tree.predict(X_test)
mse_tree = mean_squared_error(y_test, pred_tree)


# In[5]:


# 4. Train an XGBoost model
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb.fit(X_train, y_train)
pred_xgb = xgb.predict(X_test)
mse_xgb = mean_squared_error(y_test, pred_xgb)


# In[6]:


# 5. Compare performance
print("📊 Mean Squared Error Comparison:")
print(f"Single Decision Tree MSE: {mse_tree:.2f}")
print(f"XGBoost MSE:              {mse_xgb:.2f}")


# In[ ]:


get_ipython().system('jupyter nbconvert --to script "your_notebook_name.ipynb"')

