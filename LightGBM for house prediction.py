#!/usr/bin/env python
# coding: utf-8

# In[5]:


from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


# In[6]:


# 2. Generate synthetic dataset (simulating house prices)
X, y = make_regression(
    n_samples=1000,     # 1000 samples
    n_features=10,      # 10 house-related features
    noise=25,           # adds noise to make it realistic
    random_state=42
)


# In[7]:


# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# In[8]:


# 4. Train a single Decision Tree Regressor
tree_model = DecisionTreeRegressor(max_depth=4, random_state=42)
tree_model.fit(X_train, y_train)
tree_preds = tree_model.predict(X_test)
tree_mse = mean_squared_error(y_test, tree_preds)


# In[9]:


# 5. Train a LightGBM Regressor
lgb_model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
lgb_model.fit(X_train, y_train)
lgb_preds = lgb_model.predict(X_test)
lgb_mse = mean_squared_error(y_test, lgb_preds)


# In[10]:


# 6. Compare both models
print("📊 Mean Squared Error (MSE) Comparison:")
print(f"Single Decision Tree MSE: {tree_mse:.2f}")
print(f"LightGBM Regressor MSE:  {lgb_mse:.2f}")


# In[ ]:




