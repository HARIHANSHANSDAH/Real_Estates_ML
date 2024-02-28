#!/usr/bin/env python
# coding: utf-8

# ## Real Estate

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing.info()


# In[6]:


housing['CHAS']


# In[7]:


housing['CHAS'].value_counts()


# In[8]:


housing.describe()


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


import matplotlib.pyplot as plt


# In[11]:


housing.hist(bins=50, figsize=(20,15))


# ## Train-Test Splitting

# In[12]:


import numpy as np
def split_train_test(data, test_ratio):    
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[13]:


train_set, test_set = split_train_test(housing, 0.2)


# In[14]:


print(f"Row in train set: {len(train_set)}\nRow in test set: {len(test_set)}")


# In[15]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# In[16]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    start_train_set = housing.loc[train_index]
    start_test_set = housing.loc[test_index]


# In[17]:


start_test_set.describe()


# ## Looking for Correlations

# In[18]:


corr_matrix = housing.corr()


# In[19]:


corr_matrix['MEDV'].sort_values(ascending = False)


# In[20]:


from pandas.plotting import scatter_matrix
attributes = ["MEDV", "RM", "ZN", "LSTAT"]
scatter_matrix(housing[attributes], figsize = (12,8))


# In[21]:


housing.plot(kind="scatter", x="RM", y="MEDV", alpha = 0.8)


# ## Attribute combinations

# In[22]:


housing["TAXRM"] = housing['TAX']/housing['RM'] 


# In[23]:


housing.head()


# In[24]:


housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)


# In[25]:


housing = start_train_set.drop("MEDV", axis = 1)
housing_labels = start_train_set["MEDV"].copy()


# In[26]:


housing.shape


# ## Scikit-learn Design

# Primarily, three types of objects
# 1. Estimators - It estimates some parameter.
# 2. Transformers - transform method take input and returns output based on the learnings form fit(). It also has a convenience function called fit_transform() which fits and transforms.
# 3. Predictors - LinearRegression model is an example of predictor. fit() and predict() are two common functions.It also give score() function which will evaluate the predictions.

# ## Feature Scaling
Primarily, two types of feature scaling methods:
1. Min-max scaling (Normalization)
    (value - min)/(max - min)
    sklearn provides a class called MinMaxScaler for this.
2. Standardization 
    (value - min)/std 
        sklearn provides a class called Standard scaler for this.
# In[27]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler 
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = "median")),
    ('std_scaler', StandardScaler()), 
])


# In[28]:


housing_num_tr = my_pipeline.fit_transform(housing)


# In[29]:


housing_num_tr


# ## Selecting a desired model for Real Estates

# ## Linear Regression & Desision Tree Regression

# In[30]:


from sklearn.linear_model import LinearRegression 
#model = LinearRegression()
from sklearn.tree import DecisionTreeRegressor
#overfitting
#model = DecisionTreeRegressor()
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

model.fit(housing_num_tr, housing_labels)


# In[31]:


some_data = housing.iloc[:5]


# In[32]:


some_labels = housing_labels.iloc[:5]


# In[33]:


prepared_data = my_pipeline.transform(some_data)


# In[34]:


model.predict(prepared_data)


# In[35]:


list(some_labels)


# ## Evaluating the model

# In[36]:


from sklearn.metrics import mean_squared_error
housing_predictions  = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)


# In[37]:


rmse


# ## Using better evaluation technique - Cross Validation

# In[38]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring = "neg_mean_squared_error", cv = 10)
rmse_scores = np.sqrt(-scores)


# In[39]:


rmse_scores


# In[40]:


def print_scores(scores):
    print("Scores : ", scores)
    print("Mean : ", scores.mean())
    print("Standard deviation : ", scores.std())


# In[41]:


print_scores(rmse_scores)


# ## Saving the model

# In[42]:


from joblib import dump, load
dump(model, 'REP.joblib')


# ## Testing the model in test data

# In[43]:


X_test = start_test_set.drop("MEDV", axis=1)
Y_test = start_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[44]:


final_rmse


# In[45]:


prepared_data[0]


# ## Model Usage

# In[46]:


from joblib import dump, load
import numpy as np
model = load('REP.joblib')


# In[47]:


features = np.array([[-0.42152521, -0.48685178, -0.24673925, -0.27144836,  0.2311586 ,
       -0.85627886, -1.42946756, -0.4510327 , -0.42117544, -0.12039257,
        0.3268577 ,  0.41580739,  0.64788652]])
model.predict(features)


# In[ ]:




