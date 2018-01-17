
# coding: utf-8

# # Simple Linear Regression

# Simple Linear Regression: is a statistical method that allows us to summarize and study relationships between two continuous (quantitative) variables.
# 1. The one varibale, X, is regarded as the predictor, explanatory or independent variable.
# 2. The other, denoted y, is regarded as the response, outcome, or dependent variable.
# 
# Formula: y = B0 + B1 * X1 - also known as y = mx + b
#     Where:  y is the valune on the y line
#             B0 is the constant or intercept
#             B1 is the slope
#             X is the value on the x line
# 
# Problem: Given the data set with overall years of experience and salary of 30 employees, create a regression model to determine what Salary correlates with the overall years of experience of the employee, and expected salary increment.

# In[1]:


# Import Python libraries
# import numpy as np
import pandas as pd
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt


# ### Import the Dataset

# In[2]:


# Dataset
dataset = pd.read_csv('Salary_Data.csv')
dataset # show the whole datset


# In[3]:


# Separate columns in the dataset to obtain the independent variable
# iloc works on the positions of the index/columns (takes in integers as parameters).
X = dataset.iloc[:,:-1].values # x is matrix of features, the independent variable (YearsofExperience)
print(X)


# In[4]:


# y = dataset.head(n=10).iloc[:, 1].values - this is if we want to run only the first 10 columns and .values will only show values without the labels
y = dataset.iloc[:, 1].values # y is the matrix of features, the dependent varibale (Salary)
print(y)


# ### Splitting the Dataset into Training Set and Test Set.

# In[5]:


from sklearn.model_selection import train_test_split # This allows us to validate the splitting of the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0) # Test 1/3 of the dataset and let it be random 


# In[6]:


# Left side of the column is the index and its corresponding value (YearsExperience)
print(X_train) # outputs the training dataset of 20 samples out of 30 at random


# In[7]:


# Left side of the column is the index and its corresponding value (YearsExperience)
print(X_test) # outputs the test dataset of 10 datasets of 30 at random


# In[8]:


# Left side of the column is the index and its corresponding value (Salary)
print(y_train) # outputs the trainig dataset of 20 datasets of 30 at random


# In[9]:


# Left side of the column is the index and its corresponding value (Salary)
print(y_test) # outputs the test dataset of 10 datasets of 30 at random


# ### Fitting Simple Linear Regresion to the Trainig set.

# In[10]:


# Here is where our model will learn the correlation between the dependant (salary) variable and independant (yearsofexperience) variable. The machine in Machine Learning is the  
# The machine in MACHINE LEARNING is the simple linear regression model (regressor), and the learning is the regressor learning from the X and y training set. 
from sklearn.linear_model import LinearRegression # LinearRegression is a class
# object of LinearRegression class, where we create our linear regressor
regressor = LinearRegression() # no parameters needed
regressor.fit(X_train, y_train) # fit is a method


# ### Prediciting the Test Set results.

# In[11]:


y_pred = regressor.predict(X_test)
print(y_pred) # the machine model prints the predicted salary of employess based on the independant varibale (yearsofexperience)


# In[12]:


print(y_test) # prints the real salaries of the employees of the company. Stop here and compare the salaries of each employee. 
# Some predicited salaries are close to the real salaries, while others are off by quite a bit. 
# Reason is we use simple linear regresion, which is a straight line across the dataset on a graphical representation of the model.


# ### Visualize the Training Set Results.

# In[13]:


plt.scatter(X_train, y_train, color='red')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show() 


# The real values obtained from the dataset are represented by red dots.

# In[14]:


plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# The predicted values are represented by the blue line (Simple Linear Regression).
