#!/usr/bin/env python
# coding: utf-8

# **The Sparks Foundation**

# **Data Science and BUsiness Analytics Internship**

# **Task-1 :Prediction using Supervised ML**

# **By-Priyanka Mohanta**

# In[2]:


# Importing libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#reading the data
url="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"


# In[4]:


data=pd.read_csv(url)
print("Data import successfully")


# In[5]:


#check the data set
data


# In[6]:


data.info()


# In[7]:


#summery statistics for numerical columns
data.describe()


# In[8]:


#check the first 5 rows.
data.head()


# In[9]:


#check the null value of the data set
data.isnull().sum()


# **Visualizing The dataset**

# In[10]:


#Pearson's Correlation
cor=data.corr()
plt.figure(figsize=(5,5))
sns.heatmap(cor,annot=True,cmap='coolwarm')
plt.show()


# In[11]:


# Plotting the distribution between score and hours
sns.regplot(x=data.Hours, y=data.Scores,color='r')
plt.title("Hours Vs Scores")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


# **From the above graph, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.**

# In[12]:


x=data.Hours.values.reshape(-1,1)
y=data.Scores.values.reshape(-1,1)


# In[13]:


#divide into training and testing
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)


# In[14]:


#Apply Linear Regression

from sklearn.linear_model import LinearRegression
Linear_reg= LinearRegression()
Linear_reg.fit(x,y)
yp=Linear_reg.predict(x)     # Predicting the scores


# In[15]:


#calculate score (r-squared)

Linear_reg.score(x_test,y_test)


# In[16]:


# Plotting for the test data
plt.scatter(x=x, y=y,color='b')
plt.plot(x,yp,color='r')
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.show()


# Given task is,What will be predicted score if a student studies for 9.25 hrs/ day? 

# In[17]:


hours =9.25
predicted_values=Linear_reg.predict([[hours]])
print("Student Studies 9.25 hrs/ day,Prediction of the score is",predicted_values)


# **THANK YOU**

# In[ ]:




