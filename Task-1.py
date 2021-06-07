#!/usr/bin/env python
# coding: utf-8

# # The Spark Foundation
# ## Task-1 Prediction using Supervised ML 
# ### Aim - Predict the percentage of an student based on the no. of study hours. What will be predicted score if a student studies for 9.25 hrs/ day?
# 

# ## Author - Manthan Patel

# In[3]:


#importring the required libraries
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
#%matplotlib inline


# In[5]:


url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported successfully")

data.head(10)


# In[8]:


# Plotting the dataset of hours of study against Score
data.plot(x='Hours',y='Scores',style='o')
plt.title('Hours vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.show()


# #### From the graph we can see that number of hours studied is directly proportional to the scored. If number of study hours increase, the scored increases too.
# ### Splitting the data into x and y
# 

# In[9]:


x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values


# In[10]:


x


# In[11]:


y


# ### Splitting the data into training and test sets

# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=101)


# In[13]:


x_train


# In[14]:


x_test


# In[15]:


y_train


# In[16]:


y_test


# ### Linear Regression Model

# In[17]:


from sklearn.linear_model import LinearRegression
LR=LinearRegression()


# In[18]:


LR.fit(x_train,y_train)
print("Training complete")


# ### Plotting the Regression Line

# In[ ]:





# In[19]:


line = LR.coef_*x+LR.intercept_
plt.scatter(x,y,marker ='+')
plt.title('Regression line')
plt.plot(x,line,color="black")
plt.show()


# ### Prediction

# In[20]:


y_pred_LR=LR.predict(x_test)
x_test


# In[21]:


y_test


# In[22]:


y_pred_LR


# ### Actual data and Predicted data

# In[26]:


df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_LR})  
df1


# In[ ]:





# In[28]:


#predicting from given data
hours = 9.25
ans = LR.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(ans[0]))


# ### Predicted score for a student studying 9.25 hours is 94.29276125536514 %

# In[30]:



from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred_LR))


# In[ ]:




