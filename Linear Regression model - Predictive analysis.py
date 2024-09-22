#!/usr/bin/env python
# coding: utf-8

# In[267]:


import pandas as pd
import numpy as np


# In[268]:


mba_salary_df= pd.read_csv('/Users/arijeetbhadra/Downloads/Machine Learning (Codes and Data Files) - U Dinesh/Data/MBA Salary.csv')


# In[269]:


mba_salary_df.head(5)


# In[270]:


mba_salary_df.info()


# In[271]:


import statsmodels.api as sm
x=sm.add_constant(mba_salary_df['Percentage in Grade 10'])
x.head()


# In[272]:


y=mba_salary_df['Salary']
y.head(5).reset_index()


# In[273]:


# Split train & test data
from sklearn.model_selection import train_test_split
train_x,test_x,train_y, test_y = train_test_split(x,y,train_size=0.8,random_state=100)


# In[274]:


# Fitting the model
mba_salary_1m = sm.OLS(train_y, train_x).fit()


# In[275]:


print(mba_salary_1m.params)


# In[276]:


# Model Prediction - 

# MBA SALARY = 30587.28 + 3560.587* ( PERCENTAGE IN CLASS 10)


# In[277]:


# Model diagnostics

# * R- Square - coefficient of determination
# * Hypothesis test
# * Residual analysis
# * outlier analysis


# In[278]:


mba_salary_1m.summary2()


# In[279]:


# R-Squared is 0.211 which means model explains 21.1% of the variation in the salary
# P value for the test is 0.0029 which indicates that there is statistically significant relation
# For SLR - p value for t-test and F -test will be same since null hypothesis is same


# In[280]:


# Check For normal distribution of residual


# In[281]:


import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[296]:


mba_salary_resid = mba_salary_1m.resid
probplot = sm.ProbPlot(mba_salary_resid)
plt.figure(figsize = (8, 6))
probplot.ppplot( line='45' )
plt.title( "Fig 4.1 - Normal P-P Plot of Regression Standardized Residuals" )
plt.show()


# In[283]:


# Test of homoscedasticity


# In[284]:


def get_standarized_values(vals):
    return(vals-vals.mean())/vals.std()


# In[285]:


plt.scatter(get_standarized_values(mba_salary_1m.fittedvalues),get_standarized_values(mba_salary_resid))
plt.title('Residual plot : MBA Salary Prediction')
plt.xlabel('standardized predicted values')
plt.ylabel('standardized residuals')


# In[286]:


# As no funnel shape has been observed it means residual have constant variance (homoscedasticity)


# In[287]:


# outlier analysis through Z test


# In[288]:


from scipy.stats import zscore


# In[289]:


mba_salary_df['z_score_salary']=zscore(mba_salary_df.Salary)


# In[290]:


mba_salary_df[(mba_salary_df.z_score_salary>3.0)|(mba_salary_df.z_score_salary<3.0)]


# In[291]:


# Leverage Values-
from statsmodels.graphics.regressionplots import influence_plot
fig, ax = plt.subplots( figsize=(8,6) )
influence_plot( mba_salary_1m, ax = ax )
plt.title( "Figure 4.4 - Leverage Value Vs Residuals")
plt.show();


# In[292]:


# Cook's Distance-
import numpy as np
mba_influence = mba_salary_1m.get_influence()
(c, p) = mba_influence.cooks_distance
plt.stem( np.arange( len( train_x) ),
         np.round( c, 3 ),
         markerfmt="," );
plt.title( "Cooks distance for all observations in MBA Salaray dataset" );
plt.xlabel( "Row index")
plt.ylabel( "Cooks Distance");


# In[293]:


# Making prediction & measuring Accuracy


# In[299]:


# Predicting on validation set - 
pred_y=mba_salary_1m.predict(test_x)


# In[300]:


# Finding R-Square and RMSE


# In[301]:


from sklearn.metrics import r2_score, mean_squared_error


# In[302]:


np.abs(r2_score(test_y,pred_y))


# In[303]:


np.sqrt(mean_squared_error(test_y,pred_y))


# In[304]:


# Calculating prediction intervals


# In[ ]:




