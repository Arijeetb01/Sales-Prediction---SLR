#!/usr/bin/env python
# coding: utf-8

# In[180]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as mlt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[181]:


# Loading dataset - 


# In[182]:


advertising_df = pd.read_csv('/Users/arijeetbhadra/Downloads/advertising.csv.xls')


# In[183]:


advertising_df.head(10)


# In[184]:


advertising_df.info()


# In[185]:


advertising_df.shape


# In[186]:


advertising_df.describe()


# In[187]:


advertising_df.isnull().sum()


# In[188]:


# Finding co-relation - 


# In[189]:


influencing_factors =['TV','Radio','Newspaper','Sales']
sn.pairplot(advertising_df[influencing_factors],size=2)


# In[190]:


sn.heatmap(advertising_df[influencing_factors].corr(),annot=True)


# In[191]:


# As observed TV shows high co-releation with sales, which is directing us to build model with TV


# In[192]:


advertising_df[['TV','Sales']].head(10).reset_index()


# In[193]:


import statsmodels.api as sm
x=sm.add_constant(advertising_df['TV'])
x.head(10)


# In[194]:


y=advertising_df['Sales']
y.head(10).reset_index()


# In[195]:


# Split train & test data - 
from sklearn.model_selection import train_test_split


# In[196]:


train_x,test_x,train_y,test_y= train_test_split(x,y,train_size=0.8,random_state=100)


# In[197]:


# Fitting the model
advertising_df_1m= sm.OLS(train_y, train_x).fit()


# In[198]:


print(advertising_df_1m.params)


# In[199]:


# Model Prediction - 
# SALES = 6.9955 + 0.054*(TV)


# In[200]:


# Model diagnostics - 
advertising_df_1m.summary2()


# In[201]:


# Model diagnostics
# R- Square - coefficient of determination - 82.2%, which is decent value


# In[202]:


# p values is 0.0581 which indicates that there is statistically significant relation


# In[203]:


# For SLR - p value for t-test and F-test will be same since null hypothesis is same


# In[204]:


# Check For normal distribution of residual- 


# In[205]:


import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[206]:


advertising_df_resid = advertising_df_1m.resid
probplot = sm.ProbPlot(advertising_df_resid)
plt.figure(figsize = (8,6))
probplot.ppplot(line='45')
plt.title('Normal P-P Plot of Regression Standardized Residuals')
plt.show()


# In[207]:


# Diagnoal line is the cummulative distribution of normal distribution,
# whereas dots represents cummulative distribution of residuals. 
# Since dots are close to line, we can say that residuals are normally distributed. 


# In[208]:


# Test of Homoscedasticity


# In[209]:


def get_standardized_values(vals):
    return(vals-vals.mean())/vals.std()


# In[210]:


plt.scatter(get_standardized_values(advertising_df_1m.fittedvalues),get_standardized_values(advertising_df_resid))
plt.title('Residual Plot: Sales Prediction')
plt.xlabel('Standardized predicted values')
plt.ylabel('Standardized Residuals')


# In[211]:


# Shape is not simillar to funnel type structure, which validates homoscedasticity 


# In[212]:


# outlier analysis through Z test


# In[213]:


from scipy.stats import zscore
advertising_df['z_score_Sales']=zscore(advertising_df.Sales)
advertising_df[(advertising_df.z_score_Sales>3.0)|(advertising_df.z_score_Sales<-3.0)]


# In[214]:


# Hence no outliers detected


# In[215]:


# Leverage Values


# In[216]:


from statsmodels.graphics.regressionplots import influence_plot
fig, ax = plt.subplots( figsize=(6,4) )
influence_plot(advertising_df_1m,ax = ax )
plt.title('Leverage Value Vs Residuals')
plt.show();


# In[229]:


# Making prediction using the model


# In[233]:


from sklearn.metrics import r2_score,mean_squared_error
pred_y=advertising_df_1m.predict(test_x)


# In[234]:


# Finding R-Square and RMSE


# In[235]:


np.abs(r2_score(test_y,pred_y))


# In[226]:


# So the model can explains 72.8% of variance in the validation set


# In[239]:


np.sqrt(mean_squared_error(test_y,pred_y))


# In[240]:


# RSME means average error of the model makes in predicting the outcome. The smaller the value of RMSE better the model is


# In[ ]:


# Calculating prediction intervals


# In[248]:


from statsmodels.sandbox.regression.predstd import wls_prediction_std
# Predict the y values
pred_y = advertising_df_1m.predict(test_x)

# Predict the low and high interval values for y
_, pred_y_low, pred_y_high = wls_prediction_std(advertising_df_1m,test_x,alpha = 0.1)

# Store all the values in a dataframe
pred_y_df = pd.DataFrame( { 'Marketing through TV': test_x['TV'],
                            'pred_y': pred_y,
                            'pred_y_left': pred_y_low,
                            'pred_y_right': pred_y_high } )


# In[250]:


pred_y_df[0:10]


# In[ ]:




