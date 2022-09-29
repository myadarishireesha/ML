#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression on Diabetes data
# 

# DATA Loading and Reading

# In[1]:


import pandas as pd
df=pd.read_csv("https://raw.githubusercontent.com/rktrojan/DataSciencePython/main/DataFiles/diabetes.csv",header=None)
df.head(5)


# # Understanding the data:

# In[2]:


df.shape


# In[3]:


df.info()


# In[4]:


df.nunique()


# *total 768 records with 9 columns
# * Independent variables are: 0,1,2,3,4,5,6,7,8. All of these features are continuous values.
# * Dependednt feature or column or target  is : 8, is categorical
# 
# ***No column is a sting or object type.

# 

# # Feature Engineering

# In[5]:


df.isnull().sum()


# In[43]:


df.duplicated().sum()     


# In[6]:


df.describe()


# * No duplicates and mull values found.
# 
# * Some features in data contain 0 values as shown min=0
# 
# * some features like 1,2,3,4 contain  extreme values as compair to their mean.

# # DATA VISUALIZATION

# In[7]:


import matplotlib.pyplot as plt
import seaborn as sb


plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
sb.countplot(df[0])

plt.subplot(2,2,2)
sb.countplot(df[8])



# * Out of 768 records around 500 record have outcome as zero which means that these people don't have Diabetes, 
# and more than 250 have outcome as 1 which means these people have Diabetes.
# * Among all the records the maximum pregnencies are 1.and 0,2.

# In[ ]:


len(df.columns)
get_ipython().run_line_magic('pinfo2', 'enumerate')


# In[8]:



plt.figure(figsize=(20,20))
for i,col in enumerate(df.drop([0,8],axis=1)):
    plt.subplot(4,2,i+1)
    sb.distplot(df[col])


# 1.from above histograms, features 1,2,5 are having 0's are outliers.
# 2.In features 4,6,7 data skewed to right side or positively skewed.
# 3.features 1,2,5 are normally distributed.
# 

# In[9]:


sb.barplot(x=0,y=8,data=df,ci=None)  


# so, increase in feature_0 showing some impact on our target as linearly. 
# but we don't know either it shows a positive impact or neagtive.as person have diabetes or not having diabetes.

# In[10]:


sb.countplot(x=0,hue=8,data=df)


# so, from this graph, having less feature_0 having a chance of low risk of having diabetes,
# as feature -0 increases risk also increases.
# 

# # FINDING OUTLIERS

# In[11]:


plt.figure(figsize=(20,12))
for i,col in enumerate(df.drop([0,8],axis=1)):
    plt.subplot(2,4,i+1)
    sb.boxplot(x=df[8],y=df[col])


# In[12]:


def outliers(df):
    out=[]
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3-q1
    Lower_tail = q1 - 1.5 * iqr
    Upper_tail = q3 + 1.5 * iqr
    for i in df:
        if i > Upper_tail or i < Lower_tail:
            out.append(i)
    print("Outliers for feature_",col,":",len(out))
    


# In[13]:


#calling the function
for col in df.drop(8,axis=1).columns:
    outliers(df[col])


# # removing outliers  with replacing them by their mean

# In[14]:


import numpy as np
for col in df.drop(8,axis=1).columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3-q1
    Lower_tail = q1 - 1.5 * iqr
    Upper_tail = q3 + 1.5 * iqr
    df[col] = np.where( (df[col]<Lower_tail) | (df[col]>Upper_tail), df[col].median(),df[col])

    
    


# In[16]:


df.describe()    #after replaing outliers with meadian of each column


# In[ ]:


#with outliers 

0	1	2	3	4	5	6	7	8
count	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000
mean	3.845052	120.894531	69.105469	20.536458	79.799479	31.992578	0.471876	33.240885	0.348958
std	3.369578	31.972618	19.355807	15.952218	115.244002	7.884160	0.331329	11.760232	0.476951
min	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.078000	21.000000	0.000000
25%	1.000000	99.000000	62.000000	0.000000	0.000000	27.300000	0.243750	24.000000	0.000000
50%	3.000000	117.000000	72.000000	23.000000	30.500000	32.000000	0.372500	29.000000	0.000000
75%	6.000000	140.250000	80.000000	32.000000	127.250000	36.600000	0.626250	41.000000	1.000000
max	17.000000	199.000000	122.000000	99.000000	846.000000	67.100000	2.420000	81.000000	1.000000


# In[17]:


df.corr()


# In[18]:


import matplotlib.pyplot as plt
import seaborn as sb
plt.figure(figsize=(10,7))
sb.heatmap(df.corr(),annot=True)


# In[ ]:


feature 0,7 are higly positively correlated.
feature  1 is correlated wth target.
feature 5,4 are correlated with feature 3.


# In[19]:


plt.figure(figsize=(15,15))
for i,col in enumerate(df.drop([0,8,1],axis=1)):
    plt.subplot(3,3,i+1)
    sb.scatterplot(1,df[col],hue=8,data=df)


# In[ ]:


so, for higher feature_1 the outcome is more as having diabetes.


# In[20]:


plt.figure(figsize=(15,15))
for i,col in enumerate(df.drop([0,8,1,2],axis=1)):
    plt.subplot(3,5,i+1)
    sb.scatterplot(2,df[col],hue=8,data=df)


# feature_2 not shows much impact on outcome

# In[22]:


plt.figure(figsize=(15,15))
for i,col in enumerate(df.drop([0,8,1,3],axis=1)):
    plt.subplot(3,5,i+1)
    sb.scatterplot(3,df[col],hue=8,data=df)


# In[ ]:


feature_3 and feature_5 are positively linear corelated.


# In[23]:


plt.figure(figsize=(15,10))
for i,col in enumerate(df.drop([0,8,5],axis=1)):
    plt.subplot(3,6,i+1)
    sb.scatterplot(5,df[col],hue=8,data=df)


# In[24]:


plt.figure(figsize=(15,15))
for i,col in enumerate(df.drop([0,8,6],axis=1)):
    plt.subplot(3,6,i+1)
    sb.scatterplot(6,df[col],hue=8,data=df)


# In[ ]:


From both graphs,for high value of feature_1  the chances are high to having diabetes.


# # DATA SCALING

# In[25]:


from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder



df["0_sc"] = df[0]/df[0].max()
df["1_sc"] = df[1]/df[1].max()
df["2_sc"] = df[2]/df[2].max()
df["3_sc"] = df[3]/df[3].max()
df["4_sc"] = df[4]/df[4].max()
df["5_sc"] = df[5]/df[5].max()
df["6_sc"] = df[6]/df[6].max()
df["7_sc"] = df[7]/df[7].max()

df


# In[26]:


scaled_df=df.drop([0,1,2,3,4,5,6,7],axis=1)
scaled_df


# # Check class imbalance problem

# In[27]:


#majority
print(len(scaled_df[scaled_df[8]==0]))
print(len(scaled_df[scaled_df[8]==0])/len(scaled_df[8]))


# In[28]:


#minority class
print(len(scaled_df[scaled_df[8]==1]))
print(len(scaled_df[scaled_df[8]==1])/len(scaled_df[8]))


# In[29]:


from imblearn.over_sampling import SMOTE,SMOTEN,SMOTENC


# In[30]:


# transform the dataset

oversample = SMOTEN(sampling_strategy="minority", random_state=100, n_jobs=-1)

X=scaled_df[["0_sc","1_sc","2_sc","3_sc","4_sc","5_sc","6_sc","7_sc"]]
Y=scaled_df[8]


data_X, data_Y = oversample.fit_resample(X, Y)


# In[31]:


len(data_X)


# In[32]:


len(data_Y)


# In[33]:


print(len(data_Y[data_Y==0]))
print(len(data_Y[data_Y==0])/len(data_Y))


# In[34]:


print(len(data_Y[data_Y==1]))
print(len(data_Y[data_Y==1])/len(data_Y))


# NOw, both classes lengths are same means class balanced.

# # SPLIT DATA into 2 parts - train and test

# In[35]:


from sklearn.model_selection import train_test_split





X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size=0.3,random_state=100)  




# In[36]:


X_train


# In[37]:


Y_train


# # Model Config

# In[38]:


# import the Python Class LogisticRegression for CLASSIFICATION



from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(
    
    random_state=100, max_iter=10000, 
                              
    penalty='l1', solver='saga', 
                              
    verbose=True, n_jobs=-1
) 


# # TRAINING Model

# In[40]:


logr.fit(X_train, Y_train)

logr


# In[62]:


#To retrieve the intercept:
print(logr.intercept_)

#For retrieving the slope:
print(logr.coef_)


# In[41]:


#train accuracy score

logr.score(X_train,Y_train)


#69%-without penalty
#71%---with l1 penalty
#70%--with l2-penalty


# # TESTING PHASE

# In[42]:


Y_predicted = logr.predict(X_test)

Y_predicted


# In[43]:


from sklearn import metrics

metrics.accuracy_score(Y_test,Y_predicted)


# # ERROR ANALYSIS
# 
# 
# CONFUSION MATRIX

# In[44]:


len(X_test)


# In[45]:


len(Y_test)


# In[71]:


len(Y_predicted)


# In[72]:


output = Y_test - Y_predicted

print(len(Y_test), "Total Test Records \n")

print(len(output[output==0]) , "are correct classifcations \n")

print(len(output[output != 0]) , "are INcorrect classifcations \n")


# In[73]:


from sklearn import metrics

print(metrics.confusion_matrix(Y_test,Y_predicted ))


# In[74]:


import seaborn as sb

sb.heatmap(metrics.confusion_matrix(Y_test,Y_predicted ),annot=True, fmt='d')


# In[75]:


#Analysis from heatmap:
# * 0 is Negative class
# * 1 is Positive class

print("False positive =47 means 47 people don't had diabetes but, they diagnosed as diabetes")

print("False Negative=48 means 48 members had diabetes, but they diagnosed as no diabetes")



Total=106+47+48+99  
print("Total is",Total)

Accuracy=(106+99)/Total
print("Test Accuracy",Accuracy)

error_rate= (48+47)/Total
print("error_rate is",error_rate)


# # Classification Report

# In[76]:


#positive --- daibetes
#negative  ---- no-diabetes


from sklearn import metrics

print(metrics.classification_report(Y_test, Y_predicted))


# # ROC

# In[77]:


from sklearn import metrics

print(metrics.roc_curve(Y_test, Y_predicted))


# In[79]:


fpr=[0.        , 0.30718954, 1.        ]

tpr=[0.        , 0.67346939, 1.        ]


# In[80]:


import matplotlib.pyplot as plt

plt.scatter(fpr,tpr)

plt.plot(fpr,tpr)

#guess line
plt.plot([0,1],[0,1])

plt.show()


# In[81]:


print(metrics.auc(fpr, tpr))


# In[82]:


#Test Accuracy

print(metrics.roc_auc_score(Y_test, Y_predicted))


# # CONCLUSION:

# Train accuracy: 0.7142857142857143
#     Test accuracy using test accuracy score function: 0.683139922635721
#     Test Accuracy using ROC and AUC: 0.683139925 and 0.683139922635721.
#    
#     Train accuracy is very less,to improve both test and train accuracy further implementation with non-linear classification models is needed.

# Let's try with KNN-Model
# # KNN-Model

# # Model-config with K=5

# In[114]:


#Import k-nearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier

#Create KNN Classifier with K=5

knn = KNeighborsClassifier(n_neighbors=5, p=1, metric='minkowski' )


# # TRAINING Phase

# In[115]:


#Train the model using the training sets
knn.fit(X_train, Y_train)


# In[116]:


#training accuracy:

knn.score(X_train, Y_train)


# In[117]:


#Predict the response for test dataset

y_pred = knn.predict(X_test)


# In[118]:


from sklearn import metrics


# Model Accuracy, how often is the classifier correct?

print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))


# # Tunning parameter- Gridsearch CV

# In[86]:


from sklearn.model_selection import GridSearchCV



params = [
    {
        'n_neighbors': [1,3,5,7,9,11,13,15,17,21],
     
         'p': [1,2,3],
     
         'metric': ['minkowski','mahalanobis']
    }
]


gs_knn = GridSearchCV(KNeighborsClassifier(),
                      
                      param_grid=params,
                     
                      scoring='accuracy',
                      
                      cv=5)



gs_knn.fit(X_train, Y_train)


# In[87]:


gs_knn.best_estimator_


# In[88]:


gs_knn.best_params_


# In[89]:


#Training accuracy

gs_knn.best_score_


# In[90]:


gs_knn.cv_results_


# # TESTING PHASE

# In[93]:


#Predict the response for test dataset

y_pred = knn.predict(X_test)


# In[94]:


y_pred


# In[96]:


#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics


# Model Accuracy, how often is the classifier correct?

print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))


# In[ ]:


so, both traing accuracy and testing accuracy are 74%,73% for metric': 'minkowski', 'n_neighbors': 5, 'p': 1. 
seems much better than previous model.

