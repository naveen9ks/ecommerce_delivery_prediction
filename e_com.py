#!/usr/bin/env python
# coding: utf-8

# # E COMMERCE PRODUCT DELIVERY PREDICTION

# IMPORTING LIBRARIES

# In[1]:


import pandas as pd     # to read data set
import numpy as np      # to do mathematical operations
import matplotlib.pyplot as plt     # to do visualizations like bar chart, etc..,
import seaborn as sns       # to do analytical visualizations like pie chart, etc..,


# # DATA PREPROCESSING

# In[2]:


df= pd.read_csv("E_Commerce.csv")       # to  load dataset
df.head()        # show top 4 rows of a dataset


# In[3]:


df.describe()        # shows description about data columns


# In[4]:


df.info()        # know information about data


# In[5]:


# remove unwanted warnings
import warnings
warnings.filterwarnings("ignore")


# # STEP 1: EXPLORATORY DATA ANALYSIS(EDA)

# In[6]:


df.columns      # shows details of columns


# # STEP1(1) UNIVARIATE ANALYSIS

# (CATEGORICAL DATA)

# 1.1 WAREHOUSE BLOCK ANALYSIS

# In[7]:


ware= df["Warehouse_block"].value_counts()      # calculates total value counts


# In[8]:


plt.pie(ware, labels=ware.values)       # creates pie chart
plt.legend(ware.index)      # shows color identification
plt.title("Warehouse Block Analysis")       # shows title of chart
plt.show()      # shows entire chart


# 1.2 SHIPMENT ANALYSIS

# In[9]:


sns.countplot(x="Mode_of_Shipment", data=df, color='r',width=0.25)      # creates bar chart
plt.xlabel("Mode of Shipment")      # shows title on x-axis
plt.ylabel("Frequency")     # shows title on y-axis
plt.title("Shipment Analysis")      # shows title of chart
plt.show()      # shows entire chart


# 1.3 CUSTOMER RATING ANALYSIS

# In[10]:


sns.countplot(x="Customer_rating", data=df, color='r', width=0.5)       # creates bar chart
plt.xlabel("Customer Rating")       # shows title on x-axis
plt.ylabel("Frequency")     # shows title on y-axis
plt.title("Customer Rating Analysis")       # shows title of chart
plt.show()      # shows entire chart


# 1.4 PRODUCT IMPORTANCE ANALYSIS

# In[11]:


sns.countplot(x="Product_importance", data=df, color='r', width=0.25)       # creates bar chart
plt.xlabel("Product Importance")        # shows title on x-axis
plt.ylabel("Frequency")     # shows title on y-axis
plt.title("Product Importance Analysis")        # shows title of chart
plt.show()      # shows entire chart


# 1.5 GENDER ANALYSIS

# In[12]:


sns.countplot(x="Gender", data=df, color='r', width=0.25)       # creates bar chart
plt.xlabel("Gender")        # shows title on x-axis
plt.ylabel("Frequency")     # shows title on y-axis
plt.title("Gender Analysis")        # shows title on chart
plt.show()      # shows entire chart


# 1.6 REACHED ON TIME ANALYSIS

# In[13]:


sns.countplot(x="Reached.on.Time_Y.N", data=df, color='r', width=0.25)      # creates bar chart
plt.xlabel("Reached ON Time Y/N")       # shows title on x-axis
plt.ylabel("Frequency")     # shows title on y-axis
plt.title("Reached ON Time Y/N")        # shows title of chart
plt.show()      # shows entire chart


# (NUMERICAL DATA)

# 1.7 CUSTOMER CARE ANALYSIS

# In[14]:


plt.hist("Customer_care_calls", color='r', data=df)     # creates histogram chart
plt.xlabel("Customer Care Calls")       # shows title on x-axis
plt.ylabel("Frequency")     # shows title on y-axis
plt.title("Customer Care Analysis")     # shows title on chart
plt.show()      # shows entire chart


# 1.8 COST OF PRODUCT ANALYSIS

# In[15]:


plt.hist("Cost_of_the_Product", data=df, color='r')     # creates histogram chart
plt.xlabel("Cost of Product")       # shows title on x-axis
plt.ylabel("Frequency")     # shows title on y-axis
plt.title("Cost of Product Analysis")       # shows title of chart
plt.show()      # shows entire chart


# 1.9 DISCOUNT ANALYSIS

# In[16]:


plt.hist("Discount_offered", data=df, color='r')        # creates histogram chart
plt.xlabel("Discount Offered")      # shows title on x-axis
plt.ylabel("Frequency")     # shows title on y-axis
plt.title("Discount Analysis")      # shows title of chart
plt.show()      # shows entire chart


# 1.10 WEIGHT ANALYSIS

# In[17]:


plt.hist("Weight_in_gms", data=df, color='r')       # creates histogram chart
plt.xlabel("Weight in gms")     # shows title on x-axis
plt.ylabel("Frequency")     # shows title on y-axis
plt.title("Weight Analysis")        # shows title of chart
plt.show()      # shows entiere chart


# 1.11 PRIOR PURCHASE ANALYSIS

# In[18]:


plt.hist("Prior_purchases", data=df, color='r')     # creates histogram chart
plt.xlabel("Prior Purchases")       # shows title on x-axis
plt.ylabel("Frequency")     # shows title on y-axis
plt.title("Prior Purchase Analysis")        # shows title of chart
plt.show()      # shows entire chart


# # STEP 1(2) BIVARIATE ANALYSIS

# Warehouse Block Delivery Analysis

# In[19]:


sns.countplot(x="Warehouse_block", hue="Reached.on.Time_Y.N", data=df, palette='dark:r', width=0.5)     # creates bar chart
plt.xlabel("Warehouse Block")       # shows title on x-axis
plt.ylabel("Count")     # shows title on y-axis
plt.title("Warehouse Block Delivery Analysis")      # shows title of chart
plt.show()      # displays entire chart


# Shipment Delivery Analysis

# In[20]:


sns.countplot(x="Mode_of_Shipment", hue="Reached.on.Time_Y.N", palette='dark:r', data=df, width=0.3)        # creates bar chart
plt.xlabel("Mode of Shipment")      # shows title on x-axis
plt.ylabel("Count")     # shows title on y-axis
plt.title("Shipment Delivery Analysis")     # shows title of chart
plt.show()      # displays entire chart


# Customer Care Delivery Analysis

# In[21]:


sns.countplot(x="Customer_care_calls", hue="Reached.on.Time_Y.N", data=df, palette='dark:r', width=0.5)     # create bar chart
plt.xlabel("Customer care calls")       # shows title on x-axis
plt.ylabel("Count")     # shows title on y-axis
plt.title("Customer Care Delivery Analysis")        # shows title for chart
plt.show()      # shows entire chart


# Customer Rating Delivery Analysis

# In[22]:


sns.countplot(x="Customer_rating", hue="Reached.on.Time_Y.N", palette='dark:r', data=df, width=0.5)     # creates bar chart
plt.xlabel("Customer Rating")       # shows title on x-axis
plt.ylabel("Count")     # shows title on y-axis
plt.title("Customer Rating Delivery Analysis")      # shows title of chart
plt.show()      # shows entire chart


# Prior Purchases Delivery Analysis

# In[23]:


sns.countplot(x="Prior_purchases", hue="Reached.on.Time_Y.N", palette='dark:r', data=df, width=0.5)     # creates bar chart
plt.xlabel("Prior Purchases")       # shows title on x-axis
plt.ylabel("Count")     # shows title on y-axis
plt.title("Prior Purchases Delivery Analysis")      # shows title of chart
plt.show()      # shows entire chart


# Product Importance Delivery Analysis

# In[24]:


sns.countplot(x="Product_importance", hue="Reached.on.Time_Y.N", palette='dark:r', data=df, width=0.5)      # creates bar chart
plt.xlabel("Product Importance")        # shows title on x-axis
plt.ylabel("Count")     # shows title on y-axis
plt.title("Product Importance Delivery Analysis")       # shows title of chart
plt.show()      # shows entire chart


# Gender Based Delivery Analysis

# In[25]:


sns.countplot(x="Gender", hue="Reached.on.Time_Y.N", palette='dark:r', data=df, width=0.3)      # creates bar chart
plt.xlabel("Gender")        # shows title on x-axis
plt.ylabel("Count")     # shows title on y-axis
plt.title("Gender Delivery Analysis")       # shows title of chart
plt.show()      # shows entire chart


# # STEP1(3): MULTIVARIATE ANALYSIS

# Cost of Product vs Discount Offered over Reached ON Time Analysis

# In[26]:


sns.scatterplot(x="Cost_of_the_Product", y="Discount_offered", hue="Reached.on.Time_Y.N", data=df, palette='dark:r')        # creates scatter plot
plt.xlabel("Cost")      # shows title on x-axis
plt.ylabel("Discount")      # shows title on y-axis
plt.title("Cost of Product vs Discount Offered Analysis")       # shows title of chart
plt.legend()        # displays color identification
plt.show()      # shows entire chart


# Cost of Product vs Weight in gms over Reached ON Time Analysis

# In[27]:


sns.scatterplot(x="Cost_of_the_Product", y="Weight_in_gms", hue="Reached.on.Time_Y.N", data=df, palette='dark:r')       # creates scatter plot
plt.xlabel("Cost")      # shows title on x-axis
plt.ylabel("Weight")        # shows title on y-axis
plt.title("Cost of Product vs Weight of Product Analysis")      # shows title of chart
plt.legend()        # creates color identification
plt.show()      # displays entire chart


# # STEP 2:CORRELATION MATRIX

# In[28]:


corr_matrix= df.corr(numeric_only=True)     # creates correlation matrix


# In[29]:


plt.figure(figsize=(10,8))      # creates chart with given size
sns.heatmap(corr_matrix, annot= True, cmap='coolwarm', fmt=".2f", linewidths=0.5)       # creates coorelation chart with values


# # STEP3: DETECTING OUTLIERS

# Cost of Product Outliers

# In[30]:


df.boxplot("Cost_of_the_Product")       # creates box plot
plt.show()           # shows box plot chart


# Product Weight Outliers

# In[31]:


df.boxplot("Weight_in_gms")     # creates box plot
plt.show()      # shows entire box plot


# Product Discount Outliers

# In[32]:


df.boxplot("Discount_offered")      # creates box plot
plt.show()      # displays entire chart


# # STEP 4: REMOVE OUTLIERS

# In[33]:


# remove outliers by Inter Quantile Range (IQR) method
Q1= df['Discount_offered'].quantile(0.25)
Q3= df['Discount_offered'].quantile(0.75)
IQR= Q3- Q1


lower= Q1-0.5*IQR
upper= Q3-0.5*IQR


# In[34]:


df_cleaned= df[(df['Discount_offered']>=lower) & (df['Discount_offered']<=upper)]       # creates cleaned data


# Visualize Cleaned Data

# In[35]:


plt.boxplot(df_cleaned['Discount_offered'])     # creates box plot
plt.grid()      # shows grid lines
plt.show()      # shows entire chart


# # STEP 5: BUILD MODEL

#     Inference:
#         From the given data set, the data's were observed as Classification problem
#         So we are moving with Supervised Learning Models

# In[36]:


df["Reached.on.Time_Y.N"].value_counts()        # calculates total values of each category


# the target is in the ratio of 60:40, so the data is maximum balanced

# IMPORT LIBRARIES

# In[38]:


from sklearn.model_selection import train_test_split        # importing train test split
from sklearn.preprocessing import StandardScaler, LabelEncoder        # importing preprocessing libraries
from sklearn.linear_model import LogisticRegression     # importing Logistic Regression library
from sklearn.tree import DecisionTreeClassifier     # importing tree model
from sklearn.ensemble import RandomForestClassifier     # importing ensemble technique
from xgboost import XGBClassifier       # importing xgbclassifier model
from sklearn.neighbors import KNeighborsClassifier      # importing neighbours model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, classification_report        # importing evaluation techniques


# In[39]:


df=df.drop("ID", axis=1)


# Label Encoding

# In[40]:


# converting categorical column to encoding
le= LabelEncoder()

for col in ['Warehouse_block', 'Mode_of_Shipment','Product_importance', 'Gender']:

    df[col]= le.fit_transform(df[col])


# In[41]:


df.head()


# Defining features and target

# In[42]:


# x is a input feature
X= df[['Warehouse_block', 'Mode_of_Shipment', 'Customer_care_calls','Customer_rating', 'Cost_of_the_Product', 'Prior_purchases','Product_importance', 'Gender', 'Discount_offered', 'Weight_in_gms']]
y= df['Reached.on.Time_Y.N']        # target feature


# In[43]:


df.info()


# Apply onehot encoding

# Split train test data

# In[44]:


X_train, X_test, y_train, y_test= train_test_split(X, y,test_size=0.3, random_state=42)     # train test split in 70:30 ratio


# (1)LOGISTIC REGRESSION MODEL

# In[45]:


model1= LogisticRegression()     # select the model
model1.fit(X_train,y_train)      # fit the model


# Prediction

# In[46]:


y_pred= model1.predict(X_test)      # predict the model


# Evaluation

# In[47]:


Accuracy = accuracy_score(y_test, y_pred)       # calculates accuracy value
Precision= precision_score(y_test, y_pred)      # calculates precision value
Confusion= confusion_matrix(y_test,y_pred)      # prepares confusion matrix
classification= classification_report(y_test, y_pred)       # prepares classification report


print(f'Accuracy:{Accuracy:.2f}')       # shows accuracy score
print(f'Precision:{Precision:.2f}')     # shows precision score
print(f'Confusion Matrix:{Confusion}')      # shows confusion matrix
print(f'Classification Report:{classification}')        # shows classification report


# (2)DECISION TREE

# In[48]:


model2= DecisionTreeClassifier()        # selects the model
model2.fit(X_train, y_train)        # fit the model


# Prediction

# In[49]:


y_pred= model2.predict(X_test)      # predicts the model


# Evaluation

# In[50]:


Accuracy = accuracy_score(y_test, y_pred)       # calculates accuracy score
Precision= precision_score(y_test, y_pred)      # calculates precision score
Confusion= confusion_matrix(y_test,y_pred)      # prepares confusion matrix
classification= classification_report(y_test, y_pred)       # prepares classification report


print(f'Accuracy:{Accuracy:.2f}')       # shows accuracy value
print(f'Precision:{Precision:.2f}')     # shows precision value
print(f'Confusion Matrix:{Confusion}')      # shows confusion matrix
print(f'Classification Report:{classification}')        # shows classification report


# (3) RANDOM FOREST

# In[51]:


model3= RandomForestClassifier()        # selects the model
model3.fit(X_train,y_train)     # fits the model


# Prediction

# In[52]:


y_pred= model3.predict(X_test)      # predicts the model


# Evaluation

# In[53]:


Accuracy= accuracy_score(y_pred, y_test)        # calculates accuracy score
Precision= precision_score(y_pred, y_test)      # calculates precision score
Confusion= confusion_matrix(y_pred, y_test)     # prepares confusion matrix
classification= classification_report(y_pred, y_test)       # prepares classification report

print(f'Accuracy:{Accuracy:.2f}')       # shows accuracy value
print(f'Precision:{Precision:.2f}')     # shows precision value
print(classification)       # shows classification report
print(Confusion)        # shows confusion matrix


# (4) KNN MODEL

# In[54]:


model4= KNeighborsClassifier()       # selects the model
model4.fit(X_train, y_train)     # fits the model


# Prediction

# In[55]:


y_pred= model4.predict(X_test)       # predicts the model


# Evaluation

# In[56]:


Accuracy = accuracy_score(y_test, y_pred)       # calculates accuracy score
Precision= precision_score(y_test, y_pred)      # calculates precision score
Confusion= confusion_matrix(y_test,y_pred)      # prepares confusion matrix
classification= classification_report(y_test, y_pred)       # prepares classification report


print(f'Accuracy:{Accuracy:.2f}')       # shows accuracy value
print(f'Precision:{Precision:.2f}')     # shows precision value
print(f'{Confusion}')      # shows confusion matrix
print(f'{classification}')        # shows classification report


# (5) XGBOOST

# In[57]:


model5= XGBClassifier()      # selects the model
model5.fit(X_train, y_train)     # fits the model


# Prediction

# In[58]:


y_pred= model5.predict(X_test)      # predicts the model


# Evaluation

# In[59]:


Accuracy = accuracy_score(y_test, y_pred)       # calculates accuracy score
Precision= precision_score(y_test, y_pred)      # calculates precision score
Confusion= confusion_matrix(y_test,y_pred)      # prepares confusion matrix
classification= classification_report(y_test, y_pred)       # prepares classification report


print(f'Accuracy:{Accuracy:.2f}')       # shows accuracy value
print(f'Precision:{Precision:.2f}')     # shows precision value
print(f'{Confusion}')       # shows cofusion matrix
print(f'{classification}')      # shows classification report


# # STEP 6:HYPER PARAMETER TUNING

# Import Libraries

# In[60]:


from sklearn.model_selection import GridSearchCV        # selects GridSearchCV for model tuning


# Parametric Tuning

# In[61]:


# choosing parameter grid
param_grid= {
    'n_estimators':[100, 150, 200],
    'max_depth':[5,10,None],
    'max_samples_split':[2,5,10],
    'max_samples_leaf':[1,2,4],
    'criterion':['gini', 'entropy']

}


# GridSearch CV

# In[63]:


# grid search operations
grid_search= GridSearchCV(
    estimator=XGBClassifier(),
    param_grid= param_grid,
    cv=5,       # cross validation
    n_jobs=1,
    verbose=1
)


# Fit Model

# In[ ]:


grid_search.fit(X_train,y_train)        # fits the model


# Best Estimator

# In[62]:


best_xg= grid_search.best_estimator_        # selects best model


# Prediction

# In[63]:


y_pred= best_xg.predict(X_test)     # predicts the value


# Evaluation

# In[64]:


Accuracy = accuracy_score(y_test, y_pred)       # calculates accuracy score
Precision= precision_score(y_test, y_pred)      # calculates precision score
Confusion= confusion_matrix(y_test,y_pred)      # prepares confusion matrix
classification= classification_report(y_test, y_pred)       # prepares classification report


print(f'Accuracy:{Accuracy:.2f}')       # shows accuracy value
print(f'Precision:{Precision:.2f}')     # shows precision value
print(f'Confusion Matrix:{Confusion}')      # shows confusion matrix
print(f'Classification Report:{classification}')        # shows classification report


# # STEP 7: DEPLOYMENT

# In[90]:


import streamlit as st      # to deploy the model
import joblib       # to save and load model


# In[ ]:


model= joblib.dump(model5,"ecom.pkl")       # save the model


# In[ ]:


model6= joblib.load("ecom.pkl")     # load the model


# In[129]:


import warnings


# In[133]:


warnings.filterwarnings("ignore")


# In[ ]:


# title of page
st.title("E-Commerce delivery prediction")


# input features
Block= st.selectbox("Ware house block",list(range(0,5)))
Shipment= st.selectbox("shipment",list(range(0,3)))
customercalls= st.selectbox("customer calls",list(range(1,8)))
CustomerRating= st.selectbox("Customer rating",list(range(1,6)))
CostofProduct= st.number_input("Enter cost of product")
Priorpurchase=st.selectbox("Prior purchase",list(range(2,11)))
ProdctImportance= st.selectbox("Product Importance",list(range(0,3)))
Gender= st.selectbox("Gender",[0,1])
Discount= st.number_input("Enter Discount Offered")
Weight= st.number_input("Enter weight in gms")


# In[136]:


# calculates and predicts
input_data= np.array([[Block,Shipment,customercalls,CustomerRating,ProdctImportance,Gender,CostofProduct,Priorpurchase,Discount,Weight]])


if st.button("Predict"):
    prediction= model6.predict(input_data)[0]
    
    if prediction==1:
        st.success("Delivered ON time")
    else:
        st.error("Delivered Not-ON time")


# # STEP 8: CONVERT VSCODE TO SCRIPT

# In[139]:


get_ipython().system('jupyter nbconvert --to script e_com.ipynb      # converts vscode to script')

