#!/usr/bin/env python
# coding: utf-8

# <h1><center><font size="6">Credit Card Fraud Detection</font></center></h1>
# 
# 
# <center><img src="Credit-Card.jpeg" width="600"></img></center>
# 
# 
# # <a id='0'>Content</a>
# 
# - <b><a href='#0' style="text-decoration:none" >Business Understanding</a></b>
# - <b><a href='#1' style="text-decoration:none" >Data</a></b>
# - <b><a href='#2' style="text-decoration:none" >Data Understanding</a></b>  
# - <b><a href='#3' style="text-decoration:none" >Data preparation</a></b>  
# - <b><a href='#7' style="text-decoration:none" >Modelling & Evaluation</a></b>  
# - <b><a href='#11' style="text-decoration:none" >Saved Models</a></b>  
# - <b><a href='#12' style="text-decoration:none" >Load & Use Models</a></b>  
# - <b><a href='#13' style="text-decoration:none" >Conclusion & Room for improvement</a></b>  
# 

# # <center><a id="0">Business Understanding</a></center>  
# **Fraud detection is the process of using tools and procedures to prevent the theft of money, information, and assets. It is a security barrier that protects against various forms of fraud, including minor infractions and felony crimes. Examples of fraud include forging signatures on checks and stealing credit card numbers from millions of account holders.    
# ***Thanks to the internet :)*** there are countless ways criminals can obtain your data, access funds, or steal assets unexpectedly.    
# Logging into an insecure Wi-Fi is enough to expose your personal information to nearby scam artists preying on the unsuspecting.    
# But it can also happen in the privacy of your home or at your place of business.      
# Having a way to detect fraud before it happens is critical to prevent becoming another victim of a cybercrime.      
# This means losing thousands, or even millions, of dollars for some businesses.**
# 

# # <center><a id="1">Data</a></center>  
# 
# This is a simulated credit card transaction dataset containing legitimate and fraud transactions from the duration 
# **1st Jan 2019 - 31st Dec 2020.**.  
# It covers credit cards of **1000 customers** doing transactions with a pool of **800 merchants.**
# so lets take a look on the data
# 
# 
# | Attribute | Representation |
# | --- | --- |
# | Unnamed:0 | An index column, possibly representing a unique identifier for each row |
# | trans_date_trans_time | The date and time of the transaction |
# | cc_num | The credit card number used for the transaction |
# | merchant | The name of the merchant where the transaction took place |
# | category | The category of the transaction (e.g., grocery, travel, etc.) |
# | amt | The amount of the transaction |
# | first | The first name of the cardholder |
# | last | The last name of the cardholder |
# | gender | The gender of the cardholder |
# | street | The street address of the cardholder |
# | city | The city of the cardholder |
# | state | The state of the cardholder |
# | zip | The zip code of the cardholder |
# | lat | The latitude of the cardholder's location |
# | long | The longitude of the cardholder's location |
# | city_pop | The population of the city where the cardholder resides |
# | job | The occupation or job title of the cardholder |
# | dob | The date of birth of the cardholder |
# | trans_num | A unique identifier for each transaction |
# | unix_time | The transaction time in UNIX timestamp format |
# | merch_lat | The latitude of the merchant's location |
# | merch_long | The longitude of the merchant's location |
# | is_fraud | A binary indicator (1 or 0) indicating whether the transaction is fraudulent or not |

# ### Acknowledgements
# <b> Dataset was obtained from </b> : <a>https://www.kaggle.com/datasets/kartik2112/fraud-detection</a>      
# <b> Forbes study mentioned </b> : <a>https://www.forbes.com/advisor/credit-cards/most-scammed-states-in-america/#:~:text=New%20York,-New%20York's%20score&text=New%20York%20secured%20the%20fourth,258%20reports%20per%20100%2C000%20residents.</a>
# 
# 

# ## Libraries

# In[60]:


import pandas as pd
import numpy as np
from datetime import datetime, date

# preprocessing
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedShuffleSplit


# plotting 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# preprocessing
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve, auc

# save and load models
import joblib
from tensorflow.keras.models import load_model
# Models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve, auc

# save and load models
import joblib
from tensorflow.keras.models import load_model


# ### Data Extraction & Loading 

# In[2]:


df = pd.read_csv("fraud_dataset_labelled/fraudTrain.csv")
test_data = pd.read_csv("fraud_dataset_labelled/fraudTest.csv")


# In[3]:


print("rows: ",df.shape[0])
print("columns: ",df.shape[1])


# In[4]:


df.head(2)


# In[5]:


df.columns


# # <center><a id="2">Data Understanding</a></center>
# - <b><a href='#4' style="text-decoration:none" >Functions</a></b>
# - <b><a href='#5' style="text-decoration:none" >EDA</a></b>  
# - <b><a href='#6' style="text-decoration:none" >Conclusion</a></b>  

# ## <a id="4">Functions</a>   

# In[6]:


def plot_featureToClass(col, classtype):
    '''
    Creates a plot of time based on the class type
    
    Args : 
    - col: column name to plot
    - classtype: class type (fraud or non fraud)
    
    Output : None
    '''
    new_df = df[col].where(df['is_fraud'] == classtype)
    new_df = new_df.dropna()
    plt.plot(new_df)
    plt.show()


# In[7]:


def handle_outlier(col, threshold=1.5):
    '''
    Function that gets maximum and minimum values for provided column
    
    Args:
        col (pandas.Series): Column name
        threshold (float): Threshold value for defining outliers
    
    Returns:
        tuple: The lower and upper bounds for outliers
    '''
    Q1, Q3 = col.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower = Q1 - (threshold * IQR)
    upper = Q3 + (threshold * IQR)
    return lower, upper


# In[8]:


def draw_boxplot(colname):
    '''
    Creates a boxplot for the specified column
    
    Args:
        colname (str): Column name
        
    Returns:
        None
    '''
    sns.boxplot(df[colname])
    plt.xlabel(colname)
    plt.ylabel('Value')
    plt.title(f'Boxplot of {colname}')
    plt.show()


# In[9]:


def boxplot_feature_vs_class(feature,dataframe):
    '''
    Creates a boxplot for every feature vs class type
    
    Args: col (str) feature name
    
    Output : Boxplot 
    '''
    sns.boxplot(x="is_fraud", y=feature, data=dataframe)
    plt.xlabel("is_fraud")
    plt.ylabel(feature)
    plt.title(f"Relationship between {feature} and is_fraud")
    plt.show()


# ## <a id="5">EDA</a>   

# **Analyze distribution of the target variable**

# In[10]:


print('0 - NonFraud :', df['is_fraud'].value_counts()[0])
print('1 - Fraud :', df['is_fraud'].value_counts()[1])


# In[11]:


# creating the dataset
data = {'0':1289169, '1':7506}
classes = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (6, 3))

colors = ['#0E2F44', '#566D7C']

# creating the bar plot
plt.bar(classes, values, color = colors,
        width = 0.4)
 
plt.xlabel("Classes")
plt.ylabel("Count")
plt.title("Count of different classes")
plt.show()


# **There are **7506 frauds** out of **1,289,169 transactions** which meabs that the data is highly unbalanced**  
# If we use this dataframe as the foundation for our prediction models and analysis, we may encounter many inaccuracies and the algorithms will likely overfit since it will "assume" that most transactions are non-fraud.

# In[12]:


# Fraud
plot_featureToClass('unix_time',1)


# In[13]:


# Fraud
plot_featureToClass('unix_time',0)


# Since the plot of the time variable does not show any distinguishable pattern or correlation between the time and the fraud cases compared to non-fraud cases,   
# It's suggested that the **time alone may not be a strong indicator for fraud detection**

# In[14]:


fig = px.histogram(df, x="unix_time", y="amt", color="is_fraud", barmode="group", hover_data=df.columns, color_discrete_sequence=['#45818e','#134f5c'])

fig.show()


# ### Observations
# 
# **It seems that the changes in the distribution of the "amt" variable over unix_time,   
# has periods of increase, drop, and increase again.**  
# The drop in the distribution starts around 80,000 seconds to 117,000 seconds from the first transaction in the plot.    
# This drop could indicate a decrease in the amount of transactions or a change in the distribution of amounts during that time period.
# 
# **It's important to note that it is difficult to determine the exact reason for this drop.**   
# It could be due to various factors, such as a change in data collection methods, outliers, or actual changes in transaction behavior.
# 
# **One surprising fact that the fraud transactions tend to have the least amounts.**    
# Typically,it's expected that fraud activities to involve higher amounts, as perpetrators try to maximize their gains. However, there are several possible explanations for this observation:
# 
# 1. **Stealthy approach:** Fraudsters might intentionally keep the transaction amounts low to avoid detection.   
#    By keeping the amounts small, they can avoid triggering alerts or suspicion from fraud detection systems.
# 
# 2. **Testing the waters:** Fraudsters might initially conduct small transactions to test the effectiveness of their fraudulent methods or to probe the vulnerabilities of the system.   
# Once they are confident in their approach, they may escalate to larger amounts.
# 
# 3. **Lower risk:** Frauds involving smaller amounts may be considered less risky for the criminals.  
#    They may believe that smaller transactions are less likely to be investigated or prosecuted, making it easier for      them to get away with their activities.
# 
# 4. **Higher volume:** Fraudsters may engage in large-scale operations involving numerous small transactions.  
#    By conducting many small frauds, they can accumulate significant gains while minimizing the risk of detection.

# ### Since we're mostly concerned with frauds will get the data with frauds only

# In[15]:


fraud = df[df['is_fraud'] == 1]


# ### Who steals more: Men or Women?

# In[16]:


plt.figure(figsize=(5, 4))
sns.countplot(data=fraud, x="gender", palette=sns.color_palette(['#0E2F44', '#566D7C']))
plt.title("Gender Counts")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()


# **It looks that both genders experience similar rates of fraudulent transactions, with men slightly surpassing women in terms of frequency.**

# ### Is there a link between age and fraud?

# In[17]:


# Get age of. the fraudster
fraud['dob'] = pd.to_datetime(fraud['dob'])
today = date.today()
fraud['age'] = today.year - fraud['dob'].dt.year


# In[18]:


plt.figure(figsize=(20, 7))
sns.countplot(data=fraud, x="age")
plt.title("Age Counts")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()


# **The majority of fraudulent transactions appear to occur among individuals in their twenties, with frequency decreasing as age increases until the forties.           
# This could be ascribed to individuals in their 40s having stable job conditions.     
# However, the curve shoots up again in the late 40s to early 60s, which could suggest layoffs or job losses during that time period.         
# Finally, the incidence of fraudulent transactions falls once more.     
# These trends could imply that numerous characteristics such as employment position, financial stability, and living circumstances influence the occurrence of fraud.**

# ### Where do frauds happen the most and what gender?

# In[19]:


plt.figure(figsize=(20, 5))
sns.barplot(data=fraud, x="category", y="amt", hue="gender", palette=sns.color_palette(['#0E2F44', '#566D7C']))
plt.title("Fraudulent Transactions by Category and Amount")
plt.xlabel("Category")
plt.ylabel("Amount")
plt.legend(title="Gender")
plt.show()


# In[20]:


# get first number in ages(object)
fraud_ages = fraud
fraud_ages['age'] = fraud_ages['age'].astype(str)
fraud_ages['1st'] = fraud_ages['age'].apply(lambda x: x[0:1])


# In[61]:


# fraud_ages['1st'] 


# In[22]:


#replace every age by it's relevant decade
fraud_ages.loc[fraud_ages['1st'] == '1', '1st'] = "Teens"
fraud_ages.loc[fraud_ages['1st'] == '2', '1st'] = "20's"
fraud_ages.loc[fraud_ages['1st'] == '3', '1st'] = "30's"
fraud_ages.loc[fraud_ages['1st'] == '4', '1st'] = "40's"
fraud_ages.loc[fraud_ages['1st'] == '5', '1st'] = "50's"
fraud_ages.loc[fraud_ages['1st'] == '6', '1st'] = "60's"
fraud_ages.loc[fraud_ages['1st'].isin(['7', '8', '9']), '1st'] = "70+"


# In[23]:


# drop original age column
fraud_ages = fraud_ages.drop('age', axis=1)


# In[24]:


# rename 1st column by age
fraud_ages.rename(columns = {'1st':'age'}, inplace = True)
fraud_ages['age'].value_counts()


# In[62]:


fig = px.histogram(fraud_ages, x="category", y="amt", color="age", barmode="group", hover_data=fraud_ages.columns, 
                   color_discrete_sequence=['#468499','#3f7689','#38697a','#315c6b','#2a4f5b','#23424c','#1c343d'])

fig.show()


# **The most popular categories for fraudulent activities are shopping_net, shopping_pos, misc_net, and grocery_pos, especially among people in their 30s, 60s, and 50s.**

# ### What year had the most frauds?

# In[26]:


# convert the 'Date' column to datetime format
fraud['trans_date_trans_time']= pd.to_datetime(fraud['trans_date_trans_time'])


# In[27]:


# make a dictionary of years and the corresponding number of frauds
years = {}
years.update(fraud['trans_date_trans_time'].dt.year.value_counts().to_dict())


# In[28]:


labels = list(years.keys())
values = list(years.values())
colors = ['#0E2F44', '#566D7C']

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3, marker=dict(colors=colors))])
fig.update_layout(
    width=500,  # Set the width of the plot
    height=500  # Set the height of the plot
)

fig.show()


# **The donut chart illustrates that there were more frauds in 2019 than in 2020.  
# It is possible that external factors such as the COVID-19 outbreak influenced the fraud rate, but it is important to note that the analysis offered is based on a hypothetical scenario, as the data provided do not match to actual data.      
# To draw definitive conclusions on the impact of COVID-19 on fraud rates, a more extensive analysis would be needed that takes into account a variety of factors such as changes in consumer behaviour, economic situations, and fraud prevention efforts implemented during the pandemic**

# ### Which state has the highest percentage of fraud?

# In[29]:


states = {}
states.update(fraud['state'].value_counts().to_dict())


# In[30]:


y = list(states.keys())
x = list(states.values())
colors = [
    '#0f4ba0', '#265da9', '#3e6eb3', '#5781bc', '#6f93c6', '#87a5cf', '#9fb7d9',
    '#b7c9e2', '#CFDBEC', '#0f4ba0', '#265da9', '#3e6eb3', '#5781bc', '#6f93c6', '#87a5cf', '#9fb7d9',
    '#b7c9e2', '#CFDBEC', '#0f4ba0', '#265da9', '#3e6eb3', '#5781bc', '#6f93c6', '#87a5cf', '#9fb7d9',
    '#b7c9e2', '#CFDBEC', '#0f4ba0', '#265da9', '#3e6eb3', '#5781bc', '#6f93c6', '#87a5cf', '#9fb7d9',
    '#b7c9e2', '#CFDBEC','#3232FF', '#4C4CFF','#0f4ba0', '#265da9', '#3e6eb3', '#5781bc', '#6f93c6', 
    '#87a5cf', '#9fb7d9','#b7c9e2', '#CFDBEC','#0f4ba0', '#265da9', '#3e6eb3', '#5781bc', '#6f93c6', 
    '#87a5cf', '#9fb7d9','#b7c9e2', '#CFDBEC']

fig = plt.figure(figsize=(5, 12))  
plt.barh(y, x, color=colors)

plt.ylabel("States")
plt.xlabel("Frequency")


plt.show()


# **This is hardly surprising given that, according to one Forbes study, New York ranked fourth in total fraud reports, with 22,688 occurrences reported.     
# The state scored 16th in terms of fraud per capita, with 258 reports per 100,000 residents, which explains why it placed first in terms of fraud frequency, despite the fact that the data is not actual, it corresponds well with reality**

# ### Which fraudster's job has the most influence on fraud?

# In[31]:


jobs = {}
jobs.update(fraud['job'].value_counts().to_dict())


# In[32]:


filtered_jobs = {x: jobs[x] for x in jobs if jobs[x] >= 30}

filtered_jobs = pd.DataFrame.from_dict(filtered_jobs, orient='index', columns=['Count'])

plt.figure(figsize=(30, 7))
sns.barplot(data=filtered_jobs, x=filtered_jobs.index, y='Count')
plt.title("Job Counts")
plt.xlabel("Job")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()


# Based on the analysis of fraud cases, it has been observed that a higher percentage of fraudsters were identified in the job roles of **Material engineers, Trading standards officers, and Naval architects**    
# There may be certain factors associated with these job roles that contribute to a higher likelihood of involvement in fraudulent activities.            
# Access to sensitive information, work duties, and other contextual components are examples of such circumstances.     
# 
# More research is needed to acquire a better understanding of these potential relationships.             
# **Please note that this example is just for illustrative purposes, it is fake and based on a hypothetical scenario, thus it should not be used as a reference.**
# 
# 

# ## <a id="6">Conclusion</a>   

# **In conclusion, the data shows that fraudulent actions in New York City in and in 2019 were not particularly gender-biased, with men and women participating at equal rates.             
# Furthermore, individuals with the occupational titles of Material engineers, Trading standards officials, and Naval architects were more commonly involved with fraudulent operations.             
# Shopping_net, shopping_pos, misc_net, and grocery_pos were the most common categories for fraudulent activities, with people in their 30s, 60s, and 50s being more likely to participate in such activities.**

# # <center><a id="3">Data preparation</a><center> 

# ## Training Data

# In[ ]:


df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], infer_datetime_format=True)
df["dob"] = pd.to_datetime(df["dob"], infer_datetime_format=True)


# <h3> Note:  Data has no nulls</h3>

# In[63]:


df.isnull().sum()


# <h3>Duplicates</h3>

# In[34]:


if df.duplicated().sum() > 0:
    df.drop_duplicates(inplace=True)
    print('Duplicates dropped')
else:
    print('No Duplicates Exist')


# <h3> Data Encoding </h3>

# In[35]:


# Encode categorical data to numerical
label_encoder = preprocessing.LabelEncoder()
for col in df.columns:
    df[col]= label_encoder.fit_transform(df[col])


# <h3> Data Scaling </h3>

# In[36]:


# Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)
# RobustScaler is less prone to outliers.
rob_scaler = RobustScaler()

for col in df.columns:
    if df[col].dtype =='object':
        df[col] = rob_scaler.fit_transform(df[col].values.reshape(-1,1))
    else:
        pass


# <h3> Outliers </h3>

# In[37]:


for col in df.columns:
    if col == 'is_fraud':
        pass
    else:
        boxplot_feature_vs_class(col,df)


# **wow, no outliers except for the extreme values in merch_lat and merch_long that fall outside the whiskers of the boxplot, which will be handled**

# In[38]:


columns_to_plot = [col for col in df.columns if col in ['merch_lat' , 'merch_long']]

for i, col in enumerate(columns_to_plot):
    # Get lower and upper bounds for outliers with a higher threshold (e.g., 3.0)
    minimum, maximum = handle_outlier(df[col], threshold=1.5)
    # Replace values above the upper bound with the maximum value and values below the lower bound with the minimum value
    df[col] = np.where(df[col] > maximum, maximum, df[col])
    df[col] = np.where(df[col] < minimum, minimum, df[col])
    draw_boxplot(col)
    plt.tight_layout()
    plt.show()


# <h3>Data Splitting</h3>

# **Stratified sampling is useful when the data contains imbalanced class distributions. 
# When partitioning the data into training and testing sets, stratified sampling guarantees that the class proportions are preserved in both sets.    
# This is significant since it helps to avoid biassed or skewed judgements when working with skewed data**
# 
# **Benefits of stratified sampling:**
# 
# 1. **Maintain Class Distribution:** Maintaining the class distribution in both the training and testing sets guarantees that the model is trained and evaluated on representative samples from each class. This is particularly crucial when dealing with unbalanced datasets, where minority classes may be underrepresented, which is the situation here.
# 
# 2. **Improved Generalisation:** Stratified sampling helps in the generalisation of the model.   By ensuring that the class proportions in both sets are almost exactly the same, your model learns from a varied range of examples, resulting in improved performance on unseen data.
# 
# 3. **Accurate Evaluation:** Using stratified sampling for assessing the performance of the model guarantees that the evaluation measures, such as accuracy, precision, recall, and F1-score, are not biassed by the uneven class distribution.  This enables a fair and accurate evaluation of your model's performance.
# 

# In[39]:


X = df.drop(['is_fraud'],axis = 1)
y = df['is_fraud']


# In[40]:


stratified_split = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
for train_index, test_index in stratified_split.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# <center><h3>SMOTE Technique (Over-Sampling):</h3></center>
# <br>
# <center><img src="smote_oversampling.png" width="600"></img></center>
# 
# SMOTE is an algorithm used to address class imbalance in a dataset, where one class has significantly fewer samples than the other.
# 
# #### How does SMOTE work :
# - **Class Imbalance Amendment:** SMOTE generates synthetic points from the minority class in order to achieve an equitable balance between the minority and majority classes.
# - **Synthetic point location:** SMOTE chooses the distance between the minority class's nearest neighbours and constructs synthetic points between these distances.
# - **Impact:** More information is maintained since no rows were deleted, as in random undersampling.
# 

# In[41]:


smote = SMOTE(random_state=42)
# Apply SMOTE to oversample the minority class
X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train, y_train)


# # <center><a id="7">Modelling</a></center> 
# - <h3><a href='#8' style="text-decoration:none" >Logistic Regression</a></h3>
# - <h3><a href='#9' style="text-decoration:none" >Random Forest</a></h3>
# 

# <h3><a id='8'>Logistic Regression</a></h3>

# In[45]:


# Instantiate & train the model
lg_model = LogisticRegression(random_state=42)
lg_model.fit(X_train_oversampled, y_train_oversampled)


# In[46]:


lg_ypred = lg_model.predict(X_test)

# Evaluate performance
print("Accuracy score is: ",round(accuracy_score(y_test, lg_ypred)*100,2),'%')


# In[47]:


# Evaluate performance
print(classification_report(y_test, lg_ypred))
print("ROC-AUC:", roc_auc_score(y_test, lg_ypred))


# In[48]:


# Create confusion matrix
lgconfusion_matrix = confusion_matrix(y_test, lg_ypred)


# In[49]:


# Plot the confusion matrix heatmap
plt.figure(figsize=(7, 5))
plt.title('LG Confusion Matrix')
sns.heatmap(lgconfusion_matrix, annot=True, xticklabels=["Not-Fraud","Fraud"],
            yticklabels=["Not-Fraud","Fraud"], fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Correct Label')
plt.show()


# The principles of sensitivty/recall and precision are crucial for evaluating fraud detection algorithms.   
# **Recall** measures how many fraud instances go unnoticed, 
# **while precision**  measures how good the model is at producing as few false predictions as possible.     
# It is essential to prioritise high recall for fraud detection in order to miss as few fraud instances as possible while simultaneously having a relatively high accuracy because too many false predictions might be an issue! 
# The recall in the Logistic Regression model above is approximately 0.77, and the overall accuracy is around 0.92, which is less than the actual percentage of non-fraud instances in the testing dataset.     
# 
# Both measures appear to indicate that,  
# **the model does not help much in detecting fraud     
# furthermore, the model's Precision is just 0.06 too many false positives to be actually helpful!**
# 
# **Another approach, Random Forest, will now be tested to determine whether it may produce better results**

# <h3><a id='9'> Random Forest </a></h3>

# In[51]:


# Instantiate & train the model
rf_model= RandomForestClassifier()
rf_model.fit(X_train_oversampled, y_train_oversampled)


# In[52]:


# Test the model
rf_ypred = rf_model.predict(X_test)

# Evaluate performance
print("Acuuracy score is: ",round(accuracy_score(y_test, rf_ypred)*100,2),'%')


# In[53]:


# Evaluate performance
print(classification_report(y_test, rf_ypred))
print("ROC-AUC:", roc_auc_score(y_test, rf_ypred))


# In[54]:


# Create confusion matrix
rf_confusion_matrix = confusion_matrix(y_test, rf_ypred)


# In[56]:


# Plot the confusion matrix heatmap
plt.figure(figsize=(7, 5))
plt.title('RF Confusion Matrix')
sns.heatmap(rf_confusion_matrix, annot=True, xticklabels=["Not-Fraud","Fraud"],
            yticklabels=["Not-Fraud","Fraud"], fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Correct Label')
plt.show()


# **The metrics above demonstrate that, while the Random Forest model has a somewhat lower recall, it has quite higher accuracy and precision!              
# While the Random Forest model can provide a more solid foundation for future deployment, it still has to be improved, particularly in terms of recall.              
# This may entail optimising model hyperparameters, experimenting with alternative feature engineering techniques, or employing more complex ensemble methods.               
# Overall, the Random Forest model is an excellent starting point for fraud detection, but it requires more refinement to guarantee that it successfully captures all fraudulent situations while retaining high accuracy and precision.**

# # <center><a id="11">Saved Models</a></center> 

# In[58]:


# Save the Logistic Regression model
joblib.dump(lg_model, 'logistic_regression_model.pkl')


# In[59]:


# Save the Random Forest model
joblib.dump(rf_model, 'random_forest_model.pkl')


# # <center><a id="12">Load & Use Models</a></center> 

# ### Prepare test data

# In[64]:


test_data["trans_date_trans_time"] = pd.to_datetime(test_data["trans_date_trans_time"], infer_datetime_format=True)
test_data["dob"] = pd.to_datetime(test_data["dob"], infer_datetime_format=True)


# In[65]:


test_data.isnull().sum()


# In[66]:


if test_data.duplicated().sum() > 0:
    test_data.drop_duplicates(inplace=True)
    print('Duplicates dropped')
else:
    print('No Duplicates Exist')


# In[67]:


# Encode categorical data to numerical
for col in test_data.columns:
    test_data[col]= label_encoder.fit_transform(test_data[col])


# In[68]:


# Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)
# RobustScaler is less prone to outliers.

for col in test_data.columns:
    if test_data[col].dtype =='object':
        test_data[col] = rob_scaler.fit_transform(test_data[col].values.reshape(-1,1))
    else:
        pass


# In[69]:


X_test = test_data.drop(['is_fraud'],axis = 1)
y_test = test_data['is_fraud']


# ### Load and predict

# In[71]:


# Load the Random Forest model
rf_model = joblib.load('random_forest_model.pkl')

# Load the Logistic Regression model
lr_model = joblib.load('logistic_regression_model.pkl')


# In[72]:


# Make predictions using the Random Forest model
rf_predictions = rf_model.predict(X_test)

# Make predictions using the Logistic Regression model
lr_predictions = lr_model.predict(X_test)


# In[73]:


# Evaluate performance
print("RF", classification_report(y_test, rf_predictions))


# In[75]:


print("LR", classification_report(y_test, lr_predictions))


# ### In a nutshell     
# Both models predict non-fraudulent occurrences (class 0) with excellent accuracy but struggle to detect fraudulent cases (class 1).       
# The RF model performs better in terms of overall accuracy and precision for class 0, but worse in terms of recall and F1-score for class 1.      
# The LR model has lesser accuracy and precision for class 0, but higher recall and F1-score for class 1.

# # <center><a id="13">Conclusion & Room for improvement</a></center> 

# ## Conclusion

# **As previously stated, this data is not real and made up only for the sake
# of modeling and analysis purposes. 
# It is crucial to emphasise that employing made-up or synthetic data for fraud detection is not suggested for real-world applications and <u><b>SHOULD NOT</b></u> be used as a reference.     
# To ensure the performance of the models and to minimise false positives or false negatives, real-world fraud detection systems require precise and trustworthy data.     
# While the research and modelling processes can give insights and a framework for constructing fraud detection systems, real-world data must be used to validate and test the models.      
# Furthermore, domain expertise, as well as continual monitoring and updating of the models, will be necessary to assure their accuracy and flexibility to changing fraud patterns.**

# ## Room for improvement

# **It's worth mentioning that there is always room for improvement when it comes to fraud detection.          
# There are various models that can be used to identify fraud, each with its own set of strengths and weaknesses. Here are a couple of examples:**
# 
# **1. Support Vector Machines (SVM):**           
#    **- Pros:** Effective in separating classes, handles non-linear relationships.              
#    **- Cons:** Computationally expensive, less interpretable.
# 
# **2. Neural Networks:**            
#    **- Pros:** Captures complex patterns, can handle high-dimensional data.            
#    **- Cons:** Requires more data and computational resources, less interpretable.         
#         
# **3. XGBoost:**               
#    **- Pros:** High performance, handles missing values, provides feature importance.             
#    **- Cons:** Computationally expensive, requires tuning.
# 
# **4. Random Forest:**                
#    **- Pros:** Handles high-dimensional data, robust to outliers, good performance.               
#    **- Cons:** Less interpretable, computationally expensive.
# 
# These models provide varied trade-offs between performance, interpretability, and computational needs.    
# The model selected depends on the unique needs and restrictions of the fraud detection task.     
# It is essential to experiment with various models, assess their performance on real-world data, and consider the application's interpretability requirements.
# 
# Furthermore, ensemble approaches like ***XGBoost and Random Forest*** may significantly improve performance by mixing many models.      
# They can also give insights regarding feature significance, which can benefit in identifying the elements that contribute to fraud.         
# 
# Finally, choosing the best model for fraud detection requires careful evaluation of the unique context, accessible data, interpretability criteria, and computing resources.

# In[ ]:





# In[ ]:





# In[ ]:




