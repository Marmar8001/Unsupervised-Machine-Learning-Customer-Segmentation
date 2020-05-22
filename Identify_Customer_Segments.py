#!/usr/bin/env python
# coding: utf-8

# # Project: Identify Customer Segments
# 
# In this project, I will apply unsupervised learning techniques to identify segments of the population that form the core customer base for a mail-order sales company in Germany. These segments can then be used to direct marketing campaigns towards audiences that will have the highest expected rate of returns. The data that I will use has been provided by the partners at Bertelsmann Arvato Analytics, and represents a real-life data science task.
# 

# In[1]:


# import libraries here; add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# magic word for producing visualizations in notebook
get_ipython().run_line_magic('matplotlib', 'inline')

'''
Import note: The classroom currently uses sklearn version 0.19.
If you need to use an imputer, it is available in sklearn.preprocessing.Imputer,
instead of sklearn.impute as in newer versions of sklearn.
'''


# ### Step 0: Load the Data
# 
# There are four files associated with this project (not including this one):
# 
# - `Udacity_AZDIAS_Subset.csv`: Demographics data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
# - `Udacity_CUSTOMERS_Subset.csv`: Demographics data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
# - `Data_Dictionary.md`: Detailed information file about the features in the provided datasets.
# - `AZDIAS_Feature_Summary.csv`: Summary of feature attributes for demographics data; 85 features (rows) x 4 columns
# 
# Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood. You will use this information to cluster the general population into groups with similar demographic properties. Then, you will see how the people in the customers dataset fit into those created clusters. The hope here is that certain clusters are over-represented in the customers data, as compared to the general population; those over-represented clusters will be assumed to be part of the core userbase. This information can then be used for further applications, such as targeting for a marketing campaign.
# 
# To start off with, load in the demographics data for the general population into a pandas DataFrame, and do the same for the feature attributes summary. Note for all of the `.csv` data files in this project: they're semicolon (`;`) delimited, so you'll need an additional argument in your [`read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) call to read in the data properly. Also, considering the size of the main dataset, it may take some time for it to load completely.
# 
# Once the dataset is loaded, it's recommended that you take a little bit of time just browsing the general structure of the dataset and feature summary file. You'll be getting deep into the innards of the cleaning in the first major step of the project, so gaining some general familiarity can help you get your bearings.

# In[2]:


# Load in the general demographics data.
azdias = pd.read_csv('Udacity_AZDIAS_Subset.csv', sep=';')

# Load in the feature summary file.
feat_info = pd.read_csv('AZDIAS_Feature_Summary.csv', sep=';')


# In[3]:


# Check the structure of the data after it's loaded (e.g. print the number of
# rows and columns, print the first few rows).
print(azdias.shape)

azdias.head(5)


# In[4]:


print(feat_info.shape)
feat_info.head(5)


# 
# ## Step 1: Preprocessing
# 
# ### Step 1.1: Assess Missing Data
# 
# The feature summary file contains a summary of properties for each demographics data column. You will use this file to help you make cleaning decisions during this stage of the project. First of all, you should assess the demographics data in terms of missing data. Pay attention to the following points as you perform your analysis, and take notes on what you observe. Make sure that you fill in the **Discussion** cell with your findings and decisions at the end of each step that has one!
# 
# #### Step 1.1.1: Convert Missing Value Codes to NaNs
# The fourth column of the feature attributes summary (loaded in above as `feat_info`) documents the codes from the data dictionary that indicate missing or unknown data. While the file encodes this as a list (e.g. `[-1,0]`), this will get read in as a string object. You'll need to do a little bit of parsing to make use of it to identify and clean the data. Convert data that matches a 'missing' or 'unknown' value code into a numpy NaN value. You might want to see how much data takes on a 'missing' or 'unknown' code, and how much data is naturally missing, as a point of interest.
# 
# **As one more reminder, you are encouraged to add additional cells to break up your analysis into manageable chunks.**

# In[5]:


# Identify missing or unknown data values and convert them to NaNs.

for index in feat_info.index:
    feat_info['missing_or_unknown'][index]=feat_info['missing_or_unknown'][index].strip('[')
    feat_info['missing_or_unknown'][index]=feat_info['missing_or_unknown'][index].strip(']')
    feat_info['missing_or_unknown'][index]=feat_info['missing_or_unknown'][index].split(',')
    
    for item in range(len(feat_info['missing_or_unknown'][index])):
        try:
            feat_info['missing_or_unknown'][index][item]=int(feat_info['missing_or_unknown'][index][item])
        except:
            continue
            
    column_name=feat_info['attribute'][index]
    azdias.loc[:,feat_info['attribute'][index]].replace(feat_info['missing_or_unknown'][index], np.nan, inplace=True)
    
azdias.head(5)


# In[6]:


missing_values=azdias.isnull().sum()


# #### Step 1.1.2: Assess Missing Data in Each Column
# 
# How much missing data is present in each column? There are a few columns that are outliers in terms of the proportion of values that are missing. You will want to use matplotlib's [`hist()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html) function to visualize the distribution of missing value counts to find these columns. Identify and document these columns. While some of these columns might have justifications for keeping or re-encoding the data, for this project you should just remove them from the dataframe. (Feel free to make remarks about these outlier columns in the discussion, however!)
# 
# For the remaining features, are there any patterns in which columns have, or share, missing data?

# In[7]:


# Perform an assessment of how much missing data there is in each column of the
# dataset.

plt.hist(missing_values);
plt.xlabel('missing values')
plt.yticks([0,5,10,15,20,25,30,35,40]);


# In[8]:


# Investigate patterns in the amount of missing data in each column.

outliers=missing_values[missing_values>200000]
print(outliers)


# In[9]:


azdias.shape


# In[10]:


# Remove the outlier columns from the dataset. (You'll perform other data
# engineering tasks such as re-encoding and imputation later.)

azdias.drop(outliers.index, axis=1, inplace=True)
azdias.shape


# #### Discussion 1.1.2: Assess Missing Data in Each Column
# After replacing all missing values with Nan, the columns evaluated.
# As you saw in above data, most of missing values are in 6 columns. All these columns have missing values more than 200000. For this reason all these columns dropped from dataframe.

# #### Step 1.1.3: Assess Missing Data in Each Row
# 
# Now, you'll perform a similar assessment for the rows of the dataset. How much data is missing in each row? As with the columns, you should see some groups of points that have a very different numbers of missing values. Divide the data into two subsets: one for data points that are above some threshold for missing values, and a second subset for points below that threshold.
# 
# In order to know what to do with the outlier rows, we should see if the distribution of data values on columns that are not missing data (or are missing very little data) are similar or different between the two groups. Select at least five of these columns and compare the distribution of values.
# - You can use seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) function to create a bar chart of code frequencies and matplotlib's [`subplot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html) function to put bar charts for the two subplots side by side.
# - To reduce repeated code, you might want to write a function that can perform this comparison, taking as one of its arguments a column to be compared.
# 
# Depending on what you observe in your comparison, this will have implications on how you approach your conclusions later in the analysis. If the distributions of non-missing features look similar between the data with many missing values and the data with few or no missing values, then we could argue that simply dropping those points from the analysis won't present a major issue. On the other hand, if the data with many missing values looks very different from the data with few or no missing values, then we should make a note on those data as special. We'll revisit these data later on. **Either way, you should continue your analysis for now using just the subset of the data with few or no missing values.**

# In[11]:


# How much data is missing in each row of the dataset?
missing_row=azdias.isnull().sum(axis=1)
plt.hist(missing_row);
plt.xlabel("missing values in each row")
plt.ylabel('Row index')



# In[12]:


# Write code to divide the data into two subsets based on the number of missing
# values in each row.

azdias_high=azdias[missing_row >20]
azdias_low=azdias[missing_row<20]

missing_row_high=azdias_high.isnull().sum(axis=1)
missing_row_low=azdias_low.isnull().sum(axis=1)
plt.hist(missing_row_low);    
plt.hist(missing_row_high);    


# In[13]:


# Compare the distribution of values for at least five columns where there are
# no or few missing values, between the two subsets.


fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(20,30))
n=0
for column in azdias.columns[:7]:
    sns.countplot(azdias_low.loc[:,column],ax=axes[n,0])
    sns.countplot(azdias_high.loc[:,column],ax=axes[n,1])
    n=n+1
        


# #### Discussion 1.1.3: Assess Missing Data in Each Row
# 
# * The data divided to two sets based on the quantity of missing values ( above and below 20). After comparing two sets in several columns in countplots, it seems the trends are not the same in two sets of data. To make sure to deal with minimum unknown values, for the rest of the work, I use the dataset with null values below 20 in each row.

# ### Step 1.2: Select and Re-Encode Features
# 
# Checking for missing data isn't the only way in which you can prepare a dataset for analysis. Since the unsupervised learning techniques to be used will only work on data that is encoded numerically, you need to make a few encoding changes or additional assumptions to be able to make progress. In addition, while almost all of the values in the dataset are encoded using numbers, not all of them represent numeric values. Check the third column of the feature summary (`feat_info`) for a summary of types of measurement.
# - For numeric and interval data, these features can be kept without changes.
# - Most of the variables in the dataset are ordinal in nature. While ordinal values may technically be non-linear in spacing, make the simplifying assumption that the ordinal variables can be treated as being interval in nature (that is, kept without any changes).
# - Special handling may be necessary for the remaining two variable types: categorical, and 'mixed'.
# 
# In the first two parts of this sub-step, you will perform an investigation of the categorical and mixed-type features and make a decision on each of them, whether you will keep, drop, or re-encode each. Then, in the last part, you will create a new data frame with only the selected and engineered columns.
# 
# Data wrangling is often the trickiest part of the data analysis process, and there's a lot of it to be done here. But stick with it: once you're done with this step, you'll be ready to get to the machine learning parts of the project!

# In[14]:


# How many features are there of each data type?

feat_info.groupby('type').count()['attribute']


# #### Step 1.2.1: Re-Encode Categorical Features
# 
# For categorical data, you would ordinarily need to encode the levels as dummy variables. Depending on the number of categories, perform one of the following:
# - For binary (two-level) categoricals that take numeric values, you can keep them without needing to do anything.
# - There is one binary variable that takes on non-numeric values. For this one, you need to re-encode the values as numbers or create a dummy variable.
# - For multi-level categoricals (three or more values), you can choose to encode the values using multiple dummy variables (e.g. via [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)), or (to keep things straightforward) just drop them from the analysis. As always, document your choices in the Discussion section.

# In[15]:


# Assess categorical variables: which are binary, which are multi-level, and
# which one needs to be re-encoded?
categorical=feat_info.query('type=="categorical"')
categorical_list=list(categorical['attribute'])
categorical=[x for x in categorical_list if x in azdias_low.columns]
binary=[x for x in categorical if azdias_low[x].nunique()==2]
multi_level=[x for x in categorical if azdias_low[x].nunique()>2]
multi_level


# In[16]:


# Re-encode categorical variable(s) to be kept in the analysis.
for x in binary:
    print("for {} variable, the variable type is {}".format(x,azdias[x].dtypes))


# In[17]:


#encode object type variable
azdias_low['OST_WEST_KZ'] = azdias_low['OST_WEST_KZ'].apply({'W':0, 'O':1}.get)
azdias_low['OST_WEST_KZ'].value_counts()


# In[18]:



azdias_low[categorical].nunique()


# In[19]:


# Remove multi-level variables from dataset
for col in multi_level:
    azdias_low.drop(col, axis=1, inplace=True)
    
    


# In[20]:


# finding columns in dataset
azdias_low.columns


# #### Discussion 1.2.1: Re-Encode Categorical Features
# 
# In binary data, I encoded W and O with 0 and 1 in "OST_WEST_KZ" column in dataset. In multilevel data, I removed multilevel columns from dataset for simplification

# #### Step 1.2.2: Engineer Mixed-Type Features
# 
# There are a handful of features that are marked as "mixed" in the feature summary that require special treatment in order to be included in the analysis. There are two in particular that deserve attention; the handling of the rest are up to your own choices:
# - "PRAEGENDE_JUGENDJAHRE" combines information on three dimensions: generation by decade, movement (mainstream vs. avantgarde), and nation (east vs. west). While there aren't enough levels to disentangle east from west, you should create two new variables to capture the other two dimensions: an interval-type variable for decade, and a binary variable for movement.
# - "CAMEO_INTL_2015" combines information on two axes: wealth and life stage. Break up the two-digit codes by their 'tens'-place and 'ones'-place digits into two new ordinal variables (which, for the purposes of this project, is equivalent to just treating them as their raw numeric values).
# - If you decide to keep or engineer new features around the other mixed-type features, make sure you note your steps in the Discussion section.
# 
# Be sure to check `Data_Dictionary.md` for the details needed to finish these tasks.

# In[21]:


# Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.

Mainstream=[1,3,5,8,10,12,14]
Avantgarde=[2,4,6,7,9,11,13,15]
def movement(i):
    if i in Mainstream:
        return 0
    elif i in Avantgarde:
        return 1
    else:
        return i
azdias_low.loc[:,'PRAEGENDE_JUGENDJAHRE']=azdias_low.loc[:,'PRAEGENDE_JUGENDJAHRE'].apply(lambda x: movement(x))

x=[x+1 for x in range(15)]
y=[40,40,50,50,60,60,60,70,70,80,80,80,80,90,90]
decade=pd.Series(y, index=x)

for i in range(15):
    azdias_low['PRAEGENDE_JUGENDJAHRE_decade']=azdias_low['PRAEGENDE_JUGENDJAHRE'].map(decade)


# In[22]:


# Investigate "CAMEO_INTL_2015" and engineer two new variables.
no_null=azdias_low["CAMEO_INTL_2015"][azdias_low["CAMEO_INTL_2015"].notnull()]
azdias_low["CAMEO_INTL_2015_Wealth"]=no_null.apply (lambda x: int(str(x)[0]))
azdias_low["CAMEO_INTL_2015_LifeStage"]=no_null.map(lambda x: int(str(x)[1]))


# In[23]:


mixed=feat_info.query('type=="mixed"')
mixed_list=list(mixed['attribute'])
mixed=[x for x in mixed_list if x in azdias_low.columns]
azdias_low[mixed].nunique()


# In[24]:


# drop columns with mixed values for simplicity
azdias_low.drop(mixed , axis=1, inplace=True)


# In[25]:


azdias_low.columns


# In[26]:


azdias_low.shape


# #### Discussion 1.2.2: Engineer Mixed-Type Features
# 
# * For "PRAEGENDE_JUGENDJAHRE" column, the column divided to two columns. One column was encoded to 0 and 1 based on the movement. the second column was based on the decade timing.
# * For "CAMEO_INTL_2015" column, the column divided to two columns based on wealth and life stage. For wealth, the data encoded to the tens place and for life stage data encoded to the ones place.
# * 'LP_LEBENSPHASE_FEIN'column was dropped because of having alot of encoded variables.

# #### Step 1.2.3: Complete Feature Selection
# 
# In order to finish this step up, you need to make sure that your data frame now only has the columns that you want to keep. To summarize, the dataframe should consist of the following:
# - All numeric, interval, and ordinal type columns from the original dataset.
# - Binary categorical features (all numerically-encoded).
# - Engineered features from other multi-level categorical features and mixed features.
# 
# Make sure that for any new columns that you have engineered, that you've excluded the original columns from the final dataset. Otherwise, their values will interfere with the analysis later on the project. For example, you should not keep "PRAEGENDE_JUGENDJAHRE", since its values won't be useful for the algorithm: only the values derived from it in the engineered features you created should be retained. As a reminder, your data should only be from **the subset with few or no missing values**.

# ### Step 1.3: Create a Cleaning Function
# 
# Even though you've finished cleaning up the general population demographics data, it's important to look ahead to the future and realize that you'll need to perform the same cleaning steps on the customer demographics data. In this substep, complete the function below to execute the main feature selection, encoding, and re-engineering steps you performed above. Then, when it comes to looking at the customer data in Step 3, you can just run this function on that DataFrame to get the trimmed dataset in a single step.

# In[27]:


def clean_data(df):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    
    # Put in code here to execute all main cleaning steps:
    # convert missing value codes into NaNs, ...

    feat_info = pd.read_csv('AZDIAS_Feature_Summary.csv', sep=';')
    
    for index in feat_info.index:
        feat_info['missing_or_unknown'][index]=feat_info['missing_or_unknown'][index].strip('[')
        feat_info['missing_or_unknown'][index]=feat_info['missing_or_unknown'][index].strip(']')
        feat_info['missing_or_unknown'][index]=feat_info['missing_or_unknown'][index].split(',')
        
        for item in range(len(feat_info['missing_or_unknown'][index])):
            try:
                feat_info['missing_or_unknown'][index][item]=int(feat_info['missing_or_unknown'][index][item])
            except:
                continue
            
        column_name=feat_info['attribute'][index]
        df.loc[:,feat_info['attribute'][index]].replace(feat_info['missing_or_unknown'][index], np.nan, inplace=True)
    
    # remove selected columns and rows, ...
    missing_values=df.isnull().sum()
    outliers=missing_values[missing_values>200000]
    df.drop(outliers.index, axis=1, inplace=True)
    missing_row=df.isnull().sum(axis=1)

    df_clean=df[missing_row<20]



    
    # select, re-encode, and engineer column values.
    categorical=feat_info.query('type=="categorical"')
    categorical_list=list(categorical['attribute'])
    categorical=[x for x in categorical_list if x in df_clean.columns]
    binary=[x for x in categorical if df_clean[x].nunique()==2]
    multi_level=[x for x in categorical if df_clean[x].nunique()>2]
    df_clean['OST_WEST_KZ'] = df_clean['OST_WEST_KZ'].apply({'W':0, 'O':1}.get)
    
    for col in multi_level:
        df_clean.drop(col, axis=1, inplace=True)
    
    Mainstream=[1,3,5,8,10,12,14]
    Avantgarde=[2,4,6,7,9,11,13,15]
    
    def movement(i):
        if i in Mainstream:
            return 0
        elif i in Avantgarde:
            return 1
        else:
            return i
    df_clean.loc[:,'PRAEGENDE_JUGENDJAHRE']=df_clean.loc[:,'PRAEGENDE_JUGENDJAHRE'].apply(lambda x: movement(x))

    x=[x+1 for x in range(15)]
    y=[40,40,50,50,60,60,60,70,70,80,80,80,80,90,90]
    decade=pd.Series(y, index=x)

    for i in range(15):
        df_clean['PRAEGENDE_JUGENDJAHRE_decade']=df_clean['PRAEGENDE_JUGENDJAHRE'].map(decade)
        
    mixed=feat_info.query('type=="mixed"')
    mixed_list=list(mixed['attribute'])
    mixed=[x for x in mixed_list if x in df_clean.columns]
    df_clean.drop(mixed , axis=1, inplace=True)

    # Return the cleaned dataframe.
    return df_clean
    
    


# ## Step 2: Feature Transformation
# 
# ### Step 2.1: Apply Feature Scaling
# 
# Before we apply dimensionality reduction techniques to the data, we need to perform feature scaling so that the principal component vectors are not influenced by the natural differences in scale for features. Starting from this part of the project, you'll want to keep an eye on the [API reference page for sklearn](http://scikit-learn.org/stable/modules/classes.html) to help you navigate to all of the classes and functions that you'll need. In this substep, you'll need to check the following:
# 
# - sklearn requires that data not have missing values in order for its estimators to work properly. So, before applying the scaler to your data, make sure that you've cleaned the DataFrame of the remaining missing values. This can be as simple as just removing all data points with missing data, or applying an [Imputer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html) to replace all missing values. You might also try a more complicated procedure where you temporarily remove missing values in order to compute the scaling parameters before re-introducing those missing values and applying imputation. Think about how much missing data you have and what possible effects each approach might have on your analysis, and justify your decision in the discussion section below.
# - For the actual scaling function, a [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) instance is suggested, scaling each feature to mean 0 and standard deviation 1.
# - For these classes, you can make use of the `.fit_transform()` method to both fit a procedure to the data as well as apply the transformation to the data at the same time. Don't forget to keep the fit sklearn objects handy, since you'll be applying them to the customer demographics data towards the end of the project.

# In[28]:


# If you've not yet cleaned the dataset of all NaN values, then investigate and
# do that now.
imputer=Imputer()
azdias_low=pd.DataFrame(imputer.fit_transform(azdias_low), columns=azdias_low.columns)


# In[29]:


# Apply feature scaling to the general population demographics data.
stdscaler=StandardScaler()
azdias_low=pd.DataFrame(stdscaler.fit_transform(azdias_low), columns=azdias_low.columns)


# ### Discussion 2.1: Apply Feature Scaling
# 
# * Imputation replace the nan values with mean by default.
# * Standard Scaler changes the data scale to 0-1 for better handling.

# ### Step 2.2: Perform Dimensionality Reduction
# 
# On your scaled data, you are now ready to apply dimensionality reduction techniques.
# 
# - Use sklearn's [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) class to apply principal component analysis on the data, thus finding the vectors of maximal variance in the data. To start, you should not set any parameters (so all components are computed) or set a number of components that is at least half the number of features (so there's enough features to see the general trend in variability).
# - Check out the ratio of variance explained by each principal component as well as the cumulative variance explained. Try plotting the cumulative or sequential values using matplotlib's [`plot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html) function. Based on what you find, select a value for the number of transformed features you'll retain for the clustering part of the project.
# - Once you've made a choice for the number of components to keep, make sure you re-fit a PCA instance to perform the decided-on transformation.

# In[30]:


# Apply PCA to the data.
pca=PCA()
x_pca=pca.fit_transform(azdias_low)


# In[31]:


# Investigate the variance accounted for by each principal component.
def scree_plot(pca):
    num_components = len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
 
    plt.figure(figsize=(30, 20))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    for i in range(num_components):
        ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=12)
 
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)
 
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')
scree_plot(pca)
    


# In[32]:


# Re-apply PCA to the data while selecting for number of components to retain.
pca=PCA(n_components=15)
x_pca=pca.fit_transform(azdias_low)


# In[33]:


pca.explained_variance_ratio_.sum()


# ### Discussion 2.2: Perform Dimensionality Reduction
# 
# By having 10 components, we can reach to more than 60% of the cumulative variance . So, the PCA model modified to 10 components.

# ### Step 2.3: Interpret Principal Components
# 
# Now that we have our transformed principal components, it's a nice idea to check out the weight of each variable on the first few components to see if they can be interpreted in some fashion.
# 
# As a reminder, each principal component is a unit vector that points in the direction of highest variance (after accounting for the variance captured by earlier principal components). The further a weight is from zero, the more the principal component is in the direction of the corresponding feature. If two features have large weights of the same sign (both positive or both negative), then increases in one tend expect to be associated with increases in the other. To contrast, features with different signs can be expected to show a negative correlation: increases in one variable should result in a decrease in the other.
# 
# - To investigate the features, you should map each weight to their corresponding feature name, then sort the features according to weight. The most interesting features for each principal component, then, will be those at the beginning and end of the sorted list. Use the data dictionary document to help you understand these most prominent features, their relationships, and what a positive or negative value on the principal component might indicate.
# - You should investigate and interpret feature associations from the first three principal components in this substep. To help facilitate this, you should write a function that you can call at any time to print the sorted list of feature weights, for the *i*-th principal component. This might come in handy in the next step of the project, when you interpret the tendencies of the discovered clusters.

# In[34]:


# Map weights for the first principal component to corresponding feature names
# and then print the linked values, sorted by weight.
# HINT: Try defining a function here or in a new cell that you can reuse in the
# other cells.
# Dimension indexing

def pca_weights(pca,n):
    
    components = pd.DataFrame(pca.components_, columns = azdias_low.columns)
    weights=components.iloc[n].sort_values(ascending=False)
    return weights

print(pca_weights(pca,0))


# In[35]:


# Map weights for the second principal component to corresponding feature names
# and then print the linked values, sorted by weight.

print(pca_weights(pca,1))


# In[36]:


# Map weights for the third principal component to corresponding feature names
# and then print the linked values, sorted by weight.

print(pca_weights(pca,2))


# ### Discussion 2.3: Interpret Principal Components
# 
# * In the first principal component, it seems "PLZ8_ANTG3" and "PLZ8_ANTG3" which are related to higher family members numbers have positive effect on PCA, while lower family members have negative effect. "MOBI_REGIO" which is about movement pattern has negative effect as well. 
# * "ALTERSKATEGORIE_GROB" and "SEMIO_ERL" have positive effect on th emodel. The parameters are about age and having event-oriented personality while having religious ("SEMIO_KULT") and cultral-minded("SEMIO_REL") personality has negative effect on the model.
# * Having socially minded ("SEMIO_VERT")  and dreamful ("SEMIO_SOZ") personality show positive effect while combative attitude ("SEMIO_KAEM") and gender("ANREDE_KZ") show negative effect.

# ## Step 3: Clustering
# 
# ### Step 3.1: Apply Clustering to General Population
# 
# You've assessed and cleaned the demographics data, then scaled and transformed them. Now, it's time to see how the data clusters in the principal components space. In this substep, you will apply k-means clustering to the dataset and use the average within-cluster distances from each point to their assigned cluster's centroid to decide on a number of clusters to keep.
# 
# - Use sklearn's [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) class to perform k-means clustering on the PCA-transformed data.
# - Then, compute the average difference from each point to its assigned cluster's center. **Hint**: The KMeans object's `.score()` method might be useful here, but note that in sklearn, scores tend to be defined so that larger is better. Try applying it to a small, toy dataset, or use an internet search to help your understanding.
# - Perform the above two steps for a number of different cluster counts. You can then see how the average distance decreases with an increasing number of clusters. However, each additional cluster provides a smaller net benefit. Use this fact to select a final number of clusters in which to group the data. **Warning**: because of the large size of the dataset, it can take a long time for the algorithm to resolve. The more clusters to fit, the longer the algorithm will take. You should test for cluster counts through at least 10 clusters to get the full picture, but you shouldn't need to test for a number of clusters above about 30.
# - Once you've selected a final number of clusters to use, re-fit a KMeans instance to perform the clustering operation. Make sure that you also obtain the cluster assignments for the general demographics data, since you'll be using them in the final Step 3.3.

# In[37]:


# Over a number of different cluster counts...
def get_score(center, data):
    kmeans=KMeans(n_clusters=center , random_state=3)
    model=kmeans.fit(data)
    return (np.abs(model.score(data)))

    # run k-means clustering on the data and...
    
    
    # compute the average within-cluster distances.
scores=[]
centers=list(range(1,15))
for center in centers:
    scores.append(get_score(center, x_pca))
    


# In[38]:


# Investigate the change in within-cluster distance across number of clusters.
# HINT: Use matplotlib's plot function to visualize this relationship.
plt.plot(centers,scores, linestyle='--', marker='o', color='b')
plt.xlabel('centers')
plt.ylabel('scores')


# In[39]:


# Re-fit the k-means model with the selected number of clusters and obtain
# cluster predictions for the general population demographics data.
kmeans=KMeans(n_clusters=10 , random_state=3)
model=kmeans.fit(x_pca)
label=kmeans.predict(x_pca)


# ### Discussion 3.1: Apply Clustering to General Population
# 
# Based on elbow method in K-means it seems that 10 clusters lead to good performance for the model.

# ### Step 3.2: Apply All Steps to the Customer Data
# 
# Now that you have clusters and cluster centers for the general population, it's time to see how the customer data maps on to those clusters. Take care to not confuse this for re-fitting all of the models to the customer data. Instead, you're going to use the fits from the general population to clean, transform, and cluster the customer data. In the last step of the project, you will interpret how the general population fits apply to the customer data.
# 
# - Don't forget when loading in the customers data, that it is semicolon (`;`) delimited.
# - Apply the same feature wrangling, selection, and engineering steps to the customer demographics using the `clean_data()` function you created earlier. (You can assume that the customer demographics data has similar meaning behind missing data patterns as the general demographics data.)
# - Use the sklearn objects from the general demographics data, and apply their transformations to the customers data. That is, you should not be using a `.fit()` or `.fit_transform()` method to re-fit the old objects, nor should you be creating new sklearn objects! Carry the data through the feature scaling, PCA, and clustering steps, obtaining cluster assignments for all of the data in the customer demographics data.

# In[40]:


# Load in the customer demographics data.
customers = pd.read_csv('Udacity_CUSTOMERS_Subset.csv', sep=';')


# In[41]:


# Apply preprocessing, feature transformation, and clustering from the general
# demographics onto the customer data, obtaining cluster predictions for the
# customer demographics data.
customers_clean=clean_data(customers)
imputer=Imputer()
customers_clean=pd.DataFrame(imputer.fit_transform(customers_clean), columns=customers_clean.columns)
stdscaler=StandardScaler()
customers_clean=pd.DataFrame(stdscaler.fit_transform(customers_clean), columns=customers_clean.columns)


# In[42]:


pca1=PCA(n_components=15)
x_pca1=pca1.fit_transform(customers_clean)
kmeans1=KMeans(n_clusters=10 , random_state=3)
model1=kmeans1.fit(x_pca1)
label1=kmeans1.predict(x_pca1)


# ### Step 3.3: Compare Customer Data to Demographics Data
# 
# At this point, you have clustered data based on demographics of the general population of Germany, and seen how the customer data for a mail-order sales company maps onto those demographic clusters. In this final substep, you will compare the two cluster distributions to see where the strongest customer base for the company is.
# 
# Consider the proportion of persons in each cluster for the general population, and the proportions for the customers. If we think the company's customer base to be universal, then the cluster assignment proportions should be fairly similar between the two. If there are only particular segments of the population that are interested in the company's products, then we should see a mismatch from one to the other. If there is a higher proportion of persons in a cluster for the customer data compared to the general population (e.g. 5% of persons are assigned to a cluster for the general population, but 15% of the customer data is closest to that cluster's centroid) then that suggests the people in that cluster to be a target audience for the company. On the other hand, the proportion of the data in a cluster being larger in the general population than the customer data (e.g. only 2% of customers closest to a population centroid that captures 6% of the data) suggests that group of persons to be outside of the target demographics.
# 
# Take a look at the following points in this step:
# 
# - Compute the proportion of data points in each cluster for the general population and the customer data. Visualizations will be useful here: both for the individual dataset proportions, but also to visualize the ratios in cluster representation between groups. Seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) or [`barplot()`](https://seaborn.pydata.org/generated/seaborn.barplot.html) function could be handy.
#   - Recall the analysis you performed in step 1.1.3 of the project, where you separated out certain data points from the dataset if they had more than a specified threshold of missing values. If you found that this group was qualitatively different from the main bulk of the data, you should treat this as an additional data cluster in this analysis. Make sure that you account for the number of data points in this subset, for both the general population and customer datasets, when making your computations!
# - Which cluster or clusters are overrepresented in the customer dataset compared to the general population? Select at least one such cluster and infer what kind of people might be represented by that cluster. Use the principal component interpretations from step 2.3 or look at additional components to help you make this inference. Alternatively, you can use the `.inverse_transform()` method of the PCA and StandardScaler objects to transform centroids back to the original data space and interpret the retrieved values directly.
# - Perform a similar investigation for the underrepresented clusters. Which cluster or clusters are underrepresented in the customer dataset compared to the general population, and what kinds of people are typified by these clusters?

# In[43]:


# Compare the proportion of data in each cluster for the customer data to the
# proportion of data in each cluster for the general population.

fig, ax =plt.subplots(1,2 , figsize=(12,6))
sns.countplot(label, ax=ax[0])
ax[0].set_title('Population Clusters')

sns.countplot(label1, ax=ax[1])
ax[1].set_title('Customer Clusters')

fig.show()


# In[75]:


# What kinds of people are part of a cluster that is overrepresented in the
# customer data compared to the general population?

centroid_9=stdscaler.inverse_transform(pca1.inverse_transform(model1.cluster_centers_[9]))
underrepresented_9=pd.Series(data=centroid_9, index=customers_clean.columns)
print(overrepresented_9)


# In[76]:


# What kinds of people are part of a cluster that is underrepresented in the
# customer data compared to the general population?
centroid_4=stdscaler.inverse_transform(pca1.inverse_transform(model1.cluster_centers_[4]))
underrepresented_4=pd.Series(data=centroid_4, index=customers_clean.columns)
print(underrepresented_4)


# ### Discussion 3.3: Compare Customer Data to Demographics Data
# 
# * The overrepresented group which are very popular in the mailing company have these characteristics:
# They are males between 40-60 years old mostly in the 40s decade of their life living in high share of 1-2 family houses. They are very high materialistic with high financial interests and high net income. Most of them own houses and less academic people present in their houses. They are also low cultural minded and very low dreamful.
# 
# 
# * The underrepresented group which can be the new target for mailing company are males and females between 40-60 yearsold. They have low financial interest with very low income. There are more academic people in the house and afew of them are house ownners. They are averagely materialistic and dreaful. Also, they are high cultural minded and have low online affinities.

# > Congratulations on making it this far in the project! Before you finish, make sure to check through the entire notebook from top to bottom to make sure that your analysis follows a logical flow and all of your findings are documented in **Discussion** cells. Once you've checked over all of your work, you should export the notebook as an HTML document to submit for evaluation. You can do this from the menu, navigating to **File -> Download as -> HTML (.html)**. You will submit both that document and this notebook for your project submission.

# In[ ]:




