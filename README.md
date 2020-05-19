# Unsupervised-Machine Learning-Customer-Segmentation
In this project, unsupervised learning techniques are used to analyze demographics data of customers of a mail-order sales company in Germany against demographics information for the general population. The goal of this project is to characterize customers segment of population, and to build a model that will be able to predict customers for Arvato Financial Solutions.

The data for this project is provided by Udacity partners at Bertelsmann Arvato Analytics, and represents a real-life data science task. It includes general population dataset, customer segment data set, dataset of mailout campaign with response and test dataset that needs to make predictions.

Three main steps were used for the project: data preprocessing, feature selection and customer segmentation. I will describe these steps briefly here:
## Data Preprocessing:
* Converting missing values to NaN
* Assessing columns with the most missing values and drop them
* Assessing rows with the most missing values
* Encoding categorical variables
* Dropping multi-level variables
* Reencoding mixed datatypes using functions
At the end of data preprocessing, a cleaning function from the above steps was created to use it for both general population and mail order company demographic data. 

## Feature Selection:
* Replacing any missing value by mean using Imputer function
* feature scaling by StandardScaler

## Customer Segmentation:
* Dimentionality reduction by principal component analysis
* Performing k-means clustering on the PCA-transformed data( Elbow curve determined the best number of clusters)

At the end, population clusters were compared with customer clusters

