# 1. Business Problem 
# 1.1.	What is the business objective?
# 1.2.	What are the constraints?
# 1.3.	Define success criteria

# 2. Work on each feature of the dataset to create a data dictionary as displayed in the below image:
 

# 3. Exploratory Data Analysis (EDA):
#       3.1. Univariate analysis.
#       3.2. Bivariate analysis.

# 4. Data Pre-processing 
# 4.1 Data Cleaning, Feature Engineering, etc.
# 5. Model Building 
# 5.1 Build the model on the scaled data (try multiple options).
# 5.2 Perform the KMeans and DBscan clustering and find out the best model that minimizes Within the Sum of Squares. Compare the result with Hierarchical Clustering methods.
# 5.3 Validate the clusters (try with the different numbers of clusters), label the clusters, and derive insights (compare the results from multiple approaches).
# 6. Write about the benefits/impact of the solution - in what way does the business (client) benefit from the solution provided?
# 7. Deploy the best model using Python Flask on the local machine.






'''
Problem Statements:
Global air travel has seen an upward trend in recent times. The maintenance of operational efficiency and maximizing profitability are crucial for airlines
and airport authorities. Businesses need to optimize airline and terminal operations to enhance passenger satisfaction, improve turnover rates, and increase overall revenue. 
The airline companies with the available data want to find an opportunity to analyze and understand travel patterns, customer demand, and terminal usage.

CRISP-ML(Q) process model describes six phases:
1. Business and Data Understanding
2. Data Preparation
3. Model Building
4. Model Evaluation
5. Deployment
6. Monitoring and Maintenance

Objective: Maximize the Sales 
Constraints: Minimize the Customer Retention
Success Criteria: 
Business Success Criteria: Increase the Sales by 10% to 12% by targeting cross-selling opportunities on current customers.
ML Success Criteria: Achieve a Silhouette coefficient of at least 0.6
Economic Success Criteria: The insurance company will see an increase in revenues by at least 8% 

Data: Refer to the ‘AirTraffic_Passenger_Statistics.csv’ dataset.
'''





# Objective: Maximize the operational efficiency
# Constraints: Focus on maintaining financial health.

# Success Criteria: 
# Business Success Criteria: Increase the operational efficiency by 10% to 12% by segmenting the Airlines.
# ML Success Criteria: Achieve a Silhouette coefficient of at least 0.7
# Economic Success Criteria: The airline companies will see an increase in revenues by at least 8% 

# Proposed Plan: By segmenting the airlines on this basis of certain features.

# Data Dictionary:
    
# Activity Period: Represents the time period of the data (e.g., 200507 for July 2005).
# Operating Airline: The name of the airline operating the flights.
# Operating Airline IATA Code: The IATA code for the airline.
# GEO Region: The geographical region where the airline operates.
# Terminal: The terminal at the airport where the flights operate.
# Boarding Area: The specific boarding area within the terminal.
# Passenger Count: The number of passengers.
# Year: The year of the activity period.
# Month: The month of the activity period.

# """






# Importing the necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting data visualizations
import sweetviz as sv  # For automated exploratory data analysis (EDA)
import dtale  # For interactive data visualization
from sklearn.pipeline import Pipeline  # For creating machine learning pipelines
from sklearn.impute import SimpleImputer  # For handling missing values
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder  # For scaling and encoding data
from sklearn.compose import ColumnTransformer  # For applying different preprocessing to different columns
from scipy.cluster.hierarchy import linkage, dendrogram  # For hierarchical clustering and dendrogram visualization
from clusteval import clusteval  # For evaluating clustering models
from sklearn import metrics  # For evaluating machine learning models

from sqlalchemy import create_engine, text  # For database connections and SQL queries
from urllib.parse import quote  # For encoding database passwords
from AutoClean import AutoClean  # For automatic data cleaning
from sklearn.decomposition import PCA  # For dimensionality reduction
from sklearn.preprocessing import RobustScaler, PolynomialFeatures  # For additional scaling and feature engineering
from sklearn.cluster import KMeans
import joblib
import pickle




# Reading the CSV file into a DataFrame
flightk = pd.read_csv("C:/Users/user/Desktop/DATA SCIENCE SUBMISSION/Data Science [ 3 ] - Kmeans And Dbscan Clustering/AirTraffic_Passenger_Statistics.csv")

# Displaying information about the DataFrame, such as data types and non-null counts
flightk.info()

# Displaying basic statistical details like mean, standard deviation, and percentiles
flightk.head()

# Database Connection setup
user = 'root'  # Username for the database
pw = '12345678'  # Password for the database
db = 'univ_db'  # Name of the database

# Creating an engine to connect to the MySQL database
engine = create_engine(f"mysql+pymysql://{user}:{quote(pw)}@localhost/{db}")

# Pushing the DataFrame to a SQL table named 'airline_table'
flightk.to_sql('airline_table', con=engine, if_exists='replace', chunksize=1000, index=False)

# Reading the data back from the SQL table (if needed)
sql = 'SELECT * FROM airline_table;'
flightk_1 = pd.read_sql_query(text(sql), engine)

# Displaying the first few rows and information of the SQL query result
flightk_1.head()
flightk_1.info()






# Dropping unnecessary columns
flightk_1.drop(columns=['Operating Airline IATA Code', 'GEO Region'], inplace=True)

# Automated Exploratory Data Analysis (EDA) using Sweetviz
my_report = sv.analyze(flightk)
my_report.show_html('flightk_stats.html')  # Generates an HTML report with insights

# Interactive EDA using Dtale
my_report_dtale = dtale.show(flightk)
my_report_dtale.open_browser()  # Opens the Dtale interface in the browser





# Checking for null values
flightk_1.isnull().sum()

# Checking for duplicate records
flightk_1.duplicated().sum()

# Dropping duplicate records
flightk_1 = flightk_1.drop_duplicates()
flightk_1.duplicated().sum()  # Confirming that duplicates have been removed







# Identifying and handling outliers using the IQR method
numerical_cols = flightk_1.select_dtypes(['int64', 'float64']).columns.values

def outliers_identifier(df, column):
    q1 = df[column].quantile(0.25)  # First quartile (25th percentile)
    q3 = df[column].quantile(0.75)  # Third quartile (75th percentile)
    iqr = q3 - q1  # Interquartile range (IQR)
    outliers = df[(df[column] < (q1 - 1.5 * iqr)) | (df[column] > (q3 + 1.5 * iqr))]  # Identifying outliers
    return outliers

# Printing the number of outliers for each numerical column
for col in numerical_cols:
    outliers = outliers_identifier(flightk_1, col)
    print(f'Outliers in {col} column: {len(outliers)}')









# Aggregating passenger counts for each airline
airline_agg = flightk_1.groupby('Operating Airline').agg( 
    total_passengers=pd.NamedAgg(column='Passenger Count', aggfunc='sum'),
    avg_passengers=pd.NamedAgg(column='Passenger Count', aggfunc='mean'),
    median_passengers=pd.NamedAgg(column='Passenger Count', aggfunc='median')
).reset_index()

# Display the aggregated airline data
print("Airline Aggregation:")
print(airline_agg.head())

# Renaming the 'Operating Airline' column to avoid conflicts during merging
airline_agg.rename(columns={'Operating Airline': 'Operating Airline_'}, inplace=True)



# Counting the number of records for each airline
airline_count = flightk_1['Operating Airline'].value_counts().reset_index()
airline_count.columns = ['Operating Airline_', 'Airline_count']

# Merging the count data with the aggregated data
merged = pd.merge(airline_agg, airline_count, on='Operating Airline_')








# Terminal and Boarding Area Usage analysis
terminal_usage = flightk_1.groupby(['Operating Airline', 'Terminal', 'Boarding Area']).size().reset_index(name='frequency_terminal_broad')

# Display terminal usage information
print("Terminal Usage:")
print(terminal_usage.head())

# Pivoting the terminal usage data to create one-hot encoded features
terminal_features = terminal_usage.pivot_table(
    index='Operating Airline',
    columns=['Terminal', 'Boarding Area'],
    values='frequency_terminal_broad',
    fill_value=0
).reset_index()

# Flattening the multi-level columns
terminal_features.columns = ['_'.join(map(str, col)).strip() for col in terminal_features.columns.values]

# Display terminal features
print("Terminal Features:")
print(terminal_features.head())
print(terminal_features.info())






# # Seasonal trends analysis - Aggregating passenger counts by year and month
# seasonal_trends = flightk_1.groupby(['Operating Airline', 'Year', 'Month']).agg(
#     monthly_passengers=pd.NamedAgg(column='Passenger Count', aggfunc='sum')
# ).reset_index()

# # Display seasonal trends information
# print("Seasonal Trends:")
# print(seasonal_trends.head())

# # Pivoting the seasonal trends data to create features for each month and year
# seasonal_features = seasonal_trends.pivot_table(
#     index='Operating Airline',
#     columns=['Year', 'Month'],
#     values='monthly_passengers',
#     fill_value=0
# ).reset_index()

# # Flattening the columns by combining year and month
# seasonal_features.columns = ['_'.join(map(str, col)).strip() if isinstance(col, tuple) else col for col in seasonal_features.columns]

# # Display seasonal features after flattening
# print("Seasonal Features:")
# print(seasonal_features.head())
# print(seasonal_features.info())







# Convert the 'Month' column to a numerical format
month_mapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}
flightk_1['Month'] = flightk_1['Month'].map(month_mapping)

# Define a function to map months to seasons
def map_month_to_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

# Apply the mapping to create a 'Season' column
flightk_1['Season'] = flightk_1['Month'].apply(map_month_to_season)

# Aggregating passenger counts by year and season
seasonal_trends = flightk_1.groupby(['Operating Airline', 'Year', 'Season']).agg(
    seasonal_passengers=pd.NamedAgg(column='Passenger Count', aggfunc='sum')
).reset_index()

# Display seasonal trends information
print("Seasonal Trends:")
print(seasonal_trends.head())

# Pivoting the seasonal trends data to create features for each season and year
seasonal_features = seasonal_trends.pivot_table(
    index='Operating Airline',
    columns=['Year', 'Season'],
    values='seasonal_passengers',
    fill_value=0
).reset_index()

# Flattening the columns by combining year and season
seasonal_features.columns = ['_'.join(map(str, col)).strip() if isinstance(col, tuple) else col for col in seasonal_features.columns]

# Display seasonal features after flattening
print("Seasonal Features:")
print(seasonal_features.head())
print(seasonal_features.info())















# Merging all DataFrames on the basis of 'Operating Airline'
merged_df = pd.merge(merged, terminal_features, on='Operating Airline_' , how='outer')
merged_df = pd.merge(merged_df, seasonal_features, on='Operating Airline_', how='outer')



# Display the final merged DataFrame
print("Merged DataFrame:")
print(merged_df.head())
print(merged_df.info())







# Checking for null values in the merged DataFrame
merged_df.isnull().sum()

# Checking for duplicates in the merged DataFrame
merged_df.duplicated().sum()

# Identifying and handling outliers using the IQR method in the merged DataFrame
numerical_cols = merged_df.select_dtypes(['int64', 'float64']).columns.values

for col in numerical_cols:
    outliers = outliers_identifier(merged_df, col)
    print(f'Outliers in {col} column: {len(outliers)}')

# Handling outliers using Winsorization with AutoClean
merged_df1 = AutoClean(merged_df, mode='manual', outliers='winz').output

# Re-checking for outliers after applying AutoClean
for col in numerical_cols:
    outliers = outliers_identifier(merged_df1, col)
    print(f'Outliers in {col} column: {len(outliers)}')

# Storing the final cleaned DataFrame before scaling
final_before_scaling = merged_df1.copy()
final_before_scaling.info()

# Checking the structure of the final cleaned DataFrame
merged_df1.info()

# Separating numerical and categorical columns
numerical_column = merged_df1.select_dtypes(include=['int64', 'float64'])
categorical_column = merged_df1.select_dtypes(include=['object']).columns.values

# Printing the number of numerical columns
sum = 0
for i in numerical_column:
    sum += 1
print(sum)

# Printing the number of categorical columns
len(categorical_column)

# Dropping the 'Operating Airline_' column as it's not needed for scaling
merged_df1.drop(columns=['Operating Airline_'], inplace=True)

# Creating a pipeline for numerical data preprocessing (imputation and scaling)
pipeline_numerical = Pipeline(steps=[
    ('imputer_num', SimpleImputer(strategy='mean')),  # Handling missing values by replacing with the mean
    ('scale_num', MinMaxScaler())  # Scaling the numerical features to a range of [0, 1]
])

# Applying the preprocessing pipeline to the DataFrame
preprocessor = ColumnTransformer(transformers=[             
    ('num_transform', pipeline_numerical, numerical_column.columns)
], remainder='passthrough')

# Creating the final pipeline to transform the data
pipeline = Pipeline(steps=[('preprocessing', preprocessor)])

# Fit and transform the data using the pipeline
flight_scaled = pipeline.fit_transform(merged_df1)
print(flight_scaled)  # Print the transformed data

# Final Data Preparation
flight_dataframe = pd.DataFrame(flight_scaled, columns=merged_df1.columns)
flight_dataframe.info()










# Applying PCA to reduce dimensions while retaining 85% of the variance
pca = PCA(n_components=2)  # Initialize PCA to retain 85% of the variance in the data
final_data_pca = pca.fit_transform(flight_dataframe)  # Fit and transform the data using PCA










# List to store Total Within-Cluster Sum of Squares (TWSS)
TWSS = []
k = list(range(2, 9))  # Range of cluster numbers to try

# Loop through each number of clusters to calculate TWSS
for i in k:
    kmeans = KMeans(n_clusters=i)  # Initialize KMeans with i clusters
    kmeans.fit(final_data_pca)  # Fit KMeans on the PCA-transformed data
    TWSS.append(kmeans.inertia_)  # Append the inertia (TWSS) to the list

# Creating a scree plot to find the optimal number of clusters
plt.plot(k, TWSS, 'ro-')  # Plot TWSS against the number of clusters
plt.xlabel("Number of Clusters")  # X-axis label
plt.ylabel("Total Within-Cluster Sum of Squares")  # Y-axis label

# Install kneed package for detecting the "elbow" point in the plot (if not already installed)
# pip install kneed

# List to store inertia values for each number of clusters
List = []

# Loop through each number of clusters to calculate inertia
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k)  # Initialize KMeans with k clusters
    kmeans.fit(final_data_pca)  # Fit KMeans on the PCA-transformed data
    List.append(kmeans.inertia_)  # Append the inertia to the list

# Using KneeLocator to find the "elbow" point
from kneed import KneeLocator
kl = KneeLocator(range(2, 9), List, curve='convex', direction='decreasing')  # Detect the elbow point

# Plotting the inertia values with the detected elbow point
plt.plot(range(2, 9), List)  # Plot inertia values
plt.xticks(range(2, 9))  # Set x-ticks to cluster numbers
plt.ylabel("Inertia")  # Y-axis label
plt.axvline(x=kl.elbow, color='r', label='Elbow', ls='--')  # Add a vertical line at the elbow point
plt.show()  # Display the plot

print("Elbow point:", kl.elbow)  # Print the detected elbow point
# Note: The line is mostly linear, making it hard to detect a clear elbow point

# Building KMeans clustering with 3 clusters (initial assumption)
model = KMeans(n_clusters=4)  # Initialize KMeans with 3 clusters
yy = model.fit(final_data_pca)  # Fit KMeans on the PCA-transformed data

# Displaying the cluster labels
model.labels_  # Array of cluster labels assigned to each data point

# Evaluating the clustering performance using various metrics
# Silhouette Score: Measures how similar an object is to its own cluster compared to other clusters
metrics.silhouette_score(final_data_pca, model.labels_)

# Calinski-Harabasz Score: Higher score indicates better-defined clusters
metrics.calinski_harabasz_score(final_data_pca, model.labels_)

# Davies-Bouldin Score: Lower score indicates better clustering
metrics.davies_bouldin_score(final_data_pca, model.labels_)

# Finding the best number of clusters based on the silhouette score
silhouette_score = []  # List to store silhouette scores

for k in range(2, 9):
    kmeans = KMeans(n_clusters=k)  # Initialize KMeans with k clusters
    kmeans.fit(final_data_pca)  # Fit KMeans on the PCA-transformed data
    score = metrics.silhouette_score(final_data_pca, kmeans.labels_)  # Calculate silhouette score
    silhouette_score.append([k, score])  # Append the cluster number and score to the list

print(silhouette_score)  # Print the list of silhouette scores for each cluster number

# Sorting silhouette scores in descending order to find the best k
sorted(silhouette_score, reverse=True, key=lambda x: x[1])

# The silhouette coefficients indicate that the best number of clusters is 2
# Building the final KMeans model with the optimal number of clusters (k=2)
bestmodel = KMeans(n_clusters=2)  # Initialize KMeans with 2 clusters
result = bestmodel.fit(final_data_pca)  # Fit the final model on the PCA-transformed data

# Calculating the silhouette score for the final model
metrics.silhouette_score(final_data_pca, bestmodel.labels_)  # Higher score indicates better clustering


# saving the best model

pickle.dump(result,open('clust.pkl','wb'))


# Extract cluster labels from the best model
clusters = pd.Series(bestmodel.labels_)
print(clusters)
  # Display the distribution of cluster assignments
print("Cluster Distribution:")
print(clusters.value_counts())

# Display basic information about the clusters
print("\nCluster Information:")
print(clusters.info())

# Preview the first 77 entries of the clusters
print("\nFirst 77 Cluster Labels:")
print(clusters.head(77))

# Display information about the original DataFrame before scaling
print("\nOriginal DataFrame Information:")
print(final_before_scaling.info())

# Combine the original DataFrame with the cluster labels
final_df = pd.concat([final_before_scaling, clusters], axis=1)

# Preview the resulting DataFrame
print("\nFinal DataFrame with Clusters:")
print(final_df.head())

# Rename the cluster column for clarity
final_df.rename(columns={0: 'Cluster'}, inplace=True)

# Verify the renaming and preview the DataFrame
print("\nFinal DataFrame after Renaming Cluster Column:")
print(final_df.head())

# Save the final DataFrame to a CSV file
final_df.to_csv('Kmeans_cluster2.csv', encoding='utf-8', index=False)

print("Final DataFrame has been saved to 'Kmeans_cluster.csv'.")



import os 
os.getcwd()









import pandas as pd

# Load the dataset
cluster_data = pd.read_csv('C:/Users/user/Desktop/DATA SCIENCE SUBMISSION/Data Science [ 3 ] - Kmeans And Dbscan Clustering/Kmeans_cluster2.csv')
cluster_data.head(10)


# # Rename the clusters using map
cluster_data['Cluster'] = cluster_data['Cluster'].map({0: 'Low-Volume Airlines', 1: 'High-Volume Airlines' })
cluster_data.head()













# now loading the cluster with the updated name into the csv file.
cluster_data.to_csv('Kmeans_cluster2_final.csv', encoding = 'utf-8' , index = False)



# Group the data by the 'Cluster' column
grouped_data = cluster_data.groupby('Cluster')


# Select only numeric columns
numeric_columns = cluster_data.select_dtypes(include=['number','int64','float64']).columns


# Calculate statistical measures for each numeric column within each group
grouped_stats = grouped_data[numeric_columns].agg(['count', 'mean', 'median', 'std', 'min', 'max'])


# Display the statistical measures
print(grouped_stats)















'''
Cluster Characteristics:
    
    
    
    

Cluster 0: Low-Volume Airlines

Total Passengers: Lower (~829,000) compared to Cluster 1.

Average and Median Passengers: Lower values (~5,200 and ~4,600 respectively), indicating smaller airlines or those with fewer passengers.

Airline Count: Smaller (~107), indicating fewer airlines.

International Passengers (Terminal A): Lower average (~61) compared to Cluster 1.


Seasonal Trends:

Summer: Busiest season but with lower passenger counts compared to Cluster 1, peaking around 27,700 passengers in 2015.
Fall, Spring, and Winter: Passenger counts are relatively stable but lower compared to Cluster 1. Winter again shows the lowest traffic, 
suggesting a more seasonal demand for these airlines.







Cluster 1: High-Volume Airlines

Total Passengers: Significantly higher (~5.7 million) compared to Cluster 0.

Average and Median Passengers: Higher values (~21,000 and ~17,000 respectively), indicating airlines with larger passenger volumes.

Airline Count: Higher (~365), suggesting a wide range of airlines.

International Passengers (Terminal A): Higher average (~82), indicating a larger number of international travelers.

Seasonal Trends:

Summer: Consistently the highest passenger counts across all years, peaking around 173,000 passengers in 2015.
Fall and Spring: Both seasons show strong passenger numbers, typically second to summer.
Winter: Passenger counts are generally lower in winter, reflecting a seasonal decline, but the impact is less severe than in Cluster 0 due to a larger and more consistent passenger base.





Labeling the Clusters:
Cluster 0: "Low-Volume Airlines"
Cluster 1: "High-Volume Airlines"





High-Volume Airlines (Cluster 1):

These airlines serve a large number of passengers, likely operating on popular routes or with more frequent flights.
The higher airline count and larger number of international passengers indicate a broader market presence and a more diverse route network.
Low-Volume Airlines (Cluster 0):

These airlines have fewer passengers, likely focusing on niche markets or operating with less frequency.
The lower airline count suggests a more specialized or limited operation compared to the high-volume airlines in Cluster 1.
Seasonal Insights:

Seasonal Peaks: Both clusters exhibit a clear seasonal trend, with summer being the peak travel season. The magnitude of this seasonality is more pronounced in Cluster 1 (High-Volume Airlines), reflecting a larger influx of passengers during peak travel periods.
Winter Decline: The decline in passenger numbers during winter is notable for both clusters but is more pronounced in Cluster 0 (Low-Volume Airlines), indicating a stronger impact of seasonality on these smaller airlines.
Operational Planning:

Airlines in Cluster 1 (High-Volume Airlines) need to plan for increased resources and staffing during the summer months to manage the higher passenger volume.
Airlines in Cluster 0 (Low-Volume Airlines) might focus on maintaining service quality and operational efficiency during the off-peak winter season, as they face greater seasonality effects.






Benefits/Impact of the Solution:
    
Operational Efficiency:
Clustering airlines into high and low-volume groups can streamline resource allocation, such as gate assignments and terminal usage, 
leading to more efficient operations.


Passenger Satisfaction:
Tailoring services based on cluster profiles, such as offering premium services to high-volume airlines and more customized services to low-volume airlines, 
can enhance the passenger experience.


Revenue Optimization:
Understanding the different clusters allows targeted marketing and service offerings, potentially increasing revenues by focusing on high-value passenger segments.



Resource Allocation: Understanding seasonal peaks allows airports and airlines to allocate resources more effectively,
ensuring that staffing, gate assignments, and customer service efforts are aligned with passenger demand.


Marketing Strategies: Airlines can tailor their marketing efforts based on seasonal trends, offering promotions or special deals 
during off-peak seasons to boost passenger numbers.



Capacity Planning: Airports can optimize terminal usage and flight scheduling to manage seasonal fluctuations,
reducing congestion during peak periods and avoiding underutilization during off-peak times.


'''






















