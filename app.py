import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import pickle
import joblib
from sqlalchemy import create_engine
from urllib.parse import quote
import logging
import traceback
from AutoClean import AutoClean

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)




# Load the KMeans model and PCA
with open('clust.pkl', 'rb') as file:
    kmeans_model = pickle.load(file)
    
# Database connection setup
user = 'root'
pw = '12345678'
db = 'abc'
engine = create_engine(f"mysql+pymysql://{user}:{quote(pw)}@localhost/{db}")





def preprocess_data(data):
    logger.debug(f"Original DataFrame columns: {data.columns.tolist()}")
    
    # Dropping unnecessary columns
    columns_to_drop = ['Operating Airline IATA Code', 'GEO Region']
    data.drop(columns=[col for col in columns_to_drop if col in data.columns], inplace=True, errors='ignore')
    
    # Drop duplicates
    data = data.drop_duplicates()
    
    # Aggregating passenger counts for each airline
    airline_agg = data.groupby('Operating Airline').agg( 
        total_passengers=pd.NamedAgg(column='Passenger Count', aggfunc='sum'),
        avg_passengers=pd.NamedAgg(column='Passenger Count', aggfunc='mean'),
        median_passengers=pd.NamedAgg(column='Passenger Count', aggfunc='median')
    ).reset_index()
    
    # Renaming the 'Operating Airline' column
    airline_agg.rename(columns={'Operating Airline': 'Operating Airline_'}, inplace=True)
    
    # Counting the number of records for each airline
    airline_count = data['Operating Airline'].value_counts().reset_index()
    airline_count.columns = ['Operating Airline_', 'Airline_count']

    # Merging the count data with the aggregated data
    merged = pd.merge(airline_agg, airline_count, on='Operating Airline_')

    # Terminal and Boarding Area Usage analysis
    terminal_usage = data.groupby(['Operating Airline', 'Terminal', 'Boarding Area']).size().reset_index(name='frequency_terminal_broad')

    # Pivoting the terminal usage data
    terminal_features = terminal_usage.pivot_table(
        index='Operating Airline',
        columns=['Terminal', 'Boarding Area'],
        values='frequency_terminal_broad',
        fill_value=0
    ).reset_index()

    # Flattening the multi-level columns
    terminal_features.columns = ['_'.join(map(str, col)).strip() for col in terminal_features.columns.values]

    # Convert the 'Month' column to a numerical format
    month_mapping = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    data['Month'] = data['Month'].map(month_mapping)

    # Map months to seasons
    def map_month_to_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    data['Season'] = data['Month'].apply(map_month_to_season)

    # Aggregating passenger counts by year and season
    seasonal_trends = data.groupby(['Operating Airline', 'Year', 'Season']).agg(
        seasonal_passengers=pd.NamedAgg(column='Passenger Count', aggfunc='sum')
    ).reset_index()

    # Pivoting the seasonal trends data
    seasonal_features = seasonal_trends.pivot_table(
        index='Operating Airline',
        columns=['Year', 'Season'],
        values='seasonal_passengers',
        fill_value=0
    ).reset_index()

    # Flattening the columns
    seasonal_features.columns = ['_'.join(map(str, col)).strip() if isinstance(col, tuple) else col for col in seasonal_features.columns]

    # Merging all DataFrames
    merged_df = pd.merge(merged, terminal_features, on='Operating Airline_', how='outer')
    merged_df = pd.merge(merged_df, seasonal_features, on='Operating Airline_', how='outer')

    # Clean the merged DataFrame with AutoClean
    merged_df_cleaned = AutoClean(merged_df, mode='manual', outliers='winz').output

    return merged_df_cleaned




def scale_data(data):
    
    # Separating numerical and categorical columns
    numerical_columns = data.select_dtypes(include=['int64', 'float64'])
    categorical_columns = data.select_dtypes(include=['object']).columns.values
    
    
    # Dropping 'Operating Airline_' as it's not needed for scaling
    data = data.drop(columns=['Operating Airline_'], inplace=False)

    # # Separating numerical columns
    # numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
    
    if numerical_columns.empty:
        logging.warning("No numerical columns found for scaling.")
        return pd.DataFrame()  # Return an empty DataFrame if no numerical columns

    # Creating a pipeline for numerical data preprocessing (imputation and scaling)
    pipeline_numerical = Pipeline(steps=[
        ('imputer_num', SimpleImputer(strategy='mean')),
        ('scale_num', MinMaxScaler())
    ])

    # Creating a ColumnTransformer for numerical data
    preprocessor = ColumnTransformer(transformers=[
        ('num_transform', pipeline_numerical, numerical_columns.columns)
    ], remainder='passthrough')  # Keep non-numeric columns unchanged

    # Fit and transform the data using the pipeline
    scaled_data = preprocessor.fit_transform(data)

    scaled_data_final = pd.DataFrame(scaled_data, columns=data.columns)




    # Applying PCA
    pca = PCA(n_components=2)  # Adjust the number of components as needed
    pca_data = pca.fit_transform(scaled_data_final)

    return pca_data





def apply_clustering_predict(pca_data):
    # Make predictions using the KMeans model
    predictions = kmeans_model.predict(pca_data)

    # Map predictions to meaningful names
    cluster_names = {0: 'Low-Volume Airlines', 1: 'High-Volume Airlines'}
    named_predictions = [cluster_names.get(pred, 'Unknown') for pred in predictions]

    return named_predictions



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return render_template('error.html', message='No file part in the request')
        
        file = request.files['file']

        if file.filename == '':
            return render_template('error.html', message='No selected file')

        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            return render_template('error.html', message='Invalid file type. Please upload a CSV file.')

        logger.debug(f"Received DataFrame shape: {df.shape}")
        logger.debug(f"Received DataFrame columns: {df.columns.tolist()}")

        merged_df = preprocess_data(df)
        
        if merged_df.empty:
            return render_template('error.html', message='No data available after preprocessing')
        
        scaled_data = scale_data(merged_df)

        if scaled_data.size == 0:
            return render_template('error.html', message='No numeric data available for clustering')
 

        # pca_data = apply_pca(scaled_data)
        predictions = apply_clustering_predict(scaled_data)
        
        result_df = pd.DataFrame({
            'Operating Airline': merged_df['Operating Airline_'],
            'Cluster': predictions
        })
        
        # Save results to the database
        result_df.to_sql('clustering_results', con=engine, if_exists='replace', index=False)
        
        logger.debug(f"Results DataFrame shape: {result_df.shape}")
        logger.debug(f"Results DataFrame sample:\n{result_df.head()}")
        
        return render_template('result.html', tables=[result_df.to_html(classes='data')], titles=result_df.columns.values)
    
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return render_template('error.html', message=str(e))

if __name__ == '__main__':
    app.run(debug=True)