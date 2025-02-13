import pandas as pd
import numpy as np
import plotly.express as px
from pymongo import MongoClient
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

MONGO_USER = "rootarded"
MONGO_PASSWORD = "Yhs7a7gK"
MONGO_HOST = "192.168.1.24"
MONGO_PORT = "27017"
DB_NAME = "EnergyLearningLocal"
COLLECTION_NAME = "weightedZoneResultsv3"
MONGO_URI = f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/{DB_NAME}"
MONGO_URI_LOCAL= f"mongodb://localhost:{MONGO_PORT}/"

def fetch_data():
    client = MongoClient(MONGO_URI_LOCAL)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    data = list(collection.find({}, {"_id": 0}))
    return pd.DataFrame(data)

def scale_data(df):

    scaler = MinMaxScaler()
    numeric_df = df.select_dtypes(include=[np.number])
    scaled_df = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_df.columns)
    return scaled_df

def analyze_distributions(df):
    print("Analyzing feature distributions...")
    for col in df.select_dtypes(include=[np.number]):
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
        fig.show()


def analyze_missing_data(df):
    missing_info = df.isnull().sum()
    missing_info = missing_info[missing_info > 0].sort_values(ascending=False)

    if not missing_info.empty:
        print("Missing data detected:")
        print(missing_info)
        fig = px.bar(x=missing_info.index, y=missing_info.values, title="Missing Values Per Feature")
        fig.show()
    else:
        print("No missing data detected.")


def correlation_matrix(df):
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    fig = px.imshow(corr_matrix, text_auto=True, title="Feature Correlation Matrix")
    fig.show()
    fig.write_html("correlation_matrix.html")



def mitigate_issues(df):
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=[np.number])),
                              columns=df.select_dtypes(include=[np.number]).columns)
    print("Missing values imputed using median strategy.")
    return df_imputed


def main():

    df = fetch_data()
    df_scaled = scale_data(df)
    analyze_distributions(df_scaled)
    analyze_missing_data(df_scaled)
    correlation_matrix(df_scaled)
    df_cleaned = mitigate_issues(df_scaled)

if __name__ == "__main__":
    main()
