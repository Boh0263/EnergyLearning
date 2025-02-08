import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


MONGO_USER = "rootarded"
MONGO_PASSWORD = "Yhs7a7gK"
MONGO_HOST = "localhost"
MONGO_PORT = "27017"
DB_NAME = "EnergyLearning"
COLLECTION_NAME = "Summarized_Zones_Dataset_Normalized"
MONGO_URI = f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/{DB_NAME}"


FEATURE_TO_ANALYZE = "weightedAvgTotalCo2ConsumptionScaled"

def fetch_data():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    data = list(collection.find({}, {
        "_id": 0,
        "zoneKey": 1,
        "numEntriesScaled": 1,
        "weightedAvgTotalCo2ConsumptionScaled": 1,
        "weightedAvgTotalConsumptionScaled": 1,
        "avgEstimatedPercentageScaled": 1,
        "avgRenewableRatioScaled": 1,
        "avgFossilFuelRatioScaled": 1
    }))

    return pd.DataFrame(data)

def perform_statistical_analysis(df, feature):
    print(f"Statistical Analysis for {feature}\n")

    # Basic Stats
    stats = df[feature].describe()
    print(stats)

    # Istogramma
    fig = px.histogram(df, x=feature, nbins=30, title=f"Distribution of {feature}")
    fig.show()

    #  Analisi della Correlazione
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()[feature].sort_values(ascending=False)

    print("\nCorrelation with other features:")
    print(correlations)


    if len(correlations) > 1:
        top_corr_feature = correlations.index[1]  # Feature più correlata.
        fig_scatter = px.scatter(df, x=top_corr_feature, y=feature,
                                 title=f"{feature} vs {top_corr_feature}")
        fig_scatter.show()

    return stats, correlations

def regression_analysis(df, feature):
    X = df.drop(columns=[feature, "zoneKey"])
    y = df[feature]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nRegression Analysis:")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    print(f"R² Score: {r2_score(y_test, y_pred)}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_test, mode='markers', name='Actual'))
    fig.add_trace(go.Scatter(y=y_pred, mode='lines', name='Predicted'))
    fig.update_layout(title=f"Regression Analysis for {feature}", xaxis_title="Index", yaxis_title=feature)
    fig.show()

df = fetch_data()
stats, correlations = perform_statistical_analysis(df, FEATURE_TO_ANALYZE)
regression_analysis(df, FEATURE_TO_ANALYZE)