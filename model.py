import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
from sklearn.preprocessing import  MinMaxScaler

MONGO_USER = "<Username>"
MONGO_PASSWORD = "<Password>"
MONGO_HOST = "localhost"
MONGO_PORT = "27017"
DB_NAME = "EnergyLearning"



MONGO_URI = f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/{DB_NAME}"

# Step 1:
def fetch_data():
    client = MongoClient(MONGO_URI)
    db = client['EnergyLearning']
    collection = db['weightedZoneResults']

    data = list(collection.find({}, {
        "_id": 0,
        "zoneKey": 1,
        "numEntries": 1,
        "weightedAvgTotalCo2Consumption": 1,
        "weightedAvgTotalConsumption": 1,
        "avgEstimatedPercentage": 1,
        "avgRenewableRatio": 1,
        "avgFossilFuelRatio": 1
    }))

    df = pd.DataFrame(data)
    return df

# Step 2:
def preprocess_data(df):
    X = df[['numEntries', 'weightedAvgTotalConsumption',
            'avgEstimatedPercentage', 'avgRenewableRatio', 'avgFossilFuelRatio']]
    y = df[['weightedAvgTotalCo2Consumption']]

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    return X_scaled, y_scaled

# Step 3:
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    return X_train, X_test, y_train, y_test

# Step 4:
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Step 5:
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R^2 Score: {r2}")

    return y_pred


def make_prediction(model, scaler, new_data):
    new_data_scaled = scaler.fit(new_data)
    prediction = model.predict([new_data_scaled])
    return prediction[0]


def plot_model_fit(y_test, y_pred):
    fig = go.Figure()


    y_test = np.ravel(y_test)
    y_pred = np.ravel(y_pred)

    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers',
                             name='Predictions vs Actual',
                             marker=dict(color='blue', size=10, opacity=0.7, line=dict(width=2, color='black'))))


    fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)],
                             mode='lines', name='Perfect Fit', line=dict(color='red', dash='dash')))

    fig.update_layout(title='Model Fit: Actual vs Predicted CO2 Consumption',
                      xaxis_title='Actual CO2 Consumption',
                      yaxis_title='Predicted CO2 Consumption',
                      template='plotly_dark')

    fig.show()

def main():

    df = fetch_data()

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    y_pred = evaluate_model(model, X_test, y_test)

    plot_model_fit(y_test, y_pred)


    print("\nPredicted vs Actual values (Test Set):")
    result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(result_df.head())

if __name__ == "__main__":
    main()