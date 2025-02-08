import os
import json
from pymongo import MongoClient


def populate_mongodb(directory, mongo_uri, db_name, collection_name):
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    data = json.load(file)

                    if isinstance(data, dict):
                        data = [data]

                    collection.insert_many(data)
                    print(f"Successfully inserted {len(data)} records from {filename}")
                except json.JSONDecodeError:
                    print(f"Error reading {filename}: Invalid JSON format")
                except Exception as e:
                    print(f"Error inserting data from {filename}: {e}")

    print("Data population complete.")


MONGO_USER = "rootarded"
MONGO_PASSWORD = "Yhs7a7gK"
MONGO_HOST = "localhost"
MONGO_PORT = "27017"
DB_NAME = "EnergyLearning"
COLLECTION_NAME = "MonthlyScans"
JSON_DIRECTORY = "../Data"


MONGO_URI = f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/{DB_NAME}"


populate_mongodb(JSON_DIRECTORY, MONGO_URI, DB_NAME, COLLECTION_NAME)
