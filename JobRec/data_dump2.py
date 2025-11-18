# ...existing code...
from JobRecommendation.config import client
import pandas as pd
import os
import sys

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "concatenated_data", "preprocessed_jobs.csv")
DB_NAME = "Job-Recomendation"
COLLECTION = "preprocessed_jobs_Data"

def load_csv_to_mongo(csv_path: str, db_name: str, collection_name: str):
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    # Ensure no pandas NaN => None
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    # Optionally clear existing collection
    client[db_name].drop_collection(collection_name)
    if records:
        res = client[db_name][collection_name].insert_many(records)
        print(f"Inserted {len(res.inserted_ids)} documents into {db_name}.{collection_name}")
    else:
        print("No records to insert.")

if __name__ == "__main__":
    load_csv_to_mongo(DATA_FILE, DB_NAME, COLLECTION)
# ...existing code...