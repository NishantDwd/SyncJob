from JobRecommendation.config import client
import pandas as pd
import os
import sys

# Path to updated_cv.csv in JobRec/data/concatenated_data/
BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "data", "concatenated_data", "updated_cv.csv")

DB_NAME = "Job-Recomendation"
COLLECTION = "Resume_Data"  # same name used in Recruiter App

def load_csv_to_mongo(csv_path: str, db_name: str, collection_name: str, drop_existing: bool = True):
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Replace NaN with None for MongoDB
    records = df.where(pd.notnull(df), None).to_dict(orient="records")

    db = client[db_name]

    if drop_existing:
        db.drop_collection(collection_name)
        print(f"🗑️ Dropped existing collection: {db_name}.{collection_name}")

    if records:
        result = db[collection_name].insert_many(records)
        print(f"✅ Inserted {len(result.inserted_ids)} documents into {db_name}.{collection_name}")
    else:
        print("⚠️ No records found to insert.")

if __name__ == "__main__":
    load_csv_to_mongo(CSV_PATH, DB_NAME, COLLECTION)
