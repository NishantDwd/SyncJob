from JobRecommendation.config import client
import pandas as pd
import os
import sys

# Path to updated_cv.csv in JobRec/data/concatenated_data/
BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "data", "concatenated_data", "updated_cv.csv")

DB_NAME = "Job-Recomendation"
COLLECTION = "Resume_Data"

def load_csv_to_mongo(csv_path: str, db_name: str, collection_name: str, drop_existing: bool = True):
    """
    Load resume CSV data into MongoDB, ensuring proper format for the recruiter app.
    """
    print(f"📂 Reading CSV from: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} records from CSV")
    
    # Show columns
    print(f"📊 Columns: {list(df.columns)}")
    
    # Ensure required columns exist
    required_cols = ['name', 'email', 'mobile_number', 'skills', 'degree', 'no_of_pages', 'All']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️ Missing columns: {missing_cols}")
        print("Adding placeholder values for missing columns...")
        for col in missing_cols:
            if col == 'All':
                df['All'] = df['skills'].astype(str) + " " + df.get('degree', '').astype(str)
            else:
                df[col] = None
    
    # Create cv_id if not exists
    if 'cv_id' not in df.columns:
        if 'Unnamed: 0' in df.columns:
            df['cv_id'] = df['Unnamed: 0']
        else:
            df['cv_id'] = range(len(df))
    
    # Clean data
    df = df.fillna('Not Provided')
    
    # Remove duplicates based on email
    if 'email' in df.columns:
        df = df.drop_duplicates(subset=['email'], keep='first')
        print(f"✅ After removing duplicates: {len(df)} records")
    
    # Convert to dict
    data_dict = df.to_dict(orient='records')
    
    # Connect to MongoDB
    try:
        db = client[db_name]
        collection = db[collection_name]
        
        if drop_existing:
            print(f"🗑️ Dropping existing collection: {collection_name}")
            collection.drop()
        
        if len(data_dict) > 0:
            print(f"📤 Inserting {len(data_dict)} records into MongoDB...")
            result = collection.insert_many(data_dict)
            print(f"✅ Successfully inserted {len(result.inserted_ids)} records")
            print(f"✅ Data loaded into database: {db_name}, collection: {collection_name}")
            
            # Verify
            count = collection.count_documents({})
            print(f"📊 Total documents in collection: {count}")
            
            # Show sample
            sample = collection.find_one()
            if sample:
                print(f"📄 Sample document keys: {list(sample.keys())}")
        else:
            print("⚠️ No data to insert")
    
    except Exception as e:
        print(f"❌ Error loading data to MongoDB: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting Resume Data Upload to MongoDB")
    print("=" * 60)
    load_csv_to_mongo(CSV_PATH, DB_NAME, COLLECTION)
    print("=" * 60)
    print("✅ Process completed!")
    print("=" * 60)