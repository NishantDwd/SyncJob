# ...existing code...
import pymongo
import pandas as pd
import json 
from JobRecommendation.config import client
import os
# ...existing code...

db = client.test

# changed code
BASE_DIR = os.path.dirname(__file__)
DATA_FILE_PATH = os.path.join(BASE_DIR, "data", "concatenated_data", "all_locations.csv")
print("Resolved DATA_FILE_PATH:", DATA_FILE_PATH)
if not os.path.exists(DATA_FILE_PATH):
    raise FileNotFoundError(f"CSV not found at {DATA_FILE_PATH}. Place all_locations.csv there or update DATA_FILE_PATH.")

# Database Name
dataBase = "Job-Recomendation"
# Collection  Name
collection = "all_locations_Data"
if __name__=="__main__":
    df=pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and columns: {df.shape}")
    # reseting the index
    df.reset_index(drop=True,inplace=True)
    #Convert dataframe to json so that we can dump these record in mongo db
    json_record = list(json.loads(df.T.to_json()).values())# this is the required format
    #print(json_record[0])

    # insert converted json record to modgoDB
    client[dataBase][collection].insert_many(json_record)
