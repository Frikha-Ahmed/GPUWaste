from pathlib import Path
from urllib.parse import quote_plus
import json

import pandas as pd
from dotenv import dotenv_values
from pymongo import MongoClient


# ---------------------------------------
# 1. MongoDB connection
# ---------------------------------------

project = Path(__file__).resolve().parent.parent

# Read credentials without displaying them
config = dotenv_values(project / "docker/mongodb.env")

username = config["MONGO_INITDB_ROOT_USERNAME"]
password = config["MONGO_INITDB_ROOT_PASSWORD"]

uri = (
    f"mongodb://{quote_plus(username)}:"
    f"{quote_plus(password)}@localhost:27017/"
    "?authSource=admin"
)

client = MongoClient(uri, serverSelectionTimeoutMS=5000)
client.admin.command("ping")

# Find the database containing our existing Kalos data
databases = [
    name
    for name in client.list_database_names()
    if "gpu_efficiency" in client[name].list_collection_names()
]

if len(databases) != 1:
    raise RuntimeError(
        "Cannot uniquely identify the existing Kalos database."
    )

db = client[databases[0]]

print("Connected to database:", db.name)


# ---------------------------------------
# 2. Seren collections to import
# ---------------------------------------

collections = [
    "seren_gpu_utilization",
    "seren_gpu_power",
    "seren_gpu_candidates",
    "seren_gpu_efficiency",
    "seren_gpu_hourly",
    "seren_job_summary",
]


# ---------------------------------------
# 3. Read Parquet and export to MongoDB
# ---------------------------------------

for name in collections:

    path = project / "data/seren_export" / name

    # Read the small analytical summary
    df = pd.read_parquet(path)

    # Convert Pandas values into JSON-compatible documents
    documents = json.loads(
        df.to_json(orient="records", date_format="iso")
    )

    # Only clear the Seren collection.
    # Existing Kalos collections remain untouched.
    collection = db[name]
    collection.delete_many({})

    if documents:
        collection.insert_many(documents)

        if "gpu_id" in documents[0]:
            collection.create_index("gpu_id")

    print(f"{name}: {len(documents)} documents")


# ---------------------------------------
# 4. Finish
# ---------------------------------------

client.close()

print("\nSeren successfully exported to MongoDB!")
