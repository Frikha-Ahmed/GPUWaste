import os

from pymongo import MongoClient
from pyspark.sql import SparkSession


# ==========================================
# STEP 1: Start Spark
# ==========================================

spark = SparkSession.builder.appName("GPUWaste-MongoDB").getOrCreate()

spark.sparkContext.setLogLevel("WARN")


# ==========================================
# STEP 2: Connect to MongoDB
# ==========================================

# Read credentials from the container environment
username = os.environ["MONGO_INITDB_ROOT_USERNAME"]
password = os.environ["MONGO_INITDB_ROOT_PASSWORD"]

# Connect to MongoDB through the Docker network
client = MongoClient(
    host="mongodb",
    port=27017,
    username=username,
    password=password,
    authSource="admin"
)

# Verify that the database is reachable
client.admin.command("ping")

# Select our database
db = client["gpuwaste"]


# ==========================================
# STEP 3: Define the datasets to export
# ==========================================

# HDFS folder name : MongoDB collection name
datasets = {
    "job_summary": "job_summary",
    "gpu_utilization_summary": "gpu_utilization",
    "gpu_hourly": "gpu_hourly",
    "gpu_low_utilization_candidates": "gpu_candidates",
    "gpu_power_summary": "gpu_power",
    "gpu_efficiency": "gpu_efficiency",
    "gpu_temporal_analysis": "gpu_temporal"
}

hdfs_base = "hdfs://namenode:8020/gpuwaste/gold"


# ==========================================
# STEP 4: Transfer each dataset to MongoDB
# ==========================================

for folder, collection_name in datasets.items():

    # Read the analytical results from HDFS
    path = f"{hdfs_base}/{folder}"
    df = spark.read.parquet(path)

    # These are small summaries, not the original big datasets.
    # Refuse to collect unexpectedly large datasets.
    rows = df.limit(10001).collect()

    if len(rows) > 10000:
        raise ValueError(f"Dataset too large to collect: {folder}")

    # Convert Spark rows into ordinary Python dictionaries
    documents = [row.asDict(recursive=True) for row in rows]

    # Select the MongoDB collection
    collection = db[collection_name]

    # Remove previous results to avoid duplicates
    collection.delete_many({})

    # Insert the updated results
    if documents:
        collection.insert_many(documents)

    print(f"{collection_name}: {len(documents)} documents inserted")


# ==========================================
# STEP 5: Create indexes for frequent queries
# ==========================================

# Faster lookups using a GPU identifier
db["gpu_efficiency"].create_index("gpu_id")
db["gpu_temporal"].create_index("gpu_id")

# Faster retrieval of hourly data
db["gpu_hourly"].create_index("hour")


# ==========================================
# STEP 6: Close connections
# ==========================================

client.close()
spark.stop()

print("\nGPUWaste: MongoDB export completed!")
