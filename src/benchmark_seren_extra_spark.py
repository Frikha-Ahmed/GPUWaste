import csv
import math
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ==========================================
# STEP 1 — START SPARK
# ==========================================

spark = SparkSession.builder.appName("Seren-Full-Benchmark").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

hdfs = "hdfs://namenode:8020/gpuwaste"


# ==========================================
# PART 1 — GPU POWER ANALYSIS
# ==========================================

start = time.perf_counter()

# Read power measurements from HDFS
lines = spark.sparkContext.textFile(
    f"{hdfs}/raw/seren/POWER_USAGE.csv"
)

# Identify the GPU columns
header = lines.first()
gpu_ids = next(csv.reader([header]))[1:]

# Remove the header
data = lines.filter(lambda row: row != header)


# Process one Spark partition
def analyze_partition(rows):

    # Statistics for every GPU:
    # valid count, missing count, invalid count, power sum
    stats = [[0, 0, 0, 0.0] for _ in gpu_ids]

    for row in csv.reader(rows):

        if len(row) != len(gpu_ids) + 1:
            raise ValueError("Invalid CSV row")

        for i, value in enumerate(row[1:]):

            if value == "":
                stats[i][1] += 1
                continue

            try:
                power = float(value)
            except ValueError:
                stats[i][2] += 1
                continue

            if not math.isfinite(power) or power < 0:
                stats[i][2] += 1
                continue

            stats[i][0] += 1
            stats[i][3] += power

    for i, gpu_id in enumerate(gpu_ids):
        yield gpu_id, tuple(stats[i])


# Combine partial statistics for the same GPU
def combine(a, b):
    return (
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
        a[3] + b[3]
    )


partial = data.mapPartitions(analyze_partition)
combined = partial.reduceByKey(combine)


# Convert results into a DataFrame
schema = """
    gpu_id STRING,
    valid_count LONG,
    missing_count LONG,
    invalid_count LONG,
    power_sum DOUBLE
"""

power_result = spark.createDataFrame(
    combined.map(lambda x: (x[0],) + x[1]),
    schema=schema
)

# Calculate average power
power_result = power_result.withColumn(
    "avg_power_w",
    F.when(
        F.col("valid_count") > 0,
        F.round(F.col("power_sum") / F.col("valid_count"), 2)
    )
).drop("power_sum")


# Save results to HDFS
power_result.write.mode("overwrite").parquet(
    f"{hdfs}/gold/benchmark_seren_power"
)

power_time = time.perf_counter() - start

print("Power analysis completed.")
print("GPUs analyzed:", len(gpu_ids))


# ==========================================
# PART 2 — JOB ANALYSIS
# ==========================================

start = time.perf_counter()

# Read Seren job history
jobs = spark.read.csv(
    f"{hdfs}/raw/seren/trace_seren.csv",
    header=True
)

# Convert the GPU request column into numbers
jobs = jobs.withColumn(
    "gpu_num",
    F.col("gpu_num").cast("double")
)

# Identify jobs requesting GPUs
gpu_job = (
    (F.col("gpu_num") > 0) &
    (F.col("gpu_num") < float("inf"))
)

# Calculate workload statistics
job_result = jobs.agg(
    F.count("*").alias("total_jobs"),

    F.sum(
        F.when(gpu_job, 1).otherwise(0)
    ).alias("gpu_jobs"),

    F.sum(
        F.when(gpu_job, F.col("gpu_num")).otherwise(0)
    ).alias("requested_gpu_sum")
)

# Save results to HDFS
job_result.write.mode("overwrite").parquet(
    f"{hdfs}/gold/benchmark_seren_jobs"
)

job_time = time.perf_counter() - start


# ==========================================
# PART 3 — DISPLAY EXECUTION TIMES
# ==========================================

print("\n=== SPARK RESULTS ===")

spark.read.parquet(
    f"{hdfs}/gold/benchmark_seren_jobs"
).show()

print(f"Power processing: {power_time:.2f} seconds")
print(f"Job processing:   {job_time:.2f} seconds")
print(f"Total:            {power_time + job_time:.2f} seconds")

spark.stop()
