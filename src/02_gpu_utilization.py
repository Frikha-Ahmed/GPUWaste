import csv
import math

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# 1. Start Spark
spark = (
    SparkSession.builder
    .appName("GPUWaste-GPU-Utilization")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# 2. Read the GPU data from HDFS
input_path = "hdfs://namenode:8020/gpuwaste/raw/kalos/GPU_UTIL.csv"

lines = spark.sparkContext.textFile(input_path)

header = lines.first()
gpu_ids = next(csv.reader([header]))[1:]

print(f"Number of GPUs: {len(gpu_ids)}")


# 3. Calculate statistics within each partition
def analyze_partition(rows):

    # For every GPU: valid count, low count,
    # missing count, invalid count, utilization sum
    stats = [[0, 0, 0, 0, 0.0] for _ in gpu_ids]

    for row in csv.reader(rows):

        if len(row) != len(gpu_ids) + 1:
            raise ValueError("Unexpected number of CSV columns")

        for i, value in enumerate(row[1:]):

            if value.strip() == "":
                stats[i][2] += 1
                continue

            try:
                utilization = float(value)
            except ValueError:
                stats[i][3] += 1
                continue

            if not math.isfinite(utilization) or not 0 <= utilization <= 100:
                stats[i][3] += 1
                continue

            stats[i][0] += 1
            stats[i][4] += utilization

            if utilization < 10:
                stats[i][1] += 1

    for gpu_id, values in zip(gpu_ids, stats):
        yield gpu_id, tuple(values)


# 4. Combine results from different Spark partitions
data = lines.filter(lambda line: line != header)

partial = data.mapPartitions(analyze_partition)

combined = partial.reduceByKey(
    lambda a, b: tuple(x + y for x, y in zip(a, b))
)


# 5. Convert results to a Spark DataFrame
schema = """
    gpu_id STRING,
    valid_count LONG,
    low_count LONG,
    missing_count LONG,
    invalid_count LONG,
    utilization_sum DOUBLE
"""

df = spark.createDataFrame(
    combined.map(lambda item: (item[0],) + item[1]),
    schema=schema
)


# 6. Calculate utilization percentages
result = (
    df.withColumn(
        "avg_utilization",
        F.when(
            F.col("valid_count") > 0,
            F.round(
                F.col("utilization_sum") / F.col("valid_count"),
                2
            )
        )
    )
    .withColumn(
        "low_utilization_pct",
        F.when(
            F.col("valid_count") > 0,
            F.round(
                100 * F.col("low_count") / F.col("valid_count"),
                2
            )
        )
    )
    .drop("utilization_sum")
)

result.cache()


# 7. Save results to HDFS
output_path = (
    "hdfs://namenode:8020/"
    "gpuwaste/gold/gpu_utilization_summary"
)

result.write.mode("overwrite").parquet(output_path)


# 8. Display results
print("\n=== GPU UTILIZATION ANALYSIS ===")

result.orderBy(
    F.asc_nulls_last("avg_utilization")
).show(15, truncate=False)

spark.stop()
